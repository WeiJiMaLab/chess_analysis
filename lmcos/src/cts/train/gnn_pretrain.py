"""Encoder pretraining training loop.

Extracted out of ``cts_pretrain.py`` as part of the migration to the
``cts`` package layout. Holds the configuration, metrics, and training
loop that fit ``ChildWdlModel`` against the per-edge WDL targets produced
by :mod:`cts.data.preprocess_gnn.teacher_targets`.

Encoder checkpoint save/load helpers live here too because they're used
by both this trainer and downstream consumers loading the trained encoder
for controller training and analysis.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cts.core.schema import (
    TOPOLOGY_TARGET_SCALING_LOG1P_VISITS_NODES_BELOW,
    NodeFeatureSchema,
    topology_target_feature_names,
    tree_encoder_feature_schema,
)
from cts.core.tensorizer import (
    TensorizedTreeExample,
    collate_tensorized_examples,
    edge_wdl_target_tensor,
    tensorize_forest,
)
from cts.data.preprocess_gnn.teacher_targets import PackedTensorizedShardDataset, PretrainExample
from cts.models.gnn import ChildWdlModel, TopologyModel


def save_encoder_checkpoint(path: str, encoder, metadata: Optional[Mapping[str, Any]] = None) -> None:
    """Save the encoder's state dict + its architecture + any caller-supplied metadata.

    Architecture is read off the encoder module so that loaders can
    reconstruct an identically-shaped module without the caller having to
    re-specify every hyperparameter. The arch dict lives under the stable
    ``encoder_architecture`` key inside ``metadata``; user-supplied keys
    are merged on top.
    """
    architecture = {
        "k": int(encoder.k),
        "d_embed": int(encoder.d_embed),
        "d_message": int(encoder.d_message),
        "n_heads": int(encoder.upward_msg.n_heads),
        "d_att": int(encoder.upward_msg.d_att),
        "node_embed_hidden": int(encoder.node_embed[0].out_features),
        "node_feat": int(encoder.node_feat),
    }
    merged_metadata = {"encoder_architecture": architecture}
    if metadata:
        merged_metadata.update(metadata)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "metadata": merged_metadata,
        },
        path,
    )


def load_encoder_checkpoint(path: str, encoder, map_location: str = "cpu") -> Dict[str, Any]:
    """Load an encoder checkpoint into ``encoder`` in-place. Returns the stored metadata."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return dict(checkpoint.get("metadata", {}))


def load_encoder_architecture(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Return just the architecture dict embedded in an encoder checkpoint.

    Used by downstream consumers (controller materialization, evaluation)
    that need to reconstruct an identically-shaped encoder before loading
    the weights. Fails loudly if the checkpoint pre-dates the architecture
    metadata convention — those checkpoints need to be re-saved with the
    current ``save_encoder_checkpoint`` before they can be used.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    metadata = checkpoint.get("metadata") or {}
    architecture = metadata.get("encoder_architecture")
    if not architecture:
        raise ValueError(
            f"Encoder checkpoint {path!r} has no ``encoder_architecture`` block in metadata. "
            "Re-save it with the current save_encoder_checkpoint() before loading."
        )
    return dict(architecture)

@dataclass(frozen=True)
class ChildWdlPretrainConfig:
    """Hyperparameters for the child-WDL supervised pretraining loop."""

    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 5
    shuffle: bool = True
    num_workers: int = 0  # DataLoader workers; 0 means in-process
    pin_memory: bool = False
    prefetch_factor: int = 2  # only used when num_workers > 0
    persistent_workers: bool = True  # only used when num_workers > 0


@dataclass
class ChildWdlMetrics:
    """Aggregated metrics for one pass (epoch's worth) over a dataset split."""

    total_loss: float  # cross-entropy averaged over supervised edges
    num_examples: int  # number of (tree) examples seen
    num_supervised_edges: int  # number of edges with WDL targets that contributed to the loss
    target_entropy: float = 0.0  # entropy of the target distribution (lower bound on total_loss)
    loss_gap: float = 0.0  # total_loss - target_entropy; the "real" KL we're driving toward zero


class ChildWdlPretrainer:
    """Supervised pretraining loop for ``ChildWdlModel`` against per-edge WDL targets.

    Holds the model, the feature schema + tensorizer device, the train/validation
    datasets, and the best-so-far encoder/decoder states (updated each epoch when
    validation improves). The loop is plain-vanilla cross-entropy with Adam;
    ``fit`` is the public entry point.
    """

    def __init__(
        self,
        model: ChildWdlModel,
        schema: NodeFeatureSchema,
        device: torch.device | str,
        train_examples: Sequence[PretrainExample],
        validation_examples: Sequence[PretrainExample],
        config: ChildWdlPretrainConfig,
    ) -> None:
        self.model = model
        self.schema = schema
        self.device = torch.device(device)
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # Track best-validation encoder + decoder snapshots for final restore.
        self.best_validation_loss = float("inf")
        self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
        self.best_decoder_state = copy.deepcopy(self.model.child_wdl_head.state_dict())

    def _iter_batches(self, examples: Sequence[PretrainExample], shuffle: bool):
        """Build a DataLoader over ``examples`` using the dataset's collate_fn if present."""
        # Packed datasets expose a custom collate_fn; per-example datasets
        # rely on the default identity-list collation.
        collate_fn = getattr(examples, "collate_fn", list)
        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            dataloader_kwargs["persistent_workers"] = self.config.persistent_workers
        return DataLoader(examples, **dataloader_kwargs)

    def _edge_target_tensor(self, examples: Sequence[PretrainExample], device: torch.device) -> torch.Tensor:
        """Stack per-edge WDL targets across a list of raw examples, in canonical edge order."""
        targets = []
        for example in examples:
            if not example.edge_wdl_targets:
                raise ValueError(
                    "Child-WDL pretraining requires raw examples with search-consolidated edge_wdl_targets."
                )
            targets.append(
                edge_wdl_target_tensor(
                    example.tree,
                    example.edge_wdl_targets,
                    schema=self.schema,
                    device=self.device,
                )
            )
        if not targets:
            return torch.empty((0, 3), dtype=torch.float32, device=device)
        return torch.cat(targets, dim=0).to(device)

    def _tree_batch_and_edge_targets(self, batch_data, device: torch.device) -> Tuple[Any, torch.Tensor]:
        """Normalize a batch to ``(tree_batch, edge_targets)`` regardless of source dataset type.

        Packed datasets emit ``(tree_batch, node_targets)`` pairs directly;
        raw datasets emit a list of examples that we still need to tensorize.
        """
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            tree_batch, _node_targets = batch_data
            if tree_batch.edge_wdl_targets is None:
                raise ValueError(
                    "Packed tensorized child-WDL batches require edge_wdl_targets; repack the dataset with edge targets."
                )
            return tree_batch, tree_batch.edge_wdl_targets.to(device)

        batch_examples = batch_data
        tree_batch = tensorize_forest(
            [example.tree for example in batch_examples],
            schema=self.schema,
            device=self.device,
        )
        edge_targets = self._edge_target_tensor(batch_examples, device=device)
        return tree_batch, edge_targets

    def _prepare_epoch(
        self,
        examples: Sequence[PretrainExample],
        training: bool,
    ) -> tuple[DataLoader, int, str, torch.device, torch.Tensor, torch.Tensor]:
        """Set the model to train/eval, build the batch iterator, and initialize accumulators.

        Returns ``(batches, total_batches, phase, metric_device, total_loss_sum, target_entropy_sum)``.
        Sums are on-device tensors so the per-batch ``+=`` doesn't trigger a host sync.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        metric_device = self.model.encoder.device
        # Accumulate sums on-device so we don't pay host syncs per batch.
        total_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        target_entropy_sum = torch.zeros((), dtype=torch.float32, device=metric_device)

        batches = self._iter_batches(examples, shuffle=training and self.config.shuffle)
        total_batches = len(batches)
        phase = "train" if training else "validation"
        return batches, total_batches, phase, metric_device, total_loss_sum, target_entropy_sum

    def _iterate_supervised_edge_batches(
        self,
        batches: DataLoader,
        total_batches: int,
        phase: str,
        training: bool,
        epoch_index: int,
        total_loss_sum: torch.Tensor,
        target_entropy_sum: torch.Tensor,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        """Stream batches, compute per-edge cross-entropy, step the optimizer when training.

        Updates and returns the running accumulators ``(total_examples,
        total_supervised_edges, total_loss_sum, target_entropy_sum)``.
        """
        total_examples = 0
        total_supervised_edges = 0
        for batch_index, batch_data in enumerate(batches, start=1):
            tree_batch, edge_targets = self._tree_batch_and_edge_targets(batch_data, device=self.model.encoder.device)
            if edge_targets.numel() == 0:
                continue

            # Forward + (optional) backward.
            with torch.set_grad_enabled(training):
                edge_logits = self.model(tree_batch)
                log_probs = F.log_softmax(edge_logits, dim=-1)
                per_edge_loss = -(edge_targets * log_probs).sum(dim=-1)
                total_loss = per_edge_loss.mean()
                # Target entropy is a constant of the data; tracking it
                # gives the loss_gap, which is the part of the loss we can
                # actually drive down.
                target_log_probs = edge_targets.clamp_min(1e-12).log()
                per_edge_target_entropy = -(edge_targets * target_log_probs).sum(dim=-1)
                target_entropy = per_edge_target_entropy.mean()
                loss_gap = total_loss - target_entropy

                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            # Bookkeeping. We weight the running means by edge count rather
            # than batch count so micro-batches with few edges don't get
            # the same weight as full batches.
            batch_size = int(tree_batch.batch_size)
            num_supervised_edges = int(edge_targets.shape[0])
            total_examples += batch_size
            total_supervised_edges += num_supervised_edges
            total_loss_sum += total_loss.detach() * num_supervised_edges
            target_entropy_sum += target_entropy.detach() * num_supervised_edges
            if batch_progress_callback is not None:
                batch_progress_callback(
                    epoch_index,
                    phase,
                    batch_index,
                    total_batches,
                    total_examples,
                    total_supervised_edges,
                    float(total_loss.item()),
                    float(target_entropy.item()),
                    float(loss_gap.item()),
                )
        return total_examples, total_supervised_edges, total_loss_sum, target_entropy_sum

    def _finalize_epoch_metrics(
        self,
        total_examples: int,
        total_supervised_edges: int,
        total_loss_sum: torch.Tensor,
        target_entropy_sum: torch.Tensor,
    ) -> ChildWdlMetrics:
        """Reduce the on-device accumulator sums into a per-edge-averaged ``ChildWdlMetrics``."""
        if total_supervised_edges == 0:
            return ChildWdlMetrics(total_loss=0.0, num_examples=total_examples, num_supervised_edges=0)

        mean_total_loss = float((total_loss_sum / total_supervised_edges).item())
        mean_target_entropy = float((target_entropy_sum / total_supervised_edges).item())
        return ChildWdlMetrics(
            total_loss=mean_total_loss,
            num_examples=total_examples,
            num_supervised_edges=total_supervised_edges,
            target_entropy=mean_target_entropy,
            loss_gap=mean_total_loss - mean_target_entropy,
        )

    def _run_epoch(
        self,
        examples: Sequence[PretrainExample],
        epoch_index: int,
        training: bool,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> ChildWdlMetrics:
        """Run one full pass over ``examples`` in either training or eval mode.

        Computes cross-entropy of the predicted child-WDL distribution
        against the teacher target on every supervised edge, optionally
        steps the optimizer, and reports aggregated metrics + per-batch
        progress via the callback.
        """
        batches, total_batches, phase, _, total_loss_sum, target_entropy_sum = self._prepare_epoch(
            examples,
            training,
        )
        total_examples, total_supervised_edges, total_loss_sum, target_entropy_sum = (
            self._iterate_supervised_edge_batches(
                batches,
                total_batches,
                phase,
                training,
                epoch_index,
                total_loss_sum,
                target_entropy_sum,
                batch_progress_callback=batch_progress_callback,
            )
        )
        return self._finalize_epoch_metrics(
            total_examples,
            total_supervised_edges,
            total_loss_sum,
            target_entropy_sum,
        )

    def train_epoch(
        self,
        epoch_index: int = 1,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> ChildWdlMetrics:
        """Run one training epoch over ``self.train_examples``."""
        return self._run_epoch(
            self.train_examples,
            epoch_index,
            training=True,
            batch_progress_callback=batch_progress_callback,
        )

    def validate(
        self,
        epoch_index: int = 1,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> ChildWdlMetrics:
        """Run one validation pass over ``self.validation_examples`` (no grad, no shuffle)."""
        return self._run_epoch(
            self.validation_examples,
            epoch_index,
            training=False,
            batch_progress_callback=batch_progress_callback,
        )

    def save_training_state(self, path: str, epoch: int) -> None:
        """Save model+optimizer+best-snapshot state so training can be resumed."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_encoder_state": self.best_encoder_state,
            "best_decoder_state": self.best_decoder_state,
            "best_validation_loss": self.best_validation_loss,
            "epoch": epoch,
        }, path)

    def load_training_state(self, path: str) -> int:
        """Restore model+optimizer+best-snapshot state. Returns the saved epoch index."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_encoder_state = state["best_encoder_state"]
        self.best_decoder_state = state["best_decoder_state"]
        self.best_validation_loss = state["best_validation_loss"]
        return int(state["epoch"])

    def fit(
        self,
        progress_callback: Optional[Callable[[int, ChildWdlMetrics, ChildWdlMetrics], None]] = None,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
        start_epoch: int = 1,
        resume_path: Optional[str] = None,
    ) -> List[Dict[str, ChildWdlMetrics]]:
        """Drive the full epoch loop, tracking best-validation weights.

        On every epoch where validation improves, snapshot the encoder and
        decoder state dicts. After the last epoch, restore the model in
        place to those best snapshots and return the per-epoch metrics
        history.

        Args:
            progress_callback: optional per-epoch reporter; gets ``(epoch, train, validation)``.
            batch_progress_callback: optional per-batch reporter; forwarded into ``_run_epoch``.
            start_epoch: starting epoch index (1-based); used when resuming.
            resume_path: if given, the full training state is checkpointed
                to this path after every epoch so a crash can resume.
        """
        history: List[Dict[str, ChildWdlMetrics]] = []
        for epoch_index in range(start_epoch, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch_index, batch_progress_callback=batch_progress_callback)
            validation_metrics = self.validate(epoch_index, batch_progress_callback=batch_progress_callback)
            history.append({"train": train_metrics, "validation": validation_metrics})
            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
                self.best_decoder_state = copy.deepcopy(self.model.child_wdl_head.state_dict())
            if progress_callback is not None:
                progress_callback(epoch_index, train_metrics, validation_metrics)
            if resume_path is not None:
                self.save_training_state(resume_path, epoch_index)
        # Restore best-validation weights before returning.
        self.model.encoder.load_state_dict(self.best_encoder_state)
        self.model.child_wdl_head.load_state_dict(self.best_decoder_state)
        return history

    def save_best_encoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Save just the encoder state at ``self.model.encoder`` (which holds best-validation weights after ``fit``)."""
        save_encoder_checkpoint(path, self.model.encoder, metadata=metadata)

    def save_best_decoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Save just the child-WDL decoder head's state. Mirrors ``save_best_encoder``."""
        torch.save({
            "decoder_state_dict": self.model.child_wdl_head.state_dict(),
            "metadata": dict(metadata or {}),
        }, path)


@dataclass(frozen=True)
class TopologyPretrainConfig:
    """Dataloader + optimizer settings for ``TopologyModel`` (Huber/MSE vs four teacher scalars)."""

    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 5
    loss_type: Literal["huber", "mse"] = "huber"
    huber_delta: float = 1.0
    topology_target_weights: Optional[Tuple[float, float, float, float]] = None
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class TopologyMetrics:
    """Aggregated metrics for one topology-pretraining pass."""

    total_loss: float
    num_examples: int
    num_supervised_nodes: int


class TopologyPretrainer:
    """Train ``TopologyModel`` on packed shards — encoder + ``TopologyHead``, regression only."""

    def __init__(
        self,
        model: TopologyModel,
        device: torch.device | str,
        train_examples: Sequence[Any],
        validation_examples: Sequence[Any],
        config: TopologyPretrainConfig,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_validation_loss = float("inf")
        self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
        self.best_topology_head_state = copy.deepcopy(self.model.topology_head.state_dict())

    def _iter_batches(self, examples: Sequence[Any], shuffle: bool) -> DataLoader:
        collate_fn = PackedTensorizedShardDataset.collate_fn_topology
        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            dataloader_kwargs["persistent_workers"] = self.config.persistent_workers
        return DataLoader(examples, **dataloader_kwargs)

    def _regression_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "mse":
            per_example = (predictions - targets) ** 2
        else:
            per_example = F.huber_loss(predictions, targets, delta=self.config.huber_delta, reduction="none")
        weights = self.config.topology_target_weights
        if weights is None:
            return per_example.mean()
        if len(weights) != int(predictions.shape[-1]):
            raise ValueError(
                f"topology_target_weights length {len(weights)} != prediction width {predictions.shape[-1]}."
            )
        w_tensor = predictions.new_tensor(weights, dtype=per_example.dtype)
        denom = w_tensor.sum().clamp_min(torch.finfo(per_example.dtype).eps)
        per_node = (per_example * w_tensor.unsqueeze(0)).sum(dim=-1) / denom
        return per_node.mean()

    def _run_epoch(
        self,
        examples: Sequence[Any],
        *,
        training: bool,
        epoch_index: int,
        batch_progress_callback: Optional[
            Callable[[int, str, int, int, int, int, float], None]
        ] = None,
    ) -> TopologyMetrics:
        if training:
            self.model.train()
        else:
            self.model.eval()

        metric_device = self.model.encoder.device
        total_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        total_examples = 0
        total_supervised_nodes = 0

        batches = self._iter_batches(examples, shuffle=training and self.config.shuffle)
        total_batches = len(batches)
        phase = "train" if training else "validation"

        for batch_index, (tree_batch, topology_targets) in enumerate(batches, start=1):
            topology_targets = topology_targets.to(metric_device)
            with torch.set_grad_enabled(training):
                predictions = self.model(tree_batch)
                total_loss = self._regression_loss(predictions, topology_targets)
                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            num_nodes = int(topology_targets.shape[0])
            batch_size = int(tree_batch.batch_size)
            total_examples += batch_size
            total_supervised_nodes += num_nodes
            total_loss_sum += total_loss.detach() * num_nodes

            if batch_progress_callback is not None:
                batch_progress_callback(
                    epoch_index,
                    phase,
                    batch_index,
                    total_batches,
                    total_examples,
                    total_supervised_nodes,
                    float(total_loss.item()),
                )

        if total_supervised_nodes == 0:
            return TopologyMetrics(total_loss=0.0, num_examples=total_examples, num_supervised_nodes=0)
        mean_loss = float((total_loss_sum / total_supervised_nodes).item())
        return TopologyMetrics(
            total_loss=mean_loss,
            num_examples=total_examples,
            num_supervised_nodes=total_supervised_nodes,
        )

    def train_epoch(
        self,
        epoch_index: int,
        batch_progress_callback: Optional[
            Callable[[int, str, int, int, int, int, float], None]
        ] = None,
    ) -> TopologyMetrics:
        return self._run_epoch(
            self.train_examples,
            training=True,
            epoch_index=epoch_index,
            batch_progress_callback=batch_progress_callback,
        )

    def validate(
        self,
        epoch_index: int,
        batch_progress_callback: Optional[
            Callable[[int, str, int, int, int, int, float], None]
        ] = None,
    ) -> TopologyMetrics:
        return self._run_epoch(
            self.validation_examples,
            training=False,
            epoch_index=epoch_index,
            batch_progress_callback=batch_progress_callback,
        )

    def fit(
        self,
        progress_callback: Optional[Callable[[int, TopologyMetrics, TopologyMetrics], None]] = None,
        batch_progress_callback: Optional[
            Callable[[int, str, int, int, int, int, float], None]
        ] = None,
        start_epoch: int = 1,
        resume_path: Optional[str] = None,
    ) -> List[Dict[str, TopologyMetrics]]:
        history: List[Dict[str, TopologyMetrics]] = []
        for epoch_index in range(start_epoch, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch_index, batch_progress_callback=batch_progress_callback)
            validation_metrics = self.validate(epoch_index, batch_progress_callback=batch_progress_callback)
            history.append({"train": train_metrics, "validation": validation_metrics})
            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
                self.best_topology_head_state = copy.deepcopy(self.model.topology_head.state_dict())
            if progress_callback is not None:
                progress_callback(epoch_index, train_metrics, validation_metrics)
            if resume_path is not None:
                torch.save(
                    {
                        "epoch": epoch_index,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_encoder_state": self.best_encoder_state,
                        "best_topology_head_state": self.best_topology_head_state,
                        "best_validation_loss": self.best_validation_loss,
                    },
                    resume_path,
                )
        self.model.encoder.load_state_dict(self.best_encoder_state)
        self.model.topology_head.load_state_dict(self.best_topology_head_state)
        return history

    def load_training_state(self, path: str) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_encoder_state = state["best_encoder_state"]
        self.best_topology_head_state = state["best_topology_head_state"]
        self.best_validation_loss = state["best_validation_loss"]
        return int(state["epoch"])

    def save_best_encoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        merged = {
            "pretrain_objective": "topology",
            "topology_target_scaling": TOPOLOGY_TARGET_SCALING_LOG1P_VISITS_NODES_BELOW,
            "topology_target_columns": list(topology_target_feature_names()),
        }
        if self.config.topology_target_weights is not None:
            merged["topology_target_weights"] = list(self.config.topology_target_weights)
        if metadata:
            merged.update(metadata)
        save_encoder_checkpoint(path, self.model.encoder, metadata=merged)

    def save_best_topology_head(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        torch.save(
            {
                "topology_head_state_dict": self.model.topology_head.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )
