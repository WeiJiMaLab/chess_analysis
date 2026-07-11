from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cts.core.kl_buckets import (
    compute_subtree_sizes,
    depth_bin,
    depth_bin_labels,
    size_bin,
    size_bin_labels,
)
from cts.core.schema import NodeFeatureSchema
from cts.core.tensorizer import edge_wdl_target_tensor
from cts.data.preprocess_gnn.teacher_targets import PretrainExample
from cts.models.gnn import ChildWdlModel


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
    # When set, validation also accumulates per-edge KL into a (size, depth)
    # grid, exposed alongside the standard metrics for offline plotting.
    log_bucketed_kl: bool = False
    bucket_max_depth_bin: int = 12
    bucket_subtree_size_log_max: int = 10  # ≥1024 top bin; covers chess trees up to ~2000 nodes
    # BucketedKLState's size axis: subtree size (default) or, for the
    # k-steps-ahead objective, Delta-visits (TreeBatch.edge_visit_weights).
    bucket_by: Literal["subtree_size", "visit_weight"] = "subtree_size"
    # Weight each edge's cross-entropy by its child's subtree size before
    # averaging, so large-subtree children aren't drowned out by the ~97% of
    # edges that are leaves.
    loss_weight_by_subtree_size: bool = False
    # For k-steps-ahead batches carrying a per-edge Delta-visits weight instead
    # of subtree size: "identity" uses it directly, "log1p" dampens outliers.
    visit_weight_transform: Literal["identity", "log1p"] = "identity"


@dataclass
class ChildWdlMetrics:
    """Aggregated metrics for one pass (epoch's worth) over a dataset split."""

    total_loss: float  # cross-entropy averaged over supervised edges
    num_examples: int  # number of (tree) examples seen
    num_supervised_edges: int  # number of edges with WDL targets that contributed to the loss
    target_entropy: float = 0.0  # entropy of the target distribution (lower bound on total_loss)
    loss_gap: float = 0.0  # total_loss - target_entropy; the "real" KL we're driving toward zero


@dataclass
class BucketedKLState:
    """Per-validation-epoch accumulator for per-edge KL bucketed by structure.

    Mirrors the audit module's grid (rows = child-subtree-size log₂ bins,
    cols = parent-depth bins) so the time series during pretraining is
    directly comparable to the post-hoc audit numbers. Updated per batch
    via ``update`` and reduced to a JSON-serializable dict via ``summary``.
    """

    max_depth_bin: int
    subtree_size_log_max: int
    sum_kl: torch.Tensor  # [num_size_bins, num_depth_bins] float64
    count: torch.Tensor  # [num_size_bins, num_depth_bins] long
    overall_sum_kl: torch.Tensor  # scalar float64
    overall_count: int = 0
    # "subtree_size" (default, matches the audit module) or "visit_weight"
    # (TreeBatch.edge_visit_weights, for the k-steps-ahead objective).
    bucket_by: Literal["subtree_size", "visit_weight"] = "subtree_size"

    @classmethod
    def empty(
        cls,
        max_depth_bin: int,
        subtree_size_log_max: int,
        device: torch.device,
        bucket_by: Literal["subtree_size", "visit_weight"] = "subtree_size",
    ) -> "BucketedKLState":
        """Allocate zeroed accumulators sized to the given bin grid."""
        num_size_bins = subtree_size_log_max + 1
        num_depth_bins = max_depth_bin + 1
        return cls(
            max_depth_bin=int(max_depth_bin),
            subtree_size_log_max=int(subtree_size_log_max),
            sum_kl=torch.zeros((num_size_bins, num_depth_bins), dtype=torch.float64, device=device),
            count=torch.zeros((num_size_bins, num_depth_bins), dtype=torch.long, device=device),
            overall_sum_kl=torch.zeros((), dtype=torch.float64, device=device),
            overall_count=0,
            bucket_by=bucket_by,
        )

    def update(self, tree_batch: Any, per_edge_kl: torch.Tensor) -> None:
        """Bucket ``per_edge_kl`` by (child_subtree_size_or_visit_weight, parent_depth) and accumulate.

        Uses the same ``size_bin``/``depth_bin`` floor(log2(x)) bucketing helpers
        the audit module uses (subtree size and Delta-visits are both non-negative
        counts spanning orders of magnitude, so the same bucketing mechanism
        applies to either), so the per-bin numbers stay comparable across runs
        that use the same ``bucket_by``.
        """
        if per_edge_kl.numel() == 0:
            return
        device = self.sum_kl.device
        edge_parent = tree_batch.edge_parent.to(device)
        depth_all = tree_batch.depth.to(device)

        if self.bucket_by == "visit_weight":
            edge_visit_weights = getattr(tree_batch, "edge_visit_weights", None)
            if edge_visit_weights is None:
                raise ValueError("bucket_by='visit_weight' requires tree_batch.edge_visit_weights to be set.")
            child_size_bin = size_bin(edge_visit_weights.to(device), self.subtree_size_log_max)
        else:
            edge_child = tree_batch.edge_child.to(device)
            parent_index_all = tree_batch.parent_index.to(device)
            subtree_sizes = compute_subtree_sizes(parent_index_all, depth_all)
            child_size_bin = size_bin(subtree_sizes[edge_child], self.subtree_size_log_max)
        parent_depth_bin = depth_bin(depth_all[edge_parent], self.max_depth_bin)
        num_size_bins = self.subtree_size_log_max + 1
        num_depth_bins = self.max_depth_bin + 1
        flat_bin = child_size_bin * num_depth_bins + parent_depth_bin

        flat_sum = torch.zeros(num_size_bins * num_depth_bins, dtype=torch.float64, device=device)
        flat_count = torch.zeros(num_size_bins * num_depth_bins, dtype=torch.long, device=device)
        flat_sum.scatter_add_(0, flat_bin, per_edge_kl.to(torch.float64))
        flat_count.scatter_add_(0, flat_bin, torch.ones_like(flat_bin, dtype=torch.long))
        self.sum_kl += flat_sum.view(num_size_bins, num_depth_bins)
        self.count += flat_count.view(num_size_bins, num_depth_bins)
        self.overall_sum_kl += per_edge_kl.to(torch.float64).sum()
        self.overall_count += int(per_edge_kl.shape[0])

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable per-bin summary plus marginals + overall stats."""
        sum_kl_cpu = self.sum_kl.detach().cpu()
        count_cpu = self.count.detach().cpu()
        mean_kl = torch.where(
            count_cpu > 0,
            sum_kl_cpu / count_cpu.clamp_min(1).to(torch.float64),
            torch.zeros_like(sum_kl_cpu),
        )
        # Marginals: collapse one axis, weight by edge counts.
        def _axis_mean(axis: int) -> List[float]:
            sums = sum_kl_cpu.sum(dim=axis)
            counts = count_cpu.sum(dim=axis).clamp_min(1).to(torch.float64)
            return (sums / counts).tolist()

        overall_mean = float((self.overall_sum_kl / max(self.overall_count, 1)).item()) if self.overall_count > 0 else 0.0
        return {
            "overall_mean_kl": overall_mean,
            "overall_edge_count": int(self.overall_count),
            "max_depth_bin": int(self.max_depth_bin),
            "subtree_size_log_max": int(self.subtree_size_log_max),
            "mean_kl_grid": mean_kl.tolist(),
            "edge_count_grid": count_cpu.tolist(),
            "marginal_mean_kl_by_depth": _axis_mean(axis=0),
            "marginal_mean_kl_by_size_bin": _axis_mean(axis=1),
            "marginal_edge_count_by_depth": count_cpu.sum(dim=0).tolist(),
            "marginal_edge_count_by_size_bin": count_cpu.sum(dim=1).tolist(),
            "size_bin_labels": size_bin_labels(self.subtree_size_log_max),
            "depth_bin_labels": depth_bin_labels(self.max_depth_bin),
        }


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
        """For the k-steps-ahead ("packhistory_GNNpretrain") objective, ``model``
        is an unmodified ``ChildWdlModel`` applied to a T_n snapshot instead of a
        complete tree -- no new head needed, since ``ChildWdlHead``'s
        ``concat(parent_states, slot_states)`` carries each child's disambiguating
        signal on a partial tree the same way it does on a complete one. Loss,
        weighting branch, and batch loop are unchanged for this objective too.
        """
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
        bucket_state: Optional[BucketedKLState] = None,
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
                # Target entropy is a constant of the data; tracking it
                # gives the loss_gap, which is the part of the loss we can
                # actually drive down.
                target_log_probs = edge_targets.clamp_min(1e-12).log()
                per_edge_target_entropy = -(edge_targets * target_log_probs).sum(dim=-1)

                edge_visit_weights = getattr(tree_batch, "edge_visit_weights", None)
                if self.config.loss_weight_by_subtree_size:
                    # Weight each edge by its child's subtree size: the encoder's
                    # job is to summarize subtrees into the parent state, and
                    # this weighting makes "amount of information summarized"
                    # the unit of equal contribution to the loss rather than
                    # "edges predicted." Both the cross-entropy and the
                    # target-entropy reference use the same weights so
                    # ``loss_gap`` remains a clean weighted KL.
                    depth_all = tree_batch.depth.to(self.model.encoder.device)
                    parent_index_all = tree_batch.parent_index.to(self.model.encoder.device)
                    edge_child = tree_batch.edge_child.to(self.model.encoder.device)
                    subtree_sizes = compute_subtree_sizes(parent_index_all, depth_all)
                    edge_weights = subtree_sizes[edge_child].to(per_edge_loss.dtype)
                    weight_sum = edge_weights.sum()
                    total_loss = (edge_weights * per_edge_loss).sum() / weight_sum
                    target_entropy = (edge_weights * per_edge_target_entropy).sum() / weight_sum
                elif edge_visit_weights is not None:
                    # k-steps-ahead ("packhistory_GNNpretrain") packed batches:
                    # weight by Delta-visits instead of subtree size. Delta-visits=0
                    # edges are still present in the batch (never filtered out
                    # upstream) but contribute exactly zero to both sums below —
                    # a weight, not a filter, per history.md.
                    edge_weights = edge_visit_weights.to(per_edge_loss.dtype).to(self.model.encoder.device)
                    if self.config.visit_weight_transform == "log1p":
                        edge_weights = torch.log1p(edge_weights)
                    weight_sum = edge_weights.sum().clamp_min(1e-12)
                    total_loss = (edge_weights * per_edge_loss).sum() / weight_sum
                    target_entropy = (edge_weights * per_edge_target_entropy).sum() / weight_sum
                else:
                    total_loss = per_edge_loss.mean()
                    target_entropy = per_edge_target_entropy.mean()
                loss_gap = total_loss - target_entropy

                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                # Bucketed-KL accumulation runs inside the no-grad guard but
                # only when an accumulator is supplied (i.e. on validation
                # when logging is enabled). Per-edge KL is loss − target
                # entropy, which the loop has already computed; we just need
                # to detach and scatter into bins.
                if bucket_state is not None:
                    per_edge_kl = (per_edge_loss - per_edge_target_entropy).detach()
                    bucket_state.update(tree_batch, per_edge_kl)

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
        bucket_state: Optional[BucketedKLState] = None,
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
                bucket_state=bucket_state,
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
        bucket_state: Optional[BucketedKLState] = None,
    ) -> ChildWdlMetrics:
        """Run one validation pass over ``self.validation_examples`` (no grad, no shuffle).

        If ``bucket_state`` is supplied, per-edge KL is also accumulated
        into that grid as a side effect; callers query ``bucket_state.summary()``
        after the call to read the per-bin time-series row for the epoch.
        """
        return self._run_epoch(
            self.validation_examples,
            epoch_index,
            training=False,
            batch_progress_callback=batch_progress_callback,
            bucket_state=bucket_state,
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
        progress_callback: Optional[Callable[..., None]] = None,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
        start_epoch: int = 1,
        resume_path: Optional[str] = None,
        bucketed_kl_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
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
        log_buckets = bool(self.config.log_bucketed_kl)
        for epoch_index in range(start_epoch, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch_index, batch_progress_callback=batch_progress_callback)
            # Allocate a fresh accumulator per validation epoch so the per-epoch
            # summary reflects the latest weights, not a running average across epochs.
            bucket_state = (
                BucketedKLState.empty(
                    max_depth_bin=self.config.bucket_max_depth_bin,
                    subtree_size_log_max=self.config.bucket_subtree_size_log_max,
                    device=self.model.encoder.device,
                    bucket_by=self.config.bucket_by,
                )
                if log_buckets
                else None
            )
            validation_metrics = self.validate(
                epoch_index,
                batch_progress_callback=batch_progress_callback,
                bucket_state=bucket_state,
            )
            history.append({"train": train_metrics, "validation": validation_metrics})
            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
                self.best_decoder_state = copy.deepcopy(self.model.child_wdl_head.state_dict())
            if progress_callback is not None:
                progress_callback(epoch_index, train_metrics, validation_metrics)
            if bucket_state is not None and bucketed_kl_callback is not None:
                bucketed_kl_callback(epoch_index, bucket_state.summary())
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
