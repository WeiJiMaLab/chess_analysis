"""Policy-gradient (exact expected-return) trainer for the MCHalt controller.

The standard ``controller_train`` fits the per-step advantage with MSE + a sign BCE — a
DIFFERENTIABLE SURROGATE for regret, because the deployed objective (regret of the greedy
"stop at first advantage<=0" rule) is a threshold-crossing of the advantage trace and has
no usable gradient w.r.t. the head weights.

This trainer optimizes the regret objective DIRECTLY. Treat the controller as a stochastic
stop policy: at step t it continues with probability ``p_t = sigmoid(A_t)``. Because every
episode's FULL halt-reward trace is known offline, the stop-step distribution and the
expected return are written in CLOSED FORM — the policy gradient with the stop step
marginalized analytically, so there is no REINFORCE sampling / rollout variance:

    P(stop@s)   = (prod_{u<s} p_u) * (1 - p_s)      for s < T-1     (continue then stop)
    P(stop@T-1) = prod_{u<T-1} p_u                  (forced stop at the budget end)
    E[return]   = sum_s P(stop@s) * g(s)            (g(s) = return_for_stop_step(s))
    loss        = oracle_value - E[return]   ==   E[regret]

As training sharpens p_u toward 0/1 the soft policy converges to the deployed hard greedy
rule. Checkpoints are still SELECTED on the HARD greedy val regret, so the result is
directly comparable to the MSE-trained controller and the baseline tiers. Only the MLP head
trains (encoder frozen; z_t read from the materialized cache), so it is CPU-cheap.

    python -m cts.train.pg_controller_train --config config.yaml --stage train \
        --set globals.sf_elo=2000 --set device=cpu \
        --set output_checkpoint=/scratch/.../sf_mchalt_pg.pt --set epochs=20
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from cts._config import run_with_config_cli
from cts.data.preprocess_mc.oracle import return_for_stop_step
from cts.train.controller_train import (
    ControllerTrainConfig,
    ControllerEpisodeDataset,
    _aggregate_greedy_rollout_metrics,
    _build_model_and_optimizer,
    _collect_episode_metadata_and_step_count,
    _seed_and_resolve_paths,
)
from analysis.mchalt_scorer import _load_materialized_cache_unchecked


def _episode_feature_chunks(cache, episode_meta, total_steps, device):
    """Load the first ``total_steps`` cache rows (cache order == dataset episode order)
    and split them per episode into a list of [num_steps_i, feat_dim] tensors."""
    parts = []
    produced = 0
    for shard_path in cache.shard_paths:
        if produced >= total_steps:
            break
        feats = torch.load(shard_path, weights_only=False)["features"]
        take = min(feats.shape[0], total_steps - produced)
        parts.append(feats[:take])
        produced += take
    allf = torch.cat(parts, dim=0).to(device)
    chunks, off = [], 0
    for em in episode_meta:
        chunks.append(allf[off:off + em.num_steps])
        off += em.num_steps
    return chunks


def _episode_returns(episode_meta, oracle_config, device):
    """Per-episode g(s)=return_for_stop_step for every s (the reward of each possible stop)."""
    g_list, ov = [], []
    for em in episode_meta:
        g = [return_for_stop_step(em.halt_rewards, em.tree_sizes, em.time_budgets, s, oracle_config)
             for s in range(em.num_steps)]
        g_list.append(torch.tensor(g, dtype=torch.float32, device=device))
        ov.append(float(em.oracle_value))
    return g_list, ov


def _expected_regret(advantages: torch.Tensor, g: torch.Tensor, oracle_value: float) -> torch.Tensor:
    """E[regret] of the soft stop policy for one episode — closed form over stop steps."""
    A = advantages.reshape(-1)
    logp = F.logsigmoid(A)        # log P(continue) = log sigmoid(A_t)
    log1m = F.logsigmoid(-A)      # log P(stop)     = log (1 - sigmoid(A_t))
    prefix = torch.cat([A.new_zeros(1), torch.cumsum(logp, 0)[:-1]])  # prefix[s] = sum_{u<s} log p_u
    pstop = torch.exp(prefix + log1m)
    pstop = torch.cat([pstop[:-1], torch.exp(prefix[-1:])])  # last step = forced stop (continued all prior)
    e_return = (pstop * g).sum()
    return oracle_value - e_return


def expected_regret_batched(adv: torch.Tensor, g: torch.Tensor, mask: torch.Tensor,
                            lengths: torch.Tensor, oracle_values: torch.Tensor) -> torch.Tensor:
    """Mean E[regret] over a right-padded batch (closed form, fully vectorized).

    adv/g/mask: [B,Tmax]; lengths:[B]; oracle_values:[B]. Equivalent to averaging the
    per-episode :func:`_expected_regret` (verified to 1e-7), but as batched tensor ops.
    """
    logp = F.logsigmoid(adv) * mask                  # zero padding so cumsum ignores it
    log1m = F.logsigmoid(-adv)
    prefix = torch.cat([logp.new_zeros(adv.shape[0], 1), torch.cumsum(logp, 1)[:, :-1]], dim=1)
    pstop = torch.exp(prefix + log1m)
    last = (lengths - 1).clamp_min(0).unsqueeze(1)
    pstop = pstop.scatter(1, last, torch.exp(prefix.gather(1, last)))  # forced stop at each episode's last step
    pstop = pstop * mask
    e_return = (pstop * g).sum(dim=1)
    return (oracle_values - e_return).mean()


def _scatter_to_padded(flat: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reshape an episode-concatenated flat [sum(lengths)] vector into [B, max(lengths)]."""
    B, Tmax, total = lengths.numel(), int(lengths.max()), int(lengths.sum())
    rows = torch.repeat_interleave(torch.arange(B, device=flat.device), lengths)
    starts = torch.cumsum(lengths, 0) - lengths
    cols = torch.arange(total, device=flat.device) - torch.repeat_interleave(starts, lengths)
    out = flat.new_zeros(B, Tmax)
    out[rows, cols] = flat
    return out


def _pad_scalars(g_list: list[torch.Tensor], ov_list: list[float], device):
    """One-time: pad per-episode g(s) into [N,Tmax] + mask, with lengths and oracle values."""
    lengths = torch.tensor([g.numel() for g in g_list], device=device)
    n, tmax = len(g_list), int(lengths.max())
    g_pad = torch.zeros(n, tmax, device=device)
    mask = torch.zeros(n, tmax, device=device)
    for i, g in enumerate(g_list):
        t = g.numel()
        g_pad[i, :t] = g
        mask[i, :t] = 1.0
    return g_pad, mask, lengths, torch.tensor(ov_list, device=device)


def main(config: ControllerTrainConfig) -> None:
    device, train_cache_path, validation_cache_path, oracle_config, schema = _seed_and_resolve_paths(config)
    model, optimizer, scheduler = _build_model_and_optimizer(config, schema)

    cap = config.pg_max_episodes
    train_meta, train_steps = _collect_episode_metadata_and_step_count(
        ControllerEpisodeDataset(str(config.packed_train_data)), max_episodes=cap)
    val_meta, val_steps = _collect_episode_metadata_and_step_count(
        ControllerEpisodeDataset(str(config.packed_validation_data)), max_episodes=cap)
    train_cache = _load_materialized_cache_unchecked(Path(train_cache_path))
    val_cache = _load_materialized_cache_unchecked(Path(validation_cache_path))
    print(f"[pg] train_episodes={len(train_meta)} val_episodes={len(val_meta)} "
          f"episode_batch={config.pg_episode_batch} epochs={config.epochs} device={device}", flush=True)

    feats = _episode_feature_chunks(train_cache, train_meta, train_steps, device)  # per-episode [T,F]
    g_list, ov_list = _episode_returns(train_meta, oracle_config, device)
    g_pad, mask, lengths, ov = _pad_scalars(g_list, ov_list, device)
    # Cache val features ONCE (episode-concatenated) so greedy eval re-forwards them each
    # epoch without re-reading shards from disk (the old per-epoch reload was a big cost).
    val_feats = torch.cat(_episode_feature_chunks(val_cache, val_meta, val_steps, device), dim=0)

    n = len(feats)
    batch = max(1, int(config.pg_episode_batch))
    best_regret = float("inf")
    out_path = Path(config.output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, seen = 0.0, 0
        for i in range(0, n, batch):
            idxs = perm[i:i + batch]
            optimizer.zero_grad()
            cat = torch.cat([feats[j] for j in idxs.tolist()], dim=0)   # one forward for the minibatch
            A_flat = model.predict_from_features(cat)[0].reshape(-1)
            L = lengths[idxs]
            A_pad = _scatter_to_padded(A_flat, L)                       # [B, Tmax_mb], vectorized
            t = A_pad.shape[1]
            loss = expected_regret_batched(A_pad, g_pad[idxs, :t], mask[idxs, :t], L, ov[idxs])
            loss.backward()
            if config.max_grad_norm and config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), config.max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss) * idxs.numel()
            seen += idxs.numel()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.inference_mode():
            adv = model.predict_from_features(val_feats)[0].reshape(-1).cpu()
        m = _aggregate_greedy_rollout_metrics(
            val_meta, adv, oracle_config, log_interval=0, started=0.0, diagnostics_out=[])
        print(f"[pg] epoch={epoch}/{config.epochs} train_E[regret]={epoch_loss / seen:.4f} "
              f"val_greedy_regret={m.average_regret:.4f} val_stop_acc={m.exact_stop_step_accuracy:.3f} "
              f"val_expansions={m.average_expansions:.2f}", flush=True)
        if m.average_regret < best_regret:
            best_regret = m.average_regret
            torch.save({"model_state_dict": model.state_dict(), "metadata": {
                "controller_inputs": list(config.controller_inputs),
                "separate_sign_head": bool(config.separate_sign_head),
                "encoder_checkpoint": str(config.encoder_checkpoint),
                "objective": "pg_expected_return",
                "best_val_greedy_regret": best_regret,
            }}, out_path)
            print(f"[pg] saved best val_greedy_regret={best_regret:.4f} -> {out_path}", flush=True)

    print(f"[pg] DONE best_val_greedy_regret={best_regret:.4f}", flush=True)


if __name__ == "__main__":
    run_with_config_cli(ControllerTrainConfig, main)
