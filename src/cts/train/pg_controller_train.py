"""Optimizes regret directly: treats the controller as a stochastic stop policy
(continues at step t with p_t = sigmoid(A_t)) and marginalizes the stop step in
closed form over the known full halt-reward trace, so there's no REINFORCE
sampling / rollout variance:

    P(stop@s)   = (prod_{u<s} p_u) * (1 - p_s)      for s < T-1
    P(stop@T-1) = prod_{u<T-1} p_u                  (forced stop at budget end)
    E[return]   = sum_s P(stop@s) * g(s)
    loss        = oracle_value - E[return]   ==   E[regret]

Checkpoints are selected on the hard greedy val regret, not this soft loss.
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
from cts.train.controller_train import _load_materialized_cache_unchecked


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


def _use_steps_not_budget(chunks, d_embed):
    """Scrap the budget: overwrite the ``T_t`` (remaining-budget) column of each per-episode feature
    chunk with the within-episode STEP index, so the controller reads ``[z_t, steps_taken]`` instead of
    ``[z_t, T_t]``. ``controller_inputs`` stays ``['z_t','T_t']`` — the ``T_t`` slot now carries steps.

    Rationale: the per-episode budget entangles position-in-time with the episode's total budget, so we
    drop it and feed the raw expansion count. Matches ``analysis.evaluate`` (steps-in / budget-out). The
    budget idea may return later; this is the single point that would revert.
    """
    for ch in chunks:
        ch[:, d_embed + 1] = torch.arange(ch.shape[0], device=ch.device, dtype=ch.dtype)
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
                            lengths: torch.Tensor, oracle_values: torch.Tensor,
                            temperature: float = 1.0) -> torch.Tensor:
    """Mean E[regret] over a right-padded batch (closed form, fully vectorized).

    adv/g/mask: [B,Tmax]; lengths:[B]; oracle_values:[B]. Equivalent to averaging the
    per-episode :func:`_expected_regret` (verified to 1e-7), but as batched tensor ops.
    ``temperature`` softens the stop policy: continue prob = sigmoid(A_t / tau); tau > 1
    keeps the sigmoid off its saturated tails so the gradient survives (see config)."""
    if temperature != 1.0:
        adv = adv / temperature
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


def _pg_train_epoch(advantage_fn, feats, g_pad, mask, lengths, oracle_values, optimizer,
                    *, perm, episode_batch, temperature, max_grad_norm, trainable_params):
    """One PG (exact expected-return) epoch over episode-batched features; returns mean train loss.

    ``advantage_fn(cat_features) -> flat advantages`` abstracts the model — a MetaController's
    ``predict_from_features`` for the deployed controller (:func:`main`), or a bare readout head for
    the ``analysis.evaluate`` baselines (:func:`fit_readout_pg`) — so both share ONE training loop.
    """
    epoch_loss, seen = 0.0, 0
    for i in range(0, len(feats), episode_batch):
        idxs = perm[i:i + episode_batch]
        optimizer.zero_grad()
        cat = torch.cat([feats[j] for j in idxs.tolist()], dim=0)       # one forward for the minibatch
        adv_pad = _scatter_to_padded(advantage_fn(cat).reshape(-1), lengths[idxs])
        t = adv_pad.shape[1]
        loss = expected_regret_batched(adv_pad, g_pad[idxs, :t], mask[idxs, :t], lengths[idxs],
                                       oracle_values[idxs], temperature=temperature)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        epoch_loss += float(loss) * idxs.numel()
        seen += idxs.numel()
    return epoch_loss / max(seen, 1)


def _greedy_regret_from_curves(head, ev_feats, ev_curves) -> float:
    """Mean greedy-stop regret of ``head`` on a held-out set of (already-normalized) per-episode
    feature tensors + matching return curves — the same stop rule (``stop_step_from_advantages``)
    and regret definition (``oracle_value - g(stop_step)``) used everywhere else in this
    investigation (e.g. ``analysis.evaluate._regret_at``), just recomputed once per epoch here
    rather than once at the end of training."""
    from cts.models.readout import stop_step_from_advantages
    regrets = []
    for feats, curve in zip(ev_feats, ev_curves):
        adv = head(feats).reshape(-1)
        s = min(stop_step_from_advantages(adv), len(curve) - 1)
        regrets.append(float(curve.max()) - float(curve[s]))
    return sum(regrets) / len(regrets) if regrets else float("nan")


def fit_readout_pg(head, ep_feats, ep_curves, *, epochs, lr, seed=0, episode_batch=512,
                   weight_decay=0.0, temperature=1.0, temperature_final=1.0, max_grad_norm=1.0,
                   ev_feats=None, ev_curves=None, out_path=None):
    """Train a STANDALONE readout head by exact expected-return over per-episode RETURN CURVES.

    The ``analysis.evaluate`` path — no encoder / cache / checkpointing. ``ep_feats`` is a list of
    per-episode ``[T_i, F]`` feature tensors; ``ep_curves`` the matching per-episode return curves
    (``g(s)=curve[s]``, ``oracle_value=max(curve)``). Shares the exact training loop
    (:func:`_pg_train_epoch`) the deployed controller uses, so the assessment fits and the deployed
    controller optimize the identical objective. Returns the trained ``head``.

    If ``ev_feats``/``ev_curves`` (held-out, already-normalized features + matching curves) are also
    given, tracks per-epoch greedy validation regret (:func:`_greedy_regret_from_curves`) alongside the
    train loss. If ``out_path`` is further given, the resulting history is saved via
    :func:`_save_training_curves` (CSV + PNG) — same convention as the deployed controller's
    checkpoint-adjacent curve (:func:`main`), just for these standalone assessment-time fits.
    """
    torch.manual_seed(seed)
    g_list = [torch.tensor(c, dtype=torch.float32) for c in ep_curves]
    ov_list = [float(c.max()) for c in ep_curves]
    g_pad, mask, lengths, ov = _pad_scalars(g_list, ov_list, torch.device("cpu"))
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    params = list(head.parameters())
    track_val = ev_feats is not None and ev_curves is not None
    history = []
    for epoch in range(1, epochs + 1):
        tau = (temperature_final if epochs <= 1 else
               temperature + (temperature_final - temperature) * (epoch - 1) / (epochs - 1))
        head.train()
        mean_train_loss = _pg_train_epoch(
            head, ep_feats, g_pad, mask, lengths, ov, opt, perm=torch.randperm(len(ep_feats)),
            episode_batch=episode_batch, temperature=tau, max_grad_norm=max_grad_norm,
            trainable_params=params)
        if track_val:
            head.eval()
            with torch.no_grad():
                val_regret = _greedy_regret_from_curves(head, ev_feats, ev_curves)
            history.append({"epoch": epoch, "tau": tau, "train_E_regret": mean_train_loss,
                            "val_greedy_regret": val_regret})
    if out_path is not None and history:
        _save_training_curves(history, Path(out_path))
    return head


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

    # Budget scrapped: the T_t column is overwritten with steps-taken, so the controller reads
    # [z_t, steps] (see _use_steps_not_budget). controller_inputs stays ['z_t','T_t'].
    feats = _use_steps_not_budget(
        _episode_feature_chunks(train_cache, train_meta, train_steps, device), config.d_embed)  # per-episode [T,F]
    g_list, ov_list = _episode_returns(train_meta, oracle_config, device)
    g_pad, mask, lengths, ov = _pad_scalars(g_list, ov_list, device)
    # Cache val features ONCE (episode-concatenated) so greedy eval re-forwards them each
    # epoch without re-reading shards from disk (the old per-epoch reload was a big cost).
    val_feats = torch.cat(_use_steps_not_budget(
        _episode_feature_chunks(val_cache, val_meta, val_steps, device), config.d_embed), dim=0)

    n = len(feats)
    batch = max(1, int(config.pg_episode_batch))
    best_regret = float("inf")
    out_path = Path(config.output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []  # per-epoch (tau, train E[regret], val greedy regret, val stop acc, expansions)

    def _tau(ep: int) -> float:
        """Linear anneal stop_temperature -> stop_temperature_final over the run."""
        if config.epochs <= 1:
            return float(config.stop_temperature_final)
        frac = (ep - 1) / (config.epochs - 1)
        return float(config.stop_temperature + (config.stop_temperature_final - config.stop_temperature) * frac)

    for epoch in range(1, config.epochs + 1):
        tau = _tau(epoch)
        model.train()
        # SAME training loop as the analysis.evaluate baselines (fit_readout_pg); here the
        # advantage_fn is the encoder-backed MetaController rather than a bare head.
        mean_train_loss = _pg_train_epoch(
            lambda cat: model.predict_from_features(cat)[0], feats, g_pad, mask, lengths, ov, optimizer,
            perm=torch.randperm(n), episode_batch=batch, temperature=tau,
            max_grad_norm=config.max_grad_norm,
            trainable_params=[p for p in model.parameters() if p.requires_grad])
        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.inference_mode():
            adv = model.predict_from_features(val_feats)[0].reshape(-1).cpu()
        m = _aggregate_greedy_rollout_metrics(
            val_meta, adv, oracle_config, log_interval=0, started=0.0, diagnostics_out=[])
        print(f"[pg] epoch={epoch}/{config.epochs} tau={tau:.2f} train_E[regret]={mean_train_loss:.4f} "
              f"val_greedy_regret={m.average_regret:.4f} val_stop_acc={m.exact_stop_step_accuracy:.3f} "
              f"val_expansions={m.average_expansions:.2f}", flush=True)
        history.append({"epoch": epoch, "tau": tau, "train_E_regret": mean_train_loss,
                        "val_greedy_regret": m.average_regret,
                        "val_stop_acc": m.exact_stop_step_accuracy,
                        "val_expansions": m.average_expansions})
        def _payload(regret: float) -> dict:
            return {"model_state_dict": model.state_dict(), "metadata": {
                "controller_inputs": list(config.controller_inputs),
                "separate_sign_head": bool(config.separate_sign_head),
                "encoder_checkpoint": str(config.encoder_checkpoint),
                "objective": "pg_expected_return",
                "epoch": int(epoch),
                "val_greedy_regret": float(m.average_regret),
                "best_val_greedy_regret": float(regret),
            }}

        if m.average_regret < best_regret:
            best_regret = m.average_regret
            torch.save(_payload(best_regret), out_path)
            print(f"[pg] saved best val_greedy_regret={best_regret:.4f} -> {out_path}", flush=True)
        if config.save_every_epoch:
            # plan.md Agent 2 -- per-epoch snapshot (not just on best-regret improvement) so a
            # continued run can be paired-significance-evaluated at several points along its
            # curve, mirroring e2e_controller_train.py's existing always-save-every-epoch policy.
            epoch_path = out_path.with_name(f"{out_path.stem}_epoch{epoch:03d}{out_path.suffix}")
            torch.save(_payload(best_regret), epoch_path)

    _save_training_curves(history, out_path)
    print(f"[pg] DONE best_val_greedy_regret={best_regret:.4f}", flush=True)


def _save_training_curves(history: list[dict], out_path: Path) -> None:
    """Write the per-epoch regret/loss trajectory (CSV) and a curve figure next to
    the checkpoint — train E[regret] and hard val greedy regret over epochs, plus
    the temperature schedule."""
    if not history:
        return
    import csv
    base = out_path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{base}_training_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"[pg] training curve -> {csv_path}", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Match the repo's board-plot house style (analysis.utils.helpers.apply_poster_style):
        # sans-serif font, no top/right spines, light grid. The poster style's own font-size
        # constants (FONT_SIZE_LABEL=52 etc.) are calibrated for 40+-inch board dashboards, so
        # they're not reused verbatim here — this is a small (9,6) figure, so font sizes below
        # are picked to be legible on that scale instead of overflowing it.
        from analysis.utils.helpers import apply_poster_style, PHASE_COLORS
        from analysis.utils.plots import save_pdf_png
        apply_poster_style()
        # Board-plot sizing convention: a SMALLER figure at these font sizes reads as bigger,
        # more legible text (same reasoning as the board dashboards) -- no title (board plots
        # don't carry one either; the filename/caption is the title), fewer ticks.
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['axes.labelsize'] = 13
        plt.rcParams['legend.fontsize'] = 11
        train_color = PHASE_COLORS[1]  # light indigo
        val_color = PHASE_COLORS[3]    # dark indigo
        ep = [h["epoch"] for h in history]
        train_vals = [h["train_E_regret"] for h in history]
        val_vals = [h["val_greedy_regret"] for h in history]
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        ax.plot(ep, train_vals, "-", color=train_color, label="Train")
        ax.plot(ep, val_vals, "-", color=val_color, label="Val")
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax.set_ylabel("Regret")
        ax.set_xlabel("Epoch"); ax.legend(loc="upper right")
        fig.tight_layout()
        curve_path = save_pdf_png(fig, str(base.parent), f"{base.name}_training_curve", dpi=150)
        print(f"[pg] training curve figure -> {curve_path}", flush=True)
    except Exception as e:  # plotting is best-effort; the CSV is the source of truth
        print(f"[pg] curve plot skipped: {e}", flush=True)


if __name__ == "__main__":
    run_with_config_cli(ControllerTrainConfig, main)
