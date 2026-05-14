"""``FramewiseRolloutHook`` — full-rollout convergence + GIF diagnostics
for the #016 framewise diffusion stack.

Runs every Nth eval. Per eval, the hook:

1. Samples ``eval_n_charts * eval_n_windows_per_chart`` (sample_id,
   window) draws from the val sampler, plus the same on a deterministic
   train-no-aug subset.
2. For each draw, runs ``DDIMSampler.sample_with_intermediates`` and
   collects ``M_k`` for every k. Computes per-step convergence metrics
   (F1 at canonical op, MSE, mass_at_target / off_target).
3. Writes ``rollout_maps.npz`` + ``noaug_rollout_maps.npz`` with
   per-sample per-step ``M_k`` tensors + metadata.
4. Renders ``convergence_curves.png`` (3 panels: F1 / MSE / mass) with
   mean + p10/p25/p75/p90 bands over k.
5. Renders ``convergence_by_density.png`` / ``convergence_by_star.png``
   / ``convergence_by_kind.png`` 4-6 subplots each.
6. Renders 5 representative GIFs to ``rollout_gifs/`` (best / p75 / p50
   / p25 / worst by final F1).
7. Renders ``summary_histogram.gif`` — histogram of mean(M_k) at each k.

GIFs use PIL (no imageio dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..domain.training import RunSpec, TrainerHook, TrainingState


@dataclass(frozen=True, slots=True)
class FramewiseRolloutHookConfig:
    eval_n_charts: int = 32
    eval_n_windows_per_chart: int = 5
    noaug_n_charts: int = 32
    noaug_n_windows_per_chart: int = 5
    t_inf_steps: int = 16
    n_gif_samples: int = 5
    decode_threshold: float = 0.5
    canonical_tolerance_frames: int = 2
    seed: int = 42
    every_n_evals: int = 1

    def __post_init__(self) -> None:
        if self.eval_n_charts < 1:
            raise ValueError(f"eval_n_charts must be >= 1 (got {self.eval_n_charts})")
        if self.eval_n_windows_per_chart < 1:
            raise ValueError(
                f"eval_n_windows_per_chart must be >= 1 "
                f"(got {self.eval_n_windows_per_chart})"
            )
        if self.noaug_n_charts < 0:
            raise ValueError(
                f"noaug_n_charts must be >= 0 (got {self.noaug_n_charts})"
            )
        if self.t_inf_steps < 1:
            raise ValueError(f"t_inf_steps must be >= 1 (got {self.t_inf_steps})")
        if self.n_gif_samples < 0:
            raise ValueError(
                f"n_gif_samples must be >= 0 (got {self.n_gif_samples})"
            )
        if not 0.0 <= self.decode_threshold <= 1.0:
            raise ValueError(
                f"decode_threshold must be in [0, 1] "
                f"(got {self.decode_threshold})"
            )
        if self.every_n_evals < 1:
            raise ValueError(
                f"every_n_evals must be >= 1 (got {self.every_n_evals})"
            )


# ─────────────────────────── helpers ─────────────────────────────────


def _dilate(target_binary: torch.Tensor, k: int) -> torch.Tensor:
    """Same 1-D max-pool dilation used by framewise_curve_metrics."""
    if k <= 0:
        return target_binary.float()
    import torch.nn.functional as F
    return F.max_pool1d(
        target_binary.float().unsqueeze(1),
        kernel_size=2 * k + 1, stride=1, padding=k,
    ).squeeze(1)


def _f1_at(
    m_hat: torch.Tensor, tgt_binary: torch.Tensor,
    threshold: float, tol_frames: int,
) -> float:
    """Scalar F1 — same definition the loss uses, but no python loop."""
    pred_pos = (m_hat > threshold).float()
    gt = tgt_binary.float()
    dil_gt = _dilate(tgt_binary, tol_frames)
    dil_pr = _dilate(pred_pos, tol_frames)
    n_pred = float(pred_pos.sum().item())
    n_pos = float(gt.sum().item())
    tp_p = float((pred_pos * dil_gt).sum().item())
    tp_r = float((gt * dil_pr).sum().item())
    p = tp_p / n_pred if n_pred > 0 else 0.0
    r = tp_r / n_pos if n_pos > 0 else 0.0
    return (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0


def _per_step_metrics(
    per_step: np.ndarray,           # (T_inf, B, n_bins)
    target_binary: np.ndarray,       # (B, n_bins)
    target_smoothed: np.ndarray,     # (B, n_bins)
    threshold: float,
    tol_frames: int,
) -> dict[str, np.ndarray]:
    T_inf, B, N = per_step.shape
    f1 = np.zeros((B, T_inf), dtype=np.float32)
    mse = np.zeros((B, T_inf), dtype=np.float32)
    mass_at = np.zeros((B, T_inf), dtype=np.float32)
    mass_off = np.zeros((B, T_inf), dtype=np.float32)
    total_mass = np.zeros((B, T_inf), dtype=np.float32)

    tgt_bin_t = torch.from_numpy(target_binary)
    for k in range(T_inf):
        Mk = torch.from_numpy(per_step[k]).clamp(0.0, 1.0)
        for b in range(B):
            f1[b, k] = _f1_at(
                Mk[b:b + 1], tgt_bin_t[b:b + 1],
                threshold, tol_frames,
            )
        diff = per_step[k] - target_smoothed
        mse[:, k] = (diff ** 2).mean(axis=-1)
        # Mass partitioning.
        pos = target_binary > 0.5
        for b in range(B):
            row = per_step[k, b]
            tm = float(row.sum())
            ma = float(row[pos[b]].sum()) if pos[b].any() else 0.0
            total_mass[b, k] = tm
            mass_at[b, k] = ma
            mass_off[b, k] = tm - ma
    return {
        "f1": f1,
        "mse": mse,
        "mass_at_target": mass_at,
        "mass_off_target": mass_off,
        "total_mass": total_mass,
    }


def _summary_per_sample(
    metrics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute per-sample summary scalars: best_k_step,
    final_vs_best_delta, convergence_step_90, monotone_fraction."""
    f1 = metrics["f1"]                          # (B, T_inf)
    B, T = f1.shape
    best_k = np.argmax(f1, axis=1)
    best_val = f1[np.arange(B), best_k]
    final_val = f1[:, -1]
    final_vs_best = final_val - best_val
    # Step at which F1 reaches 90 % of final value, walking forward.
    target = 0.9 * np.maximum(final_val, 1e-6)
    conv_step = np.zeros(B, dtype=np.float32)
    for b in range(B):
        idx = np.where(f1[b] >= target[b])[0]
        conv_step[b] = float(idx[0]) if idx.size > 0 else float(T - 1)
    # Monotone fraction: % of k transitions that don't strictly decrease.
    diffs = f1[:, 1:] - f1[:, :-1]
    mono = (diffs >= 0).astype(np.float32).mean(axis=1) if T > 1 else np.zeros(B)
    return {
        "best_k_step": best_k.astype(np.int64),
        "best_f1": best_val.astype(np.float32),
        "final_f1": final_val.astype(np.float32),
        "final_vs_best_delta": final_vs_best.astype(np.float32),
        "convergence_step_90": conv_step.astype(np.float32),
        "monotone_fraction": mono.astype(np.float32),
    }


# ─────────────────────────── plots ───────────────────────────────────


def _render_convergence_curves(
    metrics: dict[str, np.ndarray], out_path: Path, step: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = (
        ("f1", "F1 at op-point", axes[0]),
        ("mse", "MSE", axes[1]),
        ("mass_at_target", "Mass at target", axes[2]),
    )
    for key, title, ax in panels:
        arr = metrics[key]                # (B, T)
        T = arr.shape[1]
        xs = np.arange(T)
        p10 = np.percentile(arr, 10, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p50 = np.percentile(arr, 50, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        p90 = np.percentile(arr, 90, axis=0)
        mean = arr.mean(axis=0)
        ax.fill_between(xs, p10, p90, color="#4a90d9", alpha=0.12, label="p10/p90")
        ax.fill_between(xs, p25, p75, color="#4a90d9", alpha=0.25, label="p25/p75")
        ax.plot(xs, p50, color="#4a90d9", linewidth=1.5, label="median")
        ax.plot(xs, mean, color="#e86850", linewidth=1.5, label="mean")
        ax.set_title(title)
        ax.set_xlabel("step k")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(f"Rollout convergence - step {step:,}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_bucketed(
    metrics: dict[str, np.ndarray],
    bucket_values: np.ndarray,
    bucket_edges: list[tuple[float, float, str]],
    out_path: Path, step: int, *, title: str,
) -> None:
    """Render F1-over-k curves stratified by a per-sample bucket value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(bucket_edges)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    f1 = metrics["f1"]
    T = f1.shape[1]
    xs = np.arange(T)
    for i, (lo, hi, label) in enumerate(bucket_edges):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        mask = (bucket_values >= lo) & (bucket_values < hi)
        n_in = int(mask.sum())
        if n_in > 0:
            arr = f1[mask]
            ax.fill_between(
                xs, np.percentile(arr, 25, axis=0),
                np.percentile(arr, 75, axis=0),
                color="#4a90d9", alpha=0.2, label="p25/p75",
            )
            ax.plot(xs, arr.mean(axis=0), color="#e86850",
                    linewidth=1.5, label="mean")
        ax.set_title(f"{label} (n={n_in})")
        ax.set_xlabel("step k")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    # Disable unused axes.
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"{title} - step {step:,}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_gif_for_sample(
    per_step: np.ndarray,      # (T_inf, n_bins)
    target_binary: np.ndarray, # (n_bins,)
    out_path: Path,
    *, title_prefix: str, threshold: float,
) -> None:
    """Render one GIF showing M_k over rollout steps for a single sample.

    Uses Pillow's append_images for portability (no imageio dependency).
    """
    from PIL import Image

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    T, N = per_step.shape
    frames: list[Image.Image] = []
    gt_idx = np.where(target_binary > 0.5)[0]
    for k in range(T):
        Mk = per_step[k]
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.imshow(
            Mk[None, :], aspect="auto", origin="lower",
            cmap="viridis", vmin=0.0, vmax=1.0,
            extent=(0, N, 0, 1),
        )
        # GT positions (red ticks above).
        ax.scatter(gt_idx + 0.5, np.full_like(gt_idx, 1.1, dtype=float),
                   marker="|", color="red", s=40)
        # Predicted positives (blue ticks below).
        pred_idx = np.where(Mk > threshold)[0]
        ax.scatter(pred_idx + 0.5, np.full_like(pred_idx, -0.1, dtype=float),
                   marker="|", color="blue", s=40)
        mean_act_pos = float(Mk[gt_idx].mean()) if gt_idx.size > 0 else 0.0
        sep = mean_act_pos - float(
            Mk[np.setdiff1d(np.arange(N), gt_idx)].mean()
            if N > gt_idx.size else 0.0
        )
        ax.set_xlim(0, N)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([])
        ax.set_title(
            f"{title_prefix} k={k}  mean_act_pos={mean_act_pos:.3f}  "
            f"sep={sep:.3f}",
            fontsize=9,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=80, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
        format="GIF",
    )


def _render_summary_histogram_gif(
    per_step: np.ndarray,       # (T_inf, B, n_bins)
    out_path: Path,
) -> None:
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    T, B, N = per_step.shape
    # mean per-sample per-step.
    mean_per_sample = per_step.mean(axis=-1)            # (T, B)
    global_max = float(mean_per_sample.max()) if mean_per_sample.size else 1.0
    frames: list[Image.Image] = []
    for k in range(T):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(mean_per_sample[k], bins=30, color="#4a90d9",
                edgecolor="none")
        ax.set_xlim(0.0, max(1e-3, global_max))
        ax.set_xlabel("mean(M_k)")
        ax.set_ylabel("count")
        ax.set_title(f"k={k}  n={B}")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=80, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
        format="GIF",
    )


# ─────────────────────────── hook ────────────────────────────────────


class FramewiseRolloutHook(TrainerHook):
    """Run a full DDIM rollout on a deterministic eval subset every Nth
    eval, save convergence curves + GIFs + per-step NPZs.

    The hook composes ``DDIMSampler.sample_with_intermediates`` against
    the bound model. The bound model is expected to expose ``process``,
    ``denoiser``, and ``schedule`` (i.e., a ``FramewiseDiffusionDetector``
    or a structural duck-equivalent).
    """

    def __init__(
        self,
        *,
        config: FramewiseRolloutHookConfig,
        spec: RunSpec,
        model: Any,
        adapter: Any,
        val_sampler: Any,
        train_sampler: Any | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        self._config = config
        self._spec = spec
        self._model = model
        self._adapter = adapter
        self._val_sampler = val_sampler
        self._train_sampler = train_sampler
        self._device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self._eval_count = 0
        self._sampler = None  # built lazily

    # ── public lifecycle ─────────────────────────────────────────────

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        spec.ensure()

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        self._eval_count += 1
        if self._eval_count % self._config.every_n_evals != 0:
            return
        eval_dir = self._spec.run_dir / f"eval_{state.step}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.run_once(eval_dir=eval_dir, step=state.step)
        except Exception as exc:
            # Don't take down the training loop on rollout-only errors.
            (eval_dir / "rollout_error.txt").write_text(
                f"{type(exc).__name__}: {exc}", encoding="utf-8",
            )

    # ── core entry — also called directly from tests ─────────────────

    def run_once(self, *, eval_dir: Path, step: int) -> None:
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_sampler()
        # Val rollout.
        val_payload = self._rollout(
            sampler=self._val_sampler,
            n_charts=self._config.eval_n_charts,
            n_windows=self._config.eval_n_windows_per_chart,
            seed_salt=0,
        )
        np.savez(eval_dir / "rollout_maps.npz", **val_payload["npz"])
        summary = _summary_per_sample(val_payload["metrics"])
        for k, v in summary.items():
            np.save(eval_dir / f"rollout_{k}.npy", v)
        _render_convergence_curves(
            val_payload["metrics"], eval_dir / "convergence_curves.png", step,
        )
        # Bucketed by density.
        density = val_payload["bucket_density"]
        _render_bucketed(
            val_payload["metrics"], density,
            [(0.0, 2.0, "low"), (2.0, 5.0, "mid"),
             (5.0, 10.0, "high"), (10.0, 1e9, "very high")],
            eval_dir / "convergence_by_density.png", step,
            title="Convergence by density",
        )
        # Bucketed by star.
        star = val_payload["bucket_star"]
        _render_bucketed(
            val_payload["metrics"], star,
            [(0.0, 2.0, "<2"), (2.0, 4.0, "2-4"),
             (4.0, 6.0, "4-6"), (6.0, 1e9, ">=6")],
            eval_dir / "convergence_by_star.png", step,
            title="Convergence by star",
        )
        # Bucketed by kind (sample count if no kind info — single bucket).
        kind = val_payload["bucket_kind"]
        _render_bucketed(
            val_payload["metrics"], kind,
            [(-0.5, 0.5, "DON"), (0.5, 1.5, "KA"),
             (1.5, 2.5, "BIG_DON"), (2.5, 3.5, "BIG_KA"),
             (3.5, 1e9, "other")],
            eval_dir / "convergence_by_kind.png", step,
            title="Convergence by kind",
        )
        # GIFs of representative samples.
        self._render_gifs(val_payload, eval_dir, step)
        # Summary GIF.
        _render_summary_histogram_gif(
            val_payload["per_step"],
            eval_dir / "summary_histogram.gif",
        )

        # Train noaug pass — optional.
        if self._train_sampler is not None and self._config.noaug_n_charts > 0:
            noaug_payload = self._rollout(
                sampler=self._train_sampler,
                n_charts=self._config.noaug_n_charts,
                n_windows=self._config.noaug_n_windows_per_chart,
                seed_salt=1,
            )
            np.savez(eval_dir / "noaug_rollout_maps.npz", **noaug_payload["npz"])

    # ── internals ────────────────────────────────────────────────────

    def _ensure_sampler(self) -> None:
        if self._sampler is not None:
            return
        from ..diffusion.samplers import DDIMSampler, DDIMSamplerConfig
        cfg = DDIMSamplerConfig(
            n_inference_steps=min(
                self._config.t_inf_steps,
                int(self._model.schedule.n_steps),
            ),
            eta=0.0,
            timestep_spacing="linspace",
        )
        self._sampler = DDIMSampler(
            cfg, self._model.process, self._model.denoiser,
        )

    def _rollout(
        self, *, sampler: Any, n_charts: int, n_windows: int, seed_salt: int,
    ) -> dict[str, Any]:
        import random
        rng = random.Random(self._config.seed + seed_salt)
        # Pick samples deterministically. We use sample indices instead
        # of "charts × windows" because the data sampler exposes
        # ``get_sample(i)`` directly.
        total = sampler.count_samples()
        n_target = min(n_charts * n_windows, total)
        indices = sorted(rng.sample(range(total), n_target))

        # Pick fetch — prefer raw_sample to skip augmentation.
        fetch = getattr(sampler, "raw_sample", sampler.get_sample)
        # Batch through the adapter so the model runs B at a time.
        per_step_chunks: list[np.ndarray] = []
        tgt_binary_chunks: list[np.ndarray] = []
        tgt_smoothed_chunks: list[np.ndarray] = []
        bucket_density: list[float] = []
        bucket_star: list[float] = []
        bucket_kind: list[float] = []
        # Walk in fixed batch_size chunks.
        bs = 8
        self._model.eval()
        with torch.no_grad():
            for off in range(0, len(indices), bs):
                chunk = indices[off:off + bs]
                samples = [fetch(int(i)) for i in chunk]
                inp, tgt = self._adapter.make_batch(samples, device=self._device)
                pred = self._model.predict(inp)
                cursor_tok = pred.cursor_token
                audio_features = getattr(pred, "audio_features", None)
                result = self._sampler.sample_with_intermediates(
                    cursor_tok, audio_features=audio_features,
                )
                per_step = result["per_step"].detach().cpu().numpy()  # (T, B, N)
                per_step_chunks.append(per_step)
                tgt_binary_chunks.append(
                    tgt.target_map_binary.detach().cpu().numpy()
                )
                tgt_smoothed_chunks.append(
                    tgt.target_map_smoothed.detach().cpu().numpy()
                )
                # Per-sample bucket values.
                for s in samples:
                    bucket_density.append(float(getattr(s, "density_mean", 0.0)))
                    bucket_star.append(float(getattr(s, "star_rating", 0.0)))
                    # Kind from the first valid future event, if any.
                    kind_val = -1
                    fev = getattr(s, "future_events", None)
                    fmask = getattr(s, "future_events_mask", None)
                    if fev is not None and fmask is not None:
                        for j, ev in enumerate(fev):
                            if not bool(fmask[j]):
                                k_enum = getattr(ev.kind, "value", -1)
                                try:
                                    kind_val = int(k_enum)
                                except (TypeError, ValueError):
                                    kind_val = -1
                                break
                    bucket_kind.append(float(kind_val))

        # (T, B_total, N)
        per_step = np.concatenate(per_step_chunks, axis=1)
        target_binary = np.concatenate(tgt_binary_chunks, axis=0)
        target_smoothed = np.concatenate(tgt_smoothed_chunks, axis=0)
        metrics = _per_step_metrics(
            per_step, target_binary, target_smoothed,
            threshold=self._config.decode_threshold,
            tol_frames=self._config.canonical_tolerance_frames,
        )
        return {
            "per_step": per_step,
            "target_binary": target_binary,
            "target_smoothed": target_smoothed,
            "metrics": metrics,
            "bucket_density": np.asarray(bucket_density, dtype=np.float32),
            "bucket_star": np.asarray(bucket_star, dtype=np.float32),
            "bucket_kind": np.asarray(bucket_kind, dtype=np.float32),
            "npz": {
                "per_step": per_step,
                "target_binary": target_binary,
                "target_smoothed": target_smoothed,
                "f1": metrics["f1"],
                "mse": metrics["mse"],
                "mass_at_target": metrics["mass_at_target"],
                "mass_off_target": metrics["mass_off_target"],
                "total_mass": metrics["total_mass"],
            },
        }

    def _render_gifs(
        self, payload: dict[str, Any], eval_dir: Path, step: int,
    ) -> None:
        if self._config.n_gif_samples == 0:
            return
        f1 = payload["metrics"]["f1"]                  # (B, T)
        final_f1 = f1[:, -1]
        B = final_f1.shape[0]
        n = min(self._config.n_gif_samples, B)
        if n == 0:
            return
        order = np.argsort(final_f1)                   # ascending
        # Indices: worst, p25, p50, p75, best — fewer if n < 5.
        picks: list[tuple[int, str]] = []
        if n == 1:
            picks = [(int(order[-1]), "best")]
        else:
            quantile_points = (
                [("worst", 0)]
                + [(f"p{int(round((i + 1) * 100 / n))}", int(round((i + 1) * (B - 1) / (n))))
                   for i in range(n - 2)]
                + [("best", B - 1)]
            ) if n >= 3 else [("worst", 0), ("best", B - 1)]
            seen: set[int] = set()
            for label, q in quantile_points:
                idx = int(order[max(0, min(B - 1, q))])
                if idx in seen:
                    continue
                seen.add(idx)
                picks.append((idx, label))
        gif_dir = eval_dir / "rollout_gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        per_step = payload["per_step"]                  # (T, B, N)
        target_binary = payload["target_binary"]
        for sample_idx, label in picks:
            f1_val = float(final_f1[sample_idx])
            out = gif_dir / (
                f"sample_{label}_idx{sample_idx:03d}_f1{f1_val:.3f}.gif"
            )
            _render_gif_for_sample(
                per_step[:, sample_idx, :],
                target_binary[sample_idx],
                out,
                title_prefix=f"step={step} {label}",
                threshold=self._config.decode_threshold,
            )
