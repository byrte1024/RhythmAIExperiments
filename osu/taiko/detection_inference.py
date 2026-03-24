"""Run onset detection on an audio file, output a CSV like the preprocessed data.

Usage:
  python detection_inference.py --checkpoint checkpoints/taiko_v1/best.pt --audio song.mp3
  python detection_inference.py --checkpoint checkpoints/taiko_v1/best.pt --audio song.mp3 --density-mean 5.0 --density-peak 10
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm
from collections import Counter

from detection_model import OnsetDetector, DualStreamOnsetDetector, InterleavedOnsetDetector, ContextFiLMDetector, FramewiseOnsetDetector, EventEmbeddingDetector, AdditiveOnsetDetector, RerankerOnsetDetector, LegacyOnsetDetector, Exp17OnsetDetector, Exp18OnsetDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# must match training
SAMPLE_RATE = 22050
HOP_LENGTH = 110
N_FFT = 2048
N_MELS = 80
F_MIN = 20.0
F_MAX = 8000.0
BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000  # exact: ~4.9887ms per mel frame

A_BINS = 500
B_BINS = 500
C_EVENTS = 128
N_CLASSES = 501


def load_audio_mel(audio_path, device):
    """Load audio and compute mel spectrogram on GPU."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0,
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    wav = torch.from_numpy(y).float().to(device)
    with torch.no_grad():
        mel = amp_to_db(mel_transform(wav))  # (n_mels, T)

    return mel.cpu().numpy().astype(np.float32), len(y) / SAMPLE_RATE


def extract_mel_window(mel, cursor):
    """Extract padded mel window around cursor."""
    n_mels, total_frames = mel.shape
    start = cursor - A_BINS
    end = cursor + B_BINS

    pad_left = max(0, -start)
    pad_right = max(0, end - total_frames)
    s = max(0, start)
    e = min(total_frames, end)
    window = mel[:, s:e]

    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    return window


@torch.no_grad()
def _compute_top_u(probs, max_u=5, tolerance=0.05):
    """Cluster predictions within tolerance, return top-U as (bin, confidence) list."""
    n_classes = len(probs) - 1  # exclude STOP
    order = np.argsort(probs[:n_classes])[::-1]
    clusters = []  # [centroid, total_conf, weighted_sum]
    for cls in order:
        conf = probs[cls]
        if conf < 1e-6:
            break
        matched = False
        for c in clusters:
            centroid = c[2] / c[1]  # live centroid
            if centroid > 0 and abs(cls - centroid) / centroid <= tolerance:
                c[1] += conf
                c[2] += conf * cls
                c[0] = c[2] / c[1]  # update centroid
                matched = True
                break
        if not matched:
            clusters.append([float(cls), conf, conf * cls])
        if len(clusters) >= max_u * 3:
            break
    clusters.sort(key=lambda c: c[1], reverse=True)
    return [(int(round(c[0])), c[1]) for c in clusters[:max_u]]


def _sample_from_candidates(candidates, confs, temperature, rng):
    """Temperature-sample from candidate list. Returns chosen bin index."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else 0
    confs = np.array(confs, dtype=np.float64)
    confs = np.maximum(confs, 1e-30)
    log_c = np.log(confs) / temperature
    log_c -= log_c.max()
    tempered = np.exp(log_c)
    tempered /= tempered.sum()
    return candidates[rng.choice(len(candidates), p=tempered)]


def run_inference(model, mel, conditioning, device, hop_bins=20, max_events=10000,
                  threshold=None, sample_cfg=None, addall_cfg=None):
    """Autoregressive inference: predict events one at a time.

    sample_cfg: optional dict with keys {seed, mode, temperature, topx} for
                temperature sampling. If None, uses argmax.
    addall_cfg: optional dict with keys {mode, topx, min_conf, topu_range} for
                adding all candidates above threshold. Not compatible with sample_cfg.

    Returns (events, run_stats) where run_stats has detailed inference metrics.
    """
    model.eval()
    rng = None
    if sample_cfg:
        rng = np.random.default_rng(sample_cfg["seed"])
    total_frames = mel.shape[1]
    events = []  # list of bin positions
    cursor = 0

    # Tracking stats
    stop_count = 0
    total_calls = 0
    stop_positions = []  # cursor positions where STOP was predicted
    event_offsets = []  # raw predicted offsets (before adding cursor)
    cursor_history = []  # (call_idx, cursor_pos, prediction)
    temp_history = []  # effective temperature used per prediction (sampling mode only)
    metronome_history = []  # (cursor_bin, closeness, multiplier) for metronome tracking
    candidate_history = []  # per-prediction: (cursor_bin, chosen_bin, [(cand_bin, conf_before, conf_after), ...])

    cond_tensor = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0).to(device)
    duration_s = total_frames * BIN_MS / 1000
    pbar = tqdm(total=total_frames, desc="Inference", unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")

    t_start = time.perf_counter()

    for _ in range(max_events):
        if cursor >= total_frames:
            break
        pbar.n = min(cursor, total_frames)
        pbar.set_postfix_str(f"{cursor*BIN_MS/1000:.1f}s/{duration_s:.1f}s, {len(events)} events, {stop_count} stops")
        pbar.refresh()

        total_calls += 1

        # mel window
        mel_window = extract_mel_window(mel, cursor)
        mel_tensor = torch.from_numpy(mel_window).unsqueeze(0).to(device)

        # past events as offsets from cursor
        if len(events) > 0:
            past = np.array(events[-C_EVENTS:], dtype=np.int64) - cursor
            n_past = len(past)
        else:
            past = np.array([], dtype=np.int64)
            n_past = 0

        evt_offsets = np.zeros(C_EVENTS, dtype=np.int64)
        evt_mask = np.ones(C_EVENTS, dtype=bool)
        if n_past > 0:
            evt_offsets[-n_past:] = past
            evt_mask[-n_past:] = False

        evt_tensor = torch.from_numpy(evt_offsets).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(evt_mask).unsqueeze(0).to(device)

        output = model(mel_tensor, evt_tensor, mask_tensor, cond_tensor)
        gate_logit = None
        if isinstance(output, tuple):
            logits, gate_logit = output
            # binary stop: check gate first
            gate_prob = torch.sigmoid(gate_logit).item()
            if gate_prob < 0.5:  # gate says STOP
                pred = N_CLASSES - 1
                cursor_history.append((total_calls, cursor, pred))
                stop_count += 1
                stop_positions.append(cursor)
                cursor += hop_bins
                continue
            # gate says onset — pad logits to 501 for compatibility
            logits = F.pad(logits, (0, 1), value=-10.0)
        else:
            logits = output

        if threshold is not None:
            # multi-target: threshold scan — take earliest bin above threshold
            prob = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            above = np.where(prob[:N_CLASSES - 1] >= threshold)[0]
            if len(above) == 0 or above[0] == 0:
                pred = N_CLASSES - 1  # no onset → hop forward
            else:
                pred = int(above[0])
        elif sample_cfg:
            # temperature sampling from Top-K or Top-U candidates
            prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            topx = sample_cfg["topx"]
            temp = sample_cfg["temperature"]

            max_u = topx if topx > 0 else 50  # uncapped = gather up to 50
            min_conf = sample_cfg.get("min_conf", 0.0)

            if sample_cfg["mode"] == "U":
                clusters = _compute_top_u(prob, max_u=max_u, tolerance=sample_cfg.get("topu_range", 0.05))
                cand_bins = [c[0] for c in clusters]
                cand_confs = [c[1] for c in clusters]
            else:  # mode == "K"
                topk_idx = np.argsort(prob[:N_CLASSES - 1])[::-1][:max_u]
                cand_bins = topk_idx.tolist()
                cand_confs = prob[topk_idx].tolist()

            # save raw confidences before any modifications
            cand_confs_raw = list(cand_confs)

            # filter by min confidence (normalized)
            if min_conf > 0 and len(cand_confs) > 1:
                total_conf = sum(cand_confs)
                if total_conf > 0:
                    filtered = [(b, c) for b, c in zip(cand_bins, cand_confs)
                                if c / total_conf >= min_conf]
                    if filtered:  # keep at least 1
                        cand_bins = [f[0] for f in filtered]
                        cand_confs = [f[1] for f in filtered]

            # near-weight: upweight candidates close to previous gap
            near_w = sample_cfg.get("near_weight", 0.0)
            if near_w > 0 and len(event_offsets) > 0:
                prev_gap = event_offsets[-1]
                if prev_gap > 0:
                    for ci in range(len(cand_bins)):
                        # gaussian proximity: exp(-0.5 * ((bin - prev_gap) / (prev_gap * 0.1))^2)
                        dist = abs(cand_bins[ci] - prev_gap) / max(prev_gap, 1)
                        proximity = np.exp(-0.5 * (dist / 0.1) ** 2)
                        cand_confs[ci] *= (1.0 + near_w * proximity)

            # metronome detection: measure how metronomic recent gaps are,
            # compute a multiplier, and multiply temperature by it directly.
            met_suppress_w = sample_cfg.get("metronome_suppress_weight", 0.0)
            met_temp_w = sample_cfg.get("metronome_temp_weight", 0.0)
            met_w = max(met_suppress_w, met_temp_w)
            if met_w > 0 and len(event_offsets) >= 3:
                # gather gaps from roughly the last 2s of events
                bins_2s = int(sample_cfg.get("metronome_window_ms", 2000) / BIN_MS)
                recent_gaps = []
                for g in reversed(event_offsets):
                    recent_gaps.append(g)
                    if sum(recent_gaps) >= bins_2s:
                        break
                if len(recent_gaps) >= 3:
                    recent_gaps = np.array(recent_gaps, dtype=np.float64)
                    recent_gaps = recent_gaps[recent_gaps > 0]
                    if len(recent_gaps) >= 3:
                        # cluster gaps within 5% to find the dominant gap
                        sorted_gaps = np.sort(recent_gaps)
                        clusters_g = []
                        cluster_vals = [sorted_gaps[0]]
                        for gi in range(1, len(sorted_gaps)):
                            centroid = np.mean(cluster_vals)
                            if centroid > 0 and abs(sorted_gaps[gi] - centroid) / centroid <= 0.05:
                                cluster_vals.append(sorted_gaps[gi])
                            else:
                                clusters_g.append((np.mean(cluster_vals), len(cluster_vals)))
                                cluster_vals = [sorted_gaps[gi]]
                        clusters_g.append((np.mean(cluster_vals), len(cluster_vals)))
                        clusters_g.sort(key=lambda x: x[1], reverse=True)
                        dominant_gap = clusters_g[0][0]

                        halflife = max(sample_cfg.get("metronome_halflife", 20.0), 0.1)
                        met_mode = sample_cfg.get("metronome_mode", "frame")

                        if met_mode == "pp":
                            # pp mode: distance = % of gaps outside the peak's 5% radius
                            n_outside = sum(1 for g in recent_gaps
                                            if dominant_gap <= 0 or abs(g - dominant_gap) / dominant_gap > 0.05)
                            outside_pct = n_outside / len(recent_gaps) * 100  # 0-100
                            # halflife is in pp: at outside_pct=halflife, closeness=0.5
                            closeness = 2.0 ** (-outside_pct / halflife)
                        else:
                            # frame mode: distance = abs(avg_gap - dominant_gap) in bins
                            avg_gap = recent_gaps.mean()
                            distance = abs(avg_gap - dominant_gap)
                            closeness = 2.0 ** (-distance / halflife)
                        apply_mode = sample_cfg.get("metronome_applymode", "temp")

                        if apply_mode in ("suppress", "both"):
                            suppress_strength = closeness * met_suppress_w
                            for ci in range(len(cand_bins)):
                                if dominant_gap > 0:
                                    cand_dist = abs(cand_bins[ci] - dominant_gap) / dominant_gap
                                    if cand_dist <= 0.05:
                                        cand_confs[ci] /= max(suppress_strength, 1.0)

                        multiplier = 1.0 + (met_temp_w - 1.0) * closeness
                        if apply_mode in ("temp", "both"):
                            temp = temp * multiplier
                        metronome_history.append((cursor, float(closeness), float(multiplier)))

            temp_history.append(temp)

            # check if STOP is dominant — if STOP conf > all candidates, predict STOP
            stop_conf = prob[N_CLASSES - 1]
            if len(cand_confs) == 0 or stop_conf > max(cand_confs):
                pred = N_CLASSES - 1
            elif rng is not None:
                pred = _sample_from_candidates(cand_bins, cand_confs, temp, rng)
            else:
                # argmax of weighted candidates
                pred = cand_bins[np.argmax(cand_confs)]

            # save candidate info for viewer
            cands = []
            for ci in range(len(cand_bins)):
                raw = cand_confs_raw[ci] if ci < len(cand_confs_raw) else 0.0
                final = cand_confs[ci] if ci < len(cand_confs) else 0.0
                cands.append((cand_bins[ci], float(raw), float(final)))
            candidate_history.append((cursor, pred, cands))
        elif addall_cfg:
            # addall mode: add ALL candidates above threshold as events
            prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            topx = addall_cfg["topx"]
            max_u = topx if topx > 0 else 50
            min_conf = addall_cfg.get("min_conf", 0.0)
            topu_range = addall_cfg.get("topu_range", 0.05)

            # apply metronome suppression to raw probs BEFORE clustering
            met_suppress_w = addall_cfg.get("metronome_suppress_weight", 0.0)
            if met_suppress_w > 0 and len(event_offsets) >= 3:
                bins_window = int(addall_cfg.get("metronome_window_ms", 2000) / BIN_MS)
                recent_gaps = []
                for g in reversed(event_offsets):
                    recent_gaps.append(g)
                    if sum(recent_gaps) >= bins_window:
                        break
                if len(recent_gaps) >= 3:
                    recent_gaps_arr = np.array(recent_gaps, dtype=np.float64)
                    recent_gaps_arr = recent_gaps_arr[recent_gaps_arr > 0]
                    if len(recent_gaps_arr) >= 3:
                        sorted_gaps = np.sort(recent_gaps_arr)
                        clusters_g = []
                        cluster_vals = [sorted_gaps[0]]
                        for gi in range(1, len(sorted_gaps)):
                            centroid = np.mean(cluster_vals)
                            if centroid > 0 and abs(sorted_gaps[gi] - centroid) / centroid <= 0.05:
                                cluster_vals.append(sorted_gaps[gi])
                            else:
                                clusters_g.append((np.mean(cluster_vals), len(cluster_vals)))
                                cluster_vals = [sorted_gaps[gi]]
                        clusters_g.append((np.mean(cluster_vals), len(cluster_vals)))
                        clusters_g.sort(key=lambda x: x[1], reverse=True)
                        dominant_gap = clusters_g[0][0]

                        halflife = max(addall_cfg.get("metronome_halflife", 20.0), 0.1)
                        met_mode = addall_cfg.get("metronome_mode", "frame")
                        if met_mode == "pp":
                            n_outside = sum(1 for g in recent_gaps_arr
                                            if dominant_gap <= 0 or abs(g - dominant_gap) / dominant_gap > 0.05)
                            outside_pct = n_outside / len(recent_gaps_arr) * 100
                            closeness = 2.0 ** (-outside_pct / halflife)
                        else:
                            avg_gap = recent_gaps_arr.mean()
                            distance = abs(avg_gap - dominant_gap)
                            closeness = 2.0 ** (-distance / halflife)

                        # suppress bins near dominant gap in raw probs
                        suppress_strength = max(closeness * met_suppress_w, 1.0)
                        dom_int = int(round(dominant_gap))
                        radius = max(1, int(dominant_gap * 0.05))
                        lo = max(0, dom_int - radius)
                        hi = min(N_CLASSES - 1, dom_int + radius + 1)
                        prob[lo:hi] /= suppress_strength

            if addall_cfg["mode"] == "U":
                clusters = _compute_top_u(prob, max_u=max_u, tolerance=topu_range)
                cand_bins = [c[0] for c in clusters]
                cand_confs = [c[1] for c in clusters]
            else:
                topk_idx = np.argsort(prob[:N_CLASSES - 1])[::-1][:max_u]
                cand_bins = topk_idx.tolist()
                cand_confs = prob[topk_idx].tolist()

            # save ALL candidates for viewer (before min-conf filter)
            stop_conf = prob[N_CLASSES - 1]
            cands_addall = [(b, float(c), float(c)) for b, c in zip(cand_bins, cand_confs)]

            # filter by min confidence (only affects which become events)
            if min_conf > 0 and len(cand_confs) > 0:
                total_conf = sum(cand_confs)
                if total_conf > 0:
                    filtered = [(b, c) for b, c in zip(cand_bins, cand_confs)
                                if c / total_conf >= min_conf]
                    if filtered:
                        cand_bins = [f[0] for f in filtered]
                        cand_confs = [f[1] for f in filtered]
            # if STOP dominates all candidates, treat as STOP
            if len(cand_confs) == 0 or stop_conf > max(cand_confs):
                pred = N_CLASSES - 1
                candidate_history.append((cursor, pred, cands_addall))
            else:
                # add all candidates as events, advance cursor to the furthest
                added = sorted(set(cand_bins))
                added = [b for b in added if b > 0 and b < N_CLASSES - 1]
                if added:
                    for b in added:
                        event_bin = cursor + b
                        events.append(event_bin)
                        event_offsets.append(b)
                    candidate_history.append((cursor, added[-1], cands_addall))
                    cursor = cursor + max(added)
                    cursor_history.append((total_calls, cursor, added[-1]))
                    continue
                else:
                    pred = N_CLASSES - 1
                    candidate_history.append((cursor, pred, cands_addall))
        else:
            # plain argmax — still save top candidates for viewer
            pred = logits.argmax(dim=1).item()
            prob = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
            top_u = _compute_top_u(prob, max_u=5, tolerance=0.05)
            cands = [(b, float(c), float(c)) for b, c in top_u]
            candidate_history.append((cursor, pred, cands))

        cursor_history.append((total_calls, cursor, pred))

        if pred == N_CLASSES - 1:  # STOP
            stop_count += 1
            stop_positions.append(cursor)
            cursor += hop_bins
            continue

        # predicted bin offset from cursor
        event_offsets.append(pred)
        event_bin = cursor + pred
        events.append(event_bin)
        cursor = event_bin  # move cursor to this event

    t_end = time.perf_counter()
    inference_time = t_end - t_start

    pbar.n = total_frames
    pbar.set_postfix_str(f"{duration_s:.1f}s/{duration_s:.1f}s, {len(events)} events, {stop_count} stops")
    pbar.refresh()
    pbar.close()

    # --- Compute detailed stats ---
    run_stats = _compute_run_stats(
        events, event_offsets, stop_count, stop_positions,
        total_calls, cursor_history, total_frames, duration_s, inference_time
    )

    # temperature distribution stats (sampling mode only)
    if temp_history:
        th = np.array(temp_history)
        run_stats["temperature"] = {
            "mean": float(th.mean()),
            "median": float(np.median(th)),
            "min": float(th.min()),
            "max": float(th.max()),
            "std": float(th.std()),
            "p10": float(np.percentile(th, 10)),
            "p90": float(np.percentile(th, 90)),
        }

    # metronome detection history over time
    if metronome_history:
        closeness_vals = np.array([m[1] for m in metronome_history])
        multiplier_vals = np.array([m[2] for m in metronome_history])
        run_stats["metronome"] = {
            "n_measured": len(metronome_history),
            "closeness_mean": float(closeness_vals.mean()),
            "closeness_median": float(np.median(closeness_vals)),
            "closeness_p90": float(np.percentile(closeness_vals, 90)),
            "multiplier_mean": float(multiplier_vals.mean()),
            "multiplier_max": float(multiplier_vals.max()),
        }

    # per-prediction candidate data for viewer
    if candidate_history:
        # save as list of dicts (variable-length candidates per prediction)
        # pack into a compact format: separate arrays for fixed + variable data
        run_stats["_candidate_history"] = candidate_history

    # per-prediction sampling timeline (sparse: cursor_bin → temp, closeness)
    # stored as numpy array for viewer: (N, 3) = [cursor_bin, temperature, closeness]
    if temp_history:
        sampling_timeline = np.zeros((len(temp_history), 3), dtype=np.float32)
        for i, (call_idx, cur, pred) in enumerate(cursor_history[:len(temp_history)]):
            sampling_timeline[i, 0] = cur
            sampling_timeline[i, 1] = temp_history[i]
        # fill closeness from metronome_history (sparse — not every prediction has it)
        met_idx = 0
        for i in range(len(sampling_timeline)):
            cur = sampling_timeline[i, 0]
            if met_idx < len(metronome_history) and metronome_history[met_idx][0] <= cur:
                sampling_timeline[i, 2] = metronome_history[met_idx][1]
                met_idx += 1
        run_stats["_sampling_timeline"] = sampling_timeline

    return events, run_stats


def run_framewise_inference(model, mel, conditioning, device,
                           slide_frames=200, threshold=0.3, merge_method="max",
                           merge_window=4):
    """Sliding window framewise inference with accumulation.

    Slides the window across the song, collects onset probabilities at every
    position from multiple overlapping windows, merges them, and extracts
    final onset positions.

    Args:
        slide_frames: how many mel frames to advance per step
        threshold: minimum merged probability to count as an onset
        merge_method: how to combine overlapping predictions ("max", "avg", "vote")
        merge_window: NMS window in tokens — suppress nearby peaks
    """
    model.eval()
    total_frames = mel.shape[1]
    cursor_frame = 500  # center of 1000-frame window

    # accumulator: for each mel frame, collect all onset probabilities
    # at token resolution (4x downsampled), the future window covers 500 frames = 125 tokens
    n_tokens_total = total_frames // 4 + 1
    accum_sum = np.zeros(n_tokens_total, dtype=np.float64)
    accum_count = np.zeros(n_tokens_total, dtype=np.int32)

    cond_tensor = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0).to(device)
    duration_s = total_frames * BIN_MS / 1000

    # track past events for ramp embedding (starts empty)
    detected_events = []  # mel frame positions of detected onsets

    n_windows = max(1, (total_frames - cursor_frame) // slide_frames + 1)
    pbar = tqdm(total=n_windows, desc="Framewise inference", unit="window")

    t_start = time.perf_counter()

    cursor = 0  # left edge of the 1000-frame window (so center = cursor + 500)
    window_idx = 0

    while cursor + cursor_frame < total_frames:
        pbar.update(1)
        window_center = cursor + cursor_frame
        pbar.set_postfix_str(
            f"{window_center * BIN_MS / 1000:.1f}s/{duration_s:.1f}s, "
            f"{len(detected_events)} events"
        )

        # extract mel window
        mel_window = extract_mel_window(mel, window_center)
        mel_tensor = torch.from_numpy(mel_window).unsqueeze(0).to(device)

        # past events as offsets from window center
        if len(detected_events) > 0:
            past = np.array(detected_events[-C_EVENTS:], dtype=np.int64) - window_center
            n_past = len(past)
        else:
            past = np.array([], dtype=np.int64)
            n_past = 0

        evt_offsets = np.zeros(C_EVENTS, dtype=np.int64)
        evt_mask = np.ones(C_EVENTS, dtype=bool)
        if n_past > 0:
            evt_offsets[-n_past:] = past
            evt_mask[-n_past:] = False

        evt_tensor = torch.from_numpy(evt_offsets).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(evt_mask).unsqueeze(0).to(device)

        with torch.no_grad():
            # no teacher forcing during inference — onset feedback uses own predictions
            onset_probs = model(mel_tensor, evt_tensor, mask_tensor, cond_tensor)
            probs = onset_probs.squeeze(0).cpu().numpy()  # (125,)

        # map future tokens to global token positions
        # future tokens cover frames [window_center, window_center + 500)
        # → global tokens [window_center // 4, window_center // 4 + 125)
        global_token_start = window_center // 4
        for ti in range(125):
            gt = global_token_start + ti
            if 0 <= gt < n_tokens_total:
                accum_sum[gt] += probs[ti]
                accum_count[gt] += 1

        # detect onsets from THIS window for ramp feedback in subsequent windows
        # (greedy: take peaks above threshold in this window)
        for ti in range(125):
            if probs[ti] >= threshold:
                # check if it's a local max (simple NMS)
                lo = max(0, ti - merge_window // 2)
                hi = min(125, ti + merge_window // 2 + 1)
                if probs[ti] == probs[lo:hi].max():
                    event_frame = window_center + ti * 4
                    # avoid duplicates near existing events
                    if not detected_events or abs(event_frame - detected_events[-1]) > merge_window * 4:
                        detected_events.append(event_frame)

        cursor += slide_frames
        window_idx += 1

    pbar.close()
    t_end = time.perf_counter()

    # merge accumulated probabilities
    merged = np.zeros(n_tokens_total)
    valid = accum_count > 0
    if merge_method == "max":
        merged[valid] = accum_sum[valid] / accum_count[valid]  # avg for max approx
    elif merge_method == "avg":
        merged[valid] = accum_sum[valid] / accum_count[valid]
    elif merge_method == "vote":
        # fraction of windows that predicted above threshold
        merged[valid] = accum_sum[valid] / accum_count[valid]

    # extract final onsets: threshold + NMS
    final_events = []
    for ti in range(n_tokens_total):
        if merged[ti] < threshold:
            continue
        # NMS: is this a local maximum?
        lo = max(0, ti - merge_window)
        hi = min(n_tokens_total, ti + merge_window + 1)
        if merged[ti] == merged[lo:hi].max():
            event_frame = ti * 4  # convert token back to mel frame
            final_events.append(event_frame)

    # convert to bin positions (same as mel frames in our setup)
    events = [int(e) for e in final_events]

    run_stats = {
        "total_events": len(events),
        "total_frames": int(total_frames),
        "duration_s": round(duration_s, 2),
        "inference_time_s": round(t_end - t_start, 3),
        "n_windows": window_idx,
        "slide_frames": slide_frames,
        "threshold": threshold,
        "merge_method": merge_method,
        "events_per_second": round(len(events) / duration_s, 2) if duration_s > 0 else 0,
    }

    return events, run_stats


def _compute_run_stats(events, event_offsets, stop_count, stop_positions,
                       total_calls, cursor_history, total_frames, duration_s, inference_time):
    """Compute comprehensive inference statistics."""
    stats = {
        "total_events": len(events),
        "total_frames": int(total_frames),
        "duration_s": round(duration_s, 2),
        "events_per_sec": round(len(events) / max(duration_s, 0.01), 2),
        "stop_count": stop_count,
        "total_model_calls": total_calls,
        "event_ratio": round(len(events) / max(total_calls, 1), 4),
        "stop_ratio": round(stop_count / max(total_calls, 1), 4),
        "inference_time_s": round(inference_time, 2),
        "events_per_sec_realtime": round(len(events) / max(inference_time, 0.01), 1),
        "realtime_factor": round(duration_s / max(inference_time, 0.01), 2),
    }

    # Timing per call
    stats["timing"] = {
        "inference_s": round(inference_time, 3),
        "ms_per_call": round(inference_time * 1000 / max(total_calls, 1), 2),
        "ms_per_event": round(inference_time * 1000 / max(len(events), 1), 2),
        "calls_per_sec": round(total_calls / max(inference_time, 0.01), 1),
    }

    # Event offset distribution
    if event_offsets:
        offsets = np.array(event_offsets)
        stats["event_distribution"] = {
            "min_offset": int(offsets.min()),
            "max_offset": int(offsets.max()),
            "mean_offset": round(float(offsets.mean()), 2),
            "median_offset": int(np.median(offsets)),
            "std_offset": round(float(offsets.std()), 2),
            "mean_offset_ms": round(float(offsets.mean()) * BIN_MS, 2),
            "median_offset_ms": round(float(np.median(offsets)) * BIN_MS, 2),
        }

        # Offset histogram (top 15 most common)
        offset_counts = Counter(event_offsets)
        stats["event_distribution"]["top_offsets"] = [
            {"offset": k, "offset_ms": round(k * BIN_MS, 1), "count": v}
            for k, v in offset_counts.most_common(15)
        ]

        # What % of events are at offset 0 (same position = potential stutter)
        zero_offsets = sum(1 for o in event_offsets if o == 0)
        stats["event_distribution"]["zero_offset_count"] = zero_offsets
        stats["event_distribution"]["zero_offset_pct"] = round(zero_offsets / len(event_offsets) * 100, 2)
    else:
        stats["event_distribution"] = {}

    # Inter-onset intervals
    if len(events) > 1:
        event_times_ms = [e * BIN_MS for e in events]
        iois = np.diff(event_times_ms)
        stats["ioi"] = {
            "min_ms": round(float(iois.min()), 1),
            "max_ms": round(float(iois.max()), 1),
            "mean_ms": round(float(iois.mean()), 1),
            "median_ms": round(float(np.median(iois)), 1),
            "std_ms": round(float(iois.std()), 1),
        }

        # Very short IOIs (< 20ms) = potential double-triggers
        short_iois = sum(1 for i in iois if i < 20)
        stats["ioi"]["short_ioi_count"] = int(short_iois)
        stats["ioi"]["short_ioi_pct"] = round(short_iois / len(iois) * 100, 2)

        # Very long IOIs (> 2s) = potential gaps
        long_iois = sum(1 for i in iois if i > 2000)
        stats["ioi"]["long_gap_count"] = int(long_iois)

        # IOI histogram buckets (10ms resolution, top entries)
        ioi_buckets = Counter(int(round(i / 10)) * 10 for i in iois)
        stats["ioi"]["histogram"] = [
            {"ms": k, "count": v} for k, v in sorted(ioi_buckets.items()) if v >= 2
        ]

        # Estimated BPM from most common IOI
        common_iois = Counter(int(round(i / 5)) * 5 for i in iois if 100 < i < 1500)
        if common_iois:
            common_ioi_ms = common_iois.most_common(1)[0][0]
            stats["ioi"]["estimated_bpm"] = round(60000 / common_ioi_ms, 1)
    else:
        stats["ioi"] = {}

    # Density over time (1-second windows)
    if events:
        event_times_ms = [e * BIN_MS for e in events]
        density_timeline = []
        for t_start in range(0, int(duration_s * 1000), 1000):
            t_end = t_start + 1000
            count = sum(1 for t in event_times_ms if t_start <= t < t_end)
            density_timeline.append({"time_s": t_start / 1000, "density": count})

        densities = [d["density"] for d in density_timeline]
        stats["density"] = {
            "mean": round(np.mean(densities), 2),
            "peak": int(np.max(densities)),
            "std": round(np.std(densities), 2),
            "min": int(np.min(densities)),
            "cv": round(float(np.std(densities) / max(np.mean(densities), 0.01)), 3),
            "timeline": density_timeline,
        }

        # Silence regions (0 events for 2+ seconds)
        silence_stretches = []
        current_silence_start = None
        for d in density_timeline:
            if d["density"] == 0:
                if current_silence_start is None:
                    current_silence_start = d["time_s"]
            else:
                if current_silence_start is not None:
                    length = d["time_s"] - current_silence_start
                    if length >= 2:
                        silence_stretches.append({
                            "start_s": current_silence_start,
                            "end_s": d["time_s"],
                            "duration_s": round(length, 1),
                        })
                    current_silence_start = None
        stats["density"]["silence_regions"] = silence_stretches

        # Dense regions (>2x mean density for 3+ seconds)
        mean_d = np.mean(densities)
        dense_stretches = []
        current_dense_start = None
        for d in density_timeline:
            if d["density"] > mean_d * 2:
                if current_dense_start is None:
                    current_dense_start = d["time_s"]
            else:
                if current_dense_start is not None:
                    length = d["time_s"] - current_dense_start
                    if length >= 3:
                        dense_stretches.append({
                            "start_s": current_dense_start,
                            "end_s": d["time_s"],
                            "duration_s": round(length, 1),
                        })
                    current_dense_start = None
        stats["density"]["dense_regions"] = dense_stretches
    else:
        stats["density"] = {}

    # STOP pattern analysis
    if stop_positions:
        stop_times_ms = [s * BIN_MS for s in stop_positions]
        stats["stop_analysis"] = {
            "total_stops": stop_count,
            "first_stop_ms": round(stop_times_ms[0], 1),
            "last_stop_ms": round(stop_times_ms[-1], 1),
        }

        # Consecutive STOP runs
        runs = []
        current_run = 1
        for i in range(1, len(stop_positions)):
            # Check if consecutive (same cursor hop apart)
            if cursor_history:
                # Approximate: if stop positions are close together, they're a run
                gap = (stop_positions[i] - stop_positions[i-1]) * BIN_MS
                if gap < 200:  # within ~200ms = likely consecutive STOPs
                    current_run += 1
                else:
                    if current_run > 1:
                        runs.append(current_run)
                    current_run = 1
        if current_run > 1:
            runs.append(current_run)

        stats["stop_analysis"]["consecutive_runs"] = len(runs)
        stats["stop_analysis"]["longest_run"] = max(runs) if runs else 0
        stats["stop_analysis"]["avg_run_length"] = round(np.mean(runs), 1) if runs else 0

        # Average gap between stops
        if len(stop_times_ms) > 1:
            stop_gaps = np.diff(stop_times_ms)
            stats["stop_analysis"]["avg_stop_gap_ms"] = round(float(np.mean(stop_gaps)), 1)
            stats["stop_analysis"]["min_stop_gap_ms"] = round(float(np.min(stop_gaps)), 1)
            stats["stop_analysis"]["max_stop_gap_ms"] = round(float(np.max(stop_gaps)), 1)
    else:
        stats["stop_analysis"] = {}

    # Copy avg_stop_gap to top level for viewer
    stats["avg_stop_gap_ms"] = stats.get("stop_analysis", {}).get("avg_stop_gap_ms", 0)

    return stats


def events_to_csv(events, output_path, audio_name=""):
    """Write events as a CSV matching the preprocessed format."""
    with open(output_path, "w", encoding="utf-8") as f:
        if audio_name:
            f.write(f"# audio: {audio_name}\n")
        f.write("time_ms,type\n")
        for bin_idx in events:
            time_ms = int(bin_idx * BIN_MS)
            f.write(f"{time_ms},predicted\n")
    print(f"Wrote {len(events)} events to {output_path}")


def print_stats_report(stats):
    """Print a detailed human-readable stats report to console."""
    print("\n" + "=" * 70)
    print("  INFERENCE REPORT")
    print("=" * 70)

    print(f"\n  Audio duration:      {stats['duration_s']:.1f}s")
    print(f"  Total events:        {stats['total_events']}")
    print(f"  Events/sec:          {stats['events_per_sec']:.1f}")
    print(f"  Total model calls:   {stats['total_model_calls']}")
    print(f"  STOP predictions:    {stats['stop_count']} ({stats['stop_ratio']*100:.1f}% of calls)")
    print(f"  Event predictions:   {stats['total_events']} ({stats['event_ratio']*100:.1f}% of calls)")

    print(f"\n  --- Timing ---")
    t = stats["timing"]
    print(f"  Inference wall time: {t['inference_s']:.2f}s")
    print(f"  Realtime factor:     {stats['realtime_factor']:.1f}x {'(faster)' if stats['realtime_factor'] > 1 else '(slower)'}")
    print(f"  ms/model call:       {t['ms_per_call']:.2f}")
    print(f"  ms/event:            {t['ms_per_event']:.2f}")
    print(f"  Calls/sec:           {t['calls_per_sec']:.0f}")

    ed = stats.get("event_distribution", {})
    if ed:
        print(f"\n  --- Event Offset Distribution ---")
        print(f"  Range:     {ed['min_offset']}-{ed['max_offset']} bins ({ed['min_offset']*BIN_MS:.0f}-{ed['max_offset']*BIN_MS:.0f}ms)")
        print(f"  Mean:      {ed['mean_offset']:.1f} bins ({ed['mean_offset_ms']:.1f}ms)")
        print(f"  Median:    {ed['median_offset']} bins ({ed['median_offset_ms']:.1f}ms)")
        print(f"  Std:       {ed['std_offset']:.1f} bins")
        if ed.get("zero_offset_count", 0) > 0:
            print(f"  Zero-offset events: {ed['zero_offset_count']} ({ed['zero_offset_pct']:.1f}%) [potential stutter]")

        if ed.get("top_offsets"):
            print(f"  Top offsets:")
            for entry in ed["top_offsets"][:8]:
                bar = "#" * min(40, entry["count"])
                print(f"    {entry['offset']:4d} ({entry['offset_ms']:6.1f}ms): {entry['count']:4d} {bar}")

    ioi = stats.get("ioi", {})
    if ioi:
        print(f"\n  --- Inter-Onset Intervals ---")
        print(f"  Min:       {ioi['min_ms']:.1f}ms")
        print(f"  Max:       {ioi['max_ms']:.1f}ms")
        print(f"  Mean:      {ioi['mean_ms']:.1f}ms")
        print(f"  Median:    {ioi['median_ms']:.1f}ms")
        print(f"  Std:       {ioi['std_ms']:.1f}ms")
        if ioi.get("short_ioi_count", 0) > 0:
            print(f"  Short IOIs (<20ms): {ioi['short_ioi_count']} ({ioi['short_ioi_pct']:.1f}%) [double-triggers]")
        if ioi.get("long_gap_count", 0) > 0:
            print(f"  Long gaps (>2s):    {ioi['long_gap_count']}")
        if ioi.get("estimated_bpm"):
            print(f"  Est. BPM:  {ioi['estimated_bpm']:.0f}")

    d = stats.get("density", {})
    if d:
        print(f"\n  --- Density Profile ---")
        print(f"  Mean:      {d['mean']:.1f} events/s")
        print(f"  Peak:      {d['peak']} events/s")
        print(f"  Min:       {d['min']} events/s")
        print(f"  Std:       {d['std']:.1f}")
        print(f"  CV:        {d['cv']:.3f}")

        if d.get("silence_regions"):
            print(f"  Silence regions ({len(d['silence_regions'])}):")
            for sr in d["silence_regions"][:5]:
                print(f"    {sr['start_s']:.1f}s - {sr['end_s']:.1f}s ({sr['duration_s']:.1f}s)")

        if d.get("dense_regions"):
            print(f"  Dense regions ({len(d['dense_regions'])}; >2x mean for 3+s):")
            for dr in d["dense_regions"][:5]:
                print(f"    {dr['start_s']:.1f}s - {dr['end_s']:.1f}s ({dr['duration_s']:.1f}s)")

    sa = stats.get("stop_analysis", {})
    if sa:
        print(f"\n  --- STOP Pattern ---")
        print(f"  Total STOPs:       {sa['total_stops']}")
        if sa.get("avg_stop_gap_ms"):
            print(f"  Avg gap between:   {sa['avg_stop_gap_ms']:.0f}ms")
            print(f"  Min gap:           {sa['min_stop_gap_ms']:.0f}ms")
            print(f"  Max gap:           {sa['max_stop_gap_ms']:.0f}ms")
        if sa.get("consecutive_runs", 0) > 0:
            print(f"  Consecutive runs:  {sa['consecutive_runs']} (longest: {sa['longest_run']})")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run onset detection inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", default=None, help="Path to audio file (opens file browser if not set)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: {audio_stem}_predicted.csv)")
    parser.add_argument("--density-mean", type=float, default=4.3, help="Target mean density (events/sec)")
    parser.add_argument("--density-peak", type=float, default=8.0, help="Target peak density")
    parser.add_argument("--density-std", type=float, default=1.5, help="Target density std")
    parser.add_argument("--hop-ms", type=float, default=100, help="Cursor hop on STOP prediction (ms, default 100)")
    parser.add_argument("--slide-frames", type=int, default=200, help="Slide step for framewise inference (mel frames, default 200)")
    parser.add_argument("--fw-threshold", type=float, default=0.3, help="Onset threshold for framewise inference (default 0.3)")
    parser.add_argument("--fw-merge", default="max", choices=["max", "avg", "vote"], help="Merge method for overlapping framewise windows")
    parser.add_argument("--random-seed", type=int, default=None, help="Enable temperature sampling with this seed (default: off/argmax)")
    parser.add_argument("--random-mode", default="U", choices=["U", "K"], help="Sampling candidate mode: U=Top-Unique, K=Top-K (default: U)")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature 0.01-100 (default: 0.75)")
    parser.add_argument("--topx", type=int, default=5, help="Max candidates for sampling, 0=uncapped (default: 5)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Min normalized confidence to include a candidate (default: 0=off)")
    parser.add_argument("--near-weight", type=float, default=0.0, help="Upweight candidates near previous gap (0=off, 1=strong, default: 0)")
    parser.add_argument("--metronome-weight", type=str, default="0", help="Metronome weight: single value or suppress,temp for both mode (default: 0)")
    parser.add_argument("--metronome-halflife", type=float, default=20.0, help="Half-life: in pp mode, pp outside peak where closeness halves; in frame mode, bin distance (default: 20)")
    parser.add_argument("--metronome-window", type=float, default=2.0, help="Metronome lookback window in seconds (default: 2.0)")
    parser.add_argument("--metronome-mode", default="frame", choices=["frame", "pp"], help="Closeness metric: frame=bin distance avg-to-peak, pp=percentage of gaps outside peak's 5%% radius (default: frame)")
    parser.add_argument("--metronome-applymode", default="temp", choices=["temp", "suppress", "both"], help="How to apply metronome detection: temp=multiply temperature, suppress=downweight candidates near dominant gap, both=suppress+temp (default: temp)")
    parser.add_argument("--addall", action="store_true", default=False, help="Add ALL candidates above threshold as events (not compatible with --random-seed)")
    parser.add_argument("--topu-range", type=float, default=0.05, help="Top-U merge tolerance (default: 0.05 = 5%%)")
    parser.add_argument("--andlaunch", action="store_true", help="Launch viewer after inference")
    parser.add_argument("--gif", default=None, help="Path to GIF for beat-synced animation in viewer")
    parser.add_argument("--gif-cycles", type=int, default=1, help="Events per full GIF cycle (default: 1)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # pick audio file
    if args.audio is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        args.audio = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.ogg *.wav *.flac *.m4a"), ("All files", "*.*")],
        )
        root.destroy()
        if not args.audio:
            print("No file selected, exiting.")
            return

    t_total_start = time.perf_counter()

    # load checkpoint
    t0 = time.perf_counter()
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]

    # Detect checkpoint era by state dict keys and args
    state_keys = set(ckpt["model"].keys())
    has_fusion_layers = any("fusion_layers" in k for k in state_keys)
    has_gap_encoder = any("gap_encoder." in k for k in state_keys)
    has_output_head = any("context_path.output_head" in k for k in state_keys)
    has_gap_layers = any("context_path.gap_layers" in k for k in state_keys)
    has_exp18_event_layers = any("context_path.event_layers" in k for k in state_keys)
    is_legacy = "top_k" not in ckpt_args and not has_fusion_layers and not has_gap_encoder

    has_cross_attn_fusion = any("cross_attn_fusion." in k for k in state_keys)
    has_interleaved = any("audio_self_layers." in k for k in state_keys)
    has_context_film = any("fusion_context_film." in k for k in state_keys)
    has_framewise = any("onset_head." in k and "onset_feedback_emb" in str(state_keys) for k in state_keys)
    # simpler check: framewise has onset_feedback_emb
    has_framewise = "onset_feedback_emb" in state_keys

    has_event_embed = "event_presence_emb" in state_keys

    if has_event_embed:
        ModelClass = EventEmbeddingDetector  # exp 42+ (event embeddings)
    elif has_framewise:
        ModelClass = FramewiseOnsetDetector  # exp 38+ (framewise)
    elif has_context_film:
        ModelClass = ContextFiLMDetector  # exp 34+ (context FiLM)
    elif has_interleaved:
        ModelClass = InterleavedOnsetDetector  # exp 33 (interleaved)
    elif has_cross_attn_fusion:
        ModelClass = DualStreamOnsetDetector  # exp 31-32 (dual stream)
    elif has_fusion_layers or has_gap_encoder:
        ModelClass = OnsetDetector  # exp 25-30 (unified fusion)
    elif is_legacy:
        ModelClass = LegacyOnsetDetector  # exp 11-16
    elif has_gap_layers and has_output_head:
        ModelClass = AdditiveOnsetDetector  # exp 24 (additive context)
    elif has_gap_layers:
        ModelClass = RerankerOnsetDetector  # exp 19-23 (gap-based reranker)
    elif has_exp18_event_layers:
        ModelClass = Exp18OnsetDetector  # exp 18
    else:
        ModelClass = Exp17OnsetDetector  # exp 17

    # Build model kwargs based on checkpoint era
    if ModelClass == EventEmbeddingDetector:
        # detect gap_ratios: event_proj input is 5*d_model (1920) vs 3*d_model (1152)
        event_proj_key = next((k for k in state_keys if "event_proj.0.weight" in k), None)
        has_gap_ratios = False
        if event_proj_key:
            w = ckpt["model"][event_proj_key]
            if w.shape[1] > ckpt_args.get("d_model", 384) * 3:
                has_gap_ratios = True
        has_binary_stop = "gate_head.0.weight" in state_keys
        model_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            n_layers=ckpt_args.get("enc_layers", 4) + ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
            gap_ratios=has_gap_ratios,
            binary_stop=has_binary_stop,
        )
    elif ModelClass == FramewiseOnsetDetector:
        # exp 38+: framewise onset detection
        model_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            n_layers=ckpt_args.get("enc_layers", 4) + ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args.get("n_heads", 8),
        )
    elif ModelClass == ContextFiLMDetector:
        # exp 34+: context FiLM conditioning
        model_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            enc_layers=ckpt_args.get("enc_layers", 4),
            gap_enc_layers=ckpt_args.get("gap_enc_layers", 2),
            fusion_layers=ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
            snippet_frames=ckpt_args.get("snippet_frames", 10),
        )
    elif ModelClass == InterleavedOnsetDetector:
        # exp 33: interleaved self+cross attention
        model_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            n_blocks=ckpt_args.get("n_blocks", 4),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
            snippet_frames=ckpt_args.get("snippet_frames", 10),
        )
    elif ModelClass == DualStreamOnsetDetector:
        # exp 31-32: dual stream with cross-attention fusion
        model_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            enc_layers=ckpt_args.get("enc_layers", 4),
            gap_enc_layers=ckpt_args.get("gap_enc_layers", 4),
            cross_attn_layers=ckpt_args.get("cross_attn_layers", 2),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
            snippet_frames=ckpt_args.get("snippet_frames", 10),
        )
    elif ModelClass == OnsetDetector:
        # exp 25-30: unified fusion
        model_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            enc_layers=ckpt_args.get("enc_layers", 4),
            gap_enc_layers=ckpt_args.get("gap_enc_layers", 2),
            fusion_layers=ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
            snippet_frames=ckpt_args.get("snippet_frames", 10),
        )
    else:
        base_kwargs = dict(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            d_event=ckpt_args.get("d_event", 128),
            enc_layers=ckpt_args.get("enc_layers", 4),
            enc_event_layers=ckpt_args.get("enc_event_layers", 2),
            audio_path_layers=ckpt_args.get("audio_path_layers", 2),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
        )
        if ModelClass == LegacyOnsetDetector:
            base_kwargs["context_path_layers"] = ckpt_args.get("context_path_layers", 3)
        elif ModelClass == Exp17OnsetDetector:
            base_kwargs["context_path_layers"] = ckpt_args.get("context_path_layers", 3)
            base_kwargs["top_k"] = ckpt_args.get("top_k", 20)
        elif ModelClass == Exp18OnsetDetector:
            base_kwargs["context_event_layers"] = ckpt_args.get("context_event_layers", 2)
            base_kwargs["context_select_layers"] = ckpt_args.get("context_select_layers", 2)
            base_kwargs["top_k"] = ckpt_args.get("top_k", 20)
        elif ModelClass == RerankerOnsetDetector:
            base_kwargs["d_ctx"] = ckpt_args.get("d_ctx", 192)
            base_kwargs["context_gap_layers"] = ckpt_args.get("context_gap_layers", 2)
            base_kwargs["context_select_layers"] = ckpt_args.get("context_select_layers", 2)
            base_kwargs["top_k"] = ckpt_args.get("top_k", 20)
            base_kwargs["snippet_frames"] = ckpt_args.get("snippet_frames", 10)
        elif ModelClass == AdditiveOnsetDetector:
            base_kwargs["d_ctx"] = ckpt_args.get("d_ctx", 192)
            base_kwargs["context_gap_layers"] = ckpt_args.get("context_gap_layers", 2)
            base_kwargs["snippet_frames"] = ckpt_args.get("snippet_frames", 10)
        model_kwargs = base_kwargs

    model = ModelClass(**model_kwargs).to(args.device)
    state = ckpt["model"]
    if ModelClass == RerankerOnsetDetector:
        # Handle score_proj shape mismatches between exp 19-23 variants
        score_proj_key = "context_path.score_proj.0.weight"
        current_shape = model.state_dict().get(score_proj_key, torch.empty(0)).shape
        ckpt_shape = state.get(score_proj_key, torch.empty(0)).shape
        if current_shape != ckpt_shape:
            state = {k: v for k, v in state.items()
                     if "score_proj" not in k and "candidate_combine" not in k}
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    if ModelClass == EventEmbeddingDetector:
        print("  (exp 42+ checkpoint - event embedding detector)")
    elif ModelClass == FramewiseOnsetDetector:
        print("  (exp 38+ checkpoint - framewise onset detection)")
    elif ModelClass == ContextFiLMDetector:
        print("  (exp 34+ checkpoint - context FiLM conditioning)")
    elif ModelClass == InterleavedOnsetDetector:
        print("  (exp 33 checkpoint - interleaved self+cross attention)")
    elif ModelClass == DualStreamOnsetDetector:
        print("  (exp 31-32 checkpoint - dual stream cross-attention fusion)")
    elif ModelClass == OnsetDetector:
        print("  (exp 25-30 checkpoint - unified audio+gap fusion)")
    elif ModelClass == AdditiveOnsetDetector:
        print("  (exp 24 checkpoint - additive context logits)")
    elif ModelClass == LegacyOnsetDetector:
        print("  (legacy checkpoint - exp 11-16 additive logits)")
    elif ModelClass == Exp17OnsetDetector:
        print("  (exp 17 checkpoint - shared-gradient top-K reranking)")
    elif ModelClass == Exp18OnsetDetector:
        print("  (exp 18 checkpoint - two-stage stop-gradient reranking)")
    t_model_load = time.perf_counter() - t0
    print(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, acc={ckpt.get('val_metrics', {}).get('accuracy', 0):.3f}")

    # load audio
    t0 = time.perf_counter()
    print(f"Loading audio: {args.audio}")
    mel, duration = load_audio_mel(args.audio, args.device)
    t_audio_load = time.perf_counter() - t0
    print(f"  Duration: {duration:.1f}s, {mel.shape[1]} mel frames")

    # run inference
    conditioning = [args.density_mean, args.density_peak, args.density_std]
    print(f"  Conditioning: mean={conditioning[0]}, peak={conditioning[1]}, std={conditioning[2]}")

    if ModelClass == FramewiseOnsetDetector:
        print(f"  Framewise mode: slide={args.slide_frames}frames, threshold={args.fw_threshold}, merge={args.fw_merge}")
        events, run_stats = run_framewise_inference(
            model, mel, conditioning, args.device,
            slide_frames=args.slide_frames,
            threshold=args.fw_threshold,
            merge_method=args.fw_merge,
        )
    else:
        hop_bins = max(1, int(args.hop_ms / BIN_MS))
        print(f"  STOP hop: {args.hop_ms}ms = {hop_bins} bins")
        sample_cfg = None
        # build sample_cfg if any weighting/sampling features are enabled
        met_w_str = args.metronome_weight
        if "," in met_w_str:
            met_suppress_w, met_temp_w = [float(x) for x in met_w_str.split(",", 1)]
        else:
            met_suppress_w = met_temp_w = float(met_w_str)

        has_weighting = (args.near_weight > 0 or met_suppress_w > 0 or met_temp_w > 0
                         or args.random_seed is not None)
        if has_weighting:
            sample_cfg = {
                "seed": args.random_seed,
                "mode": args.random_mode,
                "temperature": args.temperature,
                "topx": args.topx,
                "near_weight": args.near_weight,
                "min_conf": args.min_conf,
                "topu_range": args.topu_range,
                "metronome_suppress_weight": met_suppress_w,
                "metronome_temp_weight": met_temp_w,
                "metronome_halflife": args.metronome_halflife,
                "metronome_window_ms": args.metronome_window * 1000,
                "metronome_mode": args.metronome_mode,
                "metronome_applymode": args.metronome_applymode,
            }
            if args.random_seed is not None:
                print(f"  Sampling: mode={args.random_mode} T={args.temperature} top={args.topx} near={args.near_weight} met_w={met_w_str} met_hl={args.metronome_halflife} met_mode={args.metronome_mode} met_apply={args.metronome_applymode} seed={args.random_seed}")
            else:
                print(f"  Weighted argmax: mode={args.random_mode} top={args.topx} near={args.near_weight} met_w={met_w_str} met_hl={args.metronome_halflife} met_mode={args.metronome_mode} met_apply={args.metronome_applymode}")

        addall_cfg = None
        if args.addall:
            if args.random_seed is not None:
                print("  WARNING: --addall is not compatible with --random-seed, ignoring --random-seed")
                sample_cfg = None
            addall_cfg = {
                "mode": args.random_mode,
                "topx": args.topx,
                "min_conf": args.min_conf,
                "topu_range": args.topu_range,
                "metronome_suppress_weight": met_suppress_w,
                "metronome_halflife": args.metronome_halflife,
                "metronome_window_ms": args.metronome_window * 1000,
                "metronome_mode": args.metronome_mode,
            }
            print(f"  AddAll: mode={args.random_mode} top={args.topx} min_conf={args.min_conf} topu_range={args.topu_range} met_suppress={met_suppress_w}")

        events, run_stats = run_inference(model, mel, conditioning, args.device, hop_bins=hop_bins,
                                          sample_cfg=sample_cfg, addall_cfg=addall_cfg)
    print(f"  Predicted {len(events)} events ({len(events) / duration:.1f}/s)")

    # Add extra info to stats
    run_stats["checkpoint"] = os.path.basename(args.checkpoint)
    run_stats["epoch"] = ckpt["epoch"]
    run_stats["val_loss"] = round(ckpt["val_loss"], 4)
    run_stats["val_accuracy"] = round(ckpt.get("val_metrics", {}).get("accuracy", 0), 4)
    run_stats["device"] = args.device
    run_stats["audio_file"] = os.path.basename(args.audio)
    run_stats["conditioning"] = {
        "mean": args.density_mean, "peak": args.density_peak, "std": args.density_std,
    }
    run_stats["hop_ms"] = args.hop_ms
    run_stats["hop_bins"] = hop_bins
    run_stats["timing"]["model_load_s"] = round(t_model_load, 3)
    run_stats["timing"]["audio_load_s"] = round(t_audio_load, 3)
    run_stats["timing"]["total_s"] = round(time.perf_counter() - t_total_start, 3)

    # Print report
    print_stats_report(run_stats)

    # write CSV
    tmp_dir = None
    if args.output is None:
        if args.andlaunch:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="beatdetect_")
            stem = os.path.splitext(os.path.basename(args.audio))[0]
            args.output = os.path.join(tmp_dir, f"{stem}_predicted.csv")
        else:
            stem = os.path.splitext(os.path.basename(args.audio))[0]
            args.output = os.path.join(SCRIPT_DIR, f"{stem}_predicted.csv")

    events_to_csv(events, args.output, audio_name=os.path.abspath(args.audio))

    # Save mel spectrogram and waveform envelope for viewer
    mel_npy_path = args.output.replace(".csv", "_mel.npy")
    np.save(mel_npy_path, mel)
    print(f"Wrote mel spectrogram to {mel_npy_path} (shape {mel.shape})")

    # Save waveform envelope (downsampled amplitude for visualization)
    wave_npy_path = args.output.replace(".csv", "_wave.npy")
    y_raw, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
    # Downsample to ~1 sample per mel frame using max-abs envelope
    hop = HOP_LENGTH
    n_frames = mel.shape[1]
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(y_raw))
        if start < len(y_raw):
            envelope[i] = np.max(np.abs(y_raw[start:end]))
    np.save(wave_npy_path, envelope)
    print(f"Wrote waveform envelope to {wave_npy_path} ({n_frames} frames)")

    # Save candidate history for viewer (variable-length per prediction)
    candidates_path = None
    if "_candidate_history" in run_stats:
        candidates_path = args.output.replace(".csv", "_candidates.json")
        # compact format: list of [cursor, chosen, [[bin, raw_conf, final_conf], ...]]
        compact = []
        for cursor_bin, chosen, cands in run_stats["_candidate_history"]:
            compact.append([int(cursor_bin), int(chosen),
                           [[int(b), round(r, 5), round(f, 5)] for b, r, f in cands]])
        with open(candidates_path, "w", encoding="utf-8") as f:
            json.dump(compact, f)
        print(f"Wrote candidate history to {candidates_path} ({len(compact)} predictions)")

    # Save sampling timeline if present (temperature + metronome over time)
    sampling_npy_path = None
    if "_sampling_timeline" in run_stats:
        sampling_npy_path = args.output.replace(".csv", "_sampling.npy")
        np.save(sampling_npy_path, run_stats["_sampling_timeline"])
        print(f"Wrote sampling timeline to {sampling_npy_path} ({len(run_stats['_sampling_timeline'])} points)")

    # Write stats JSON alongside CSV
    stats_path = args.output.replace(".csv", "_stats.json")
    # Remove non-serializable / large internal arrays from JSON
    json_stats = {k: v for k, v in run_stats.items() if not k.startswith("_")}
    if "density" in json_stats and "timeline" in json_stats["density"]:
        json_stats["density"] = {k: v for k, v in json_stats["density"].items() if k != "timeline"}
    if "ioi" in json_stats and "histogram" in json_stats["ioi"]:
        # Keep only top 20 histogram entries
        json_stats["ioi"]["histogram"] = json_stats["ioi"]["histogram"][:20]

    with open(stats_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")

    if args.andlaunch:
        import subprocess, sys
        viewer_path = os.path.join(SCRIPT_DIR, "viewer.py")
        print(f"\nLaunching viewer: {args.output}")
        cmd = [sys.executable, viewer_path, args.output, "--audio", args.audio,
               "--stats-json", stats_path,
               "--mel-npy", mel_npy_path, "--wave-npy", wave_npy_path]
        if sampling_npy_path:
            cmd.extend(["--sampling-npy", sampling_npy_path])
        if candidates_path:
            cmd.extend(["--candidates-json", candidates_path])
        if args.gif:
            cmd.extend(["--gif", args.gif, "--gif-cycles", str(args.gif_cycles)])
        subprocess.run(cmd)

    if tmp_dir is not None:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temp directory: {tmp_dir}")


if __name__ == "__main__":
    main()
