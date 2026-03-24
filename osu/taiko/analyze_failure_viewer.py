"""Render a video showing 25 shared failure cases.

For each failure: scrolls through ~2.5s of context like the taiko viewer,
then shows the predicted position (red) vs target position (green ghost).

Usage:
    python analyze_failure_viewer.py --label exp44 --subsample 8
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import subprocess
import wave
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_train import OnsetDataset, split_by_song, N_CLASSES, C_EVENTS, A_BINS, B_BINS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
BIN_MS = 4.988662131519274


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="exp44")
    parser.add_argument("--n-cases", type=int, default=25)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "experiments"))
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    import pygame

    data = np.load(os.path.join(args.data_dir, f"val_data_{args.label}.npz"))
    scores, preds, targets = data["scores"], data["preds"], data["targets"]
    print(f"Loaded {len(scores)} samples from {args.label}")

    # pick worst non-STOP failures, evenly spaced from bottom 20%
    non_stop = targets < (N_CLASSES - 1)
    ns_idx = np.where(non_stop)[0]
    ns_scores = scores[ns_idx]
    worst = ns_scores.argsort()
    n_worst = len(worst) // 5
    step = max(1, n_worst // args.n_cases)
    selected = [ns_idx[worst[i * step]] for i in range(args.n_cases)]
    print(f"Selected {len(selected)} cases (scores: {scores[selected[0]]:.3f} to {scores[selected[-1]]:.3f})")

    # load val dataset
    with open(os.path.join(DS_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)
    val_ds = OnsetDataset(manifest, DS_DIR, val_idx, augment=False, subsample=args.subsample,
                          multi_target=False)

    # render
    pygame.init()
    W, H = 1000, 350
    surface = pygame.Surface((W, H))
    font = pygame.font.SysFont("consolas", 14)
    font_big = pygame.font.SysFont("consolas", 16, bold=True)
    font_small = pygame.font.SysFont("consolas", 11)

    if args.output is None:
        args.output = os.path.join(args.data_dir, "experiment_48", f"failures_{args.label}.mp4")

    proc = subprocess.Popen([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", args.output,
    ], stdin=subprocess.PIPE)

    # mel colormap
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        cmap[i] = (int(68 + t * 120), int(1 + t * 180), int(84 + t * 90))

    HIT_X = 350
    SCROLL_PX_PER_MS = 0.4
    PF_TOP = 60
    PF_H = 80
    PF_CY = PF_TOP + PF_H // 2
    MEL_TOP = PF_TOP + PF_H + 5
    MEL_H = 100

    SR = 22050
    HOP = 110
    N_FFT = 2048
    N_MELS = 80
    F_MIN = 20.0
    F_MAX = 8000.0

    def mel_to_audio(mel_db, sr=SR, hop=HOP, n_fft=N_FFT, n_mels=N_MELS, n_iter=32):
        """Invert mel spectrogram to audio via Griffin-Lim."""
        import librosa
        # mel_db is (n_mels, T) in dB scale — convert back to power
        mel_power = librosa.db_to_power(mel_db)
        # invert mel filterbank
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=F_MIN, fmax=F_MAX)
        mel_basis_inv = np.linalg.pinv(mel_basis)
        spec = np.maximum(mel_basis_inv @ mel_power, 0)
        # Griffin-Lim
        audio = librosa.griffinlim(spec, n_iter=n_iter, hop_length=hop, n_fft=n_fft)
        return audio

    # pre-compute per-case timing and synthesize audio
    print("Computing timing and synthesizing audio...")
    from tqdm import tqdm
    case_info = []  # (sample_idx, cursor_bin, anim_start_ms, anim_end_ms, scroll_frames, pause_frames)

    for sample_idx in selected:
        ci, ei = val_ds.samples[sample_idx]
        evt = val_ds.events[ci]
        if ei == 0:
            cursor_bin = max(0, int(evt[0]) - B_BINS)
        else:
            cursor_bin = int(evt[ei - 1])
        cursor_ms = cursor_bin * BIN_MS
        pred_bin = int(preds[sample_idx])
        pred_abs_ms = (cursor_bin + pred_bin) * BIN_MS

        anim_start_ms = cursor_ms - 2500
        anim_end_ms = pred_abs_ms + 300
        scroll_s = (anim_end_ms - anim_start_ms) / 1000 * 0.8
        pause_s = 1.0
        sf = int(scroll_s * args.fps)
        pf = int(pause_s * args.fps)
        case_info.append((sample_idx, cursor_bin, anim_start_ms, anim_end_ms, sf, pf))

    # synthesize audio per case, stretched/trimmed to match video duration
    all_audio = []
    for case_i, (sample_idx, cursor_bin, anim_start_ms, anim_end_ms, sf, pf) in enumerate(tqdm(case_info, desc="Audio")):
        ci, ei = val_ds.samples[sample_idx]
        chart = val_ds.charts[ci]

        # mel frames covering the animated time window
        start_bin = int(anim_start_ms / BIN_MS)
        end_bin = int(anim_end_ms / BIN_MS)

        mel_full = np.load(os.path.join(DS_DIR, "mels", chart["mel_file"]), mmap_mode="r")
        f0 = max(0, start_bin)
        f1 = min(mel_full.shape[1], end_bin)
        if f1 <= f0:
            # no mel data, fill with silence
            total_video_s = (sf + pf) / args.fps
            all_audio.append(np.zeros(int(total_video_s * SR), dtype=np.float32))
            continue

        mel_clip = mel_full[:, f0:f1].astype(np.float32)
        audio = mel_to_audio(mel_clip)

        # audio covers (f1-f0) * HOP / SR seconds of real time
        real_audio_s = len(audio) / SR
        # video scroll covers this in scroll_s seconds (0.8x speed)
        total_video_s = (sf + pf) / args.fps

        # stretch audio to match scroll duration, then add silence for pause
        scroll_video_s = sf / args.fps
        if real_audio_s > 0 and scroll_video_s > 0:
            # resample to match video scroll speed
            target_samples = int(scroll_video_s * SR)
            indices = np.linspace(0, len(audio) - 1, target_samples).astype(int)
            audio_stretched = audio[indices]
        else:
            audio_stretched = audio

        # add silence for pause
        pause_samples = int((pf / args.fps) * SR)
        audio_padded = np.concatenate([audio_stretched, np.zeros(pause_samples, dtype=np.float32)])
        all_audio.append(audio_padded)

    all_audio_cat = np.concatenate(all_audio)
    peak = np.abs(all_audio_cat).max()
    if peak > 0:
        all_audio_cat = all_audio_cat / peak * 0.8
    audio_pcm = (all_audio_cat * 32767).astype(np.int16)

    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp_audio.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio_pcm.tobytes())
    print(f"Audio: {len(all_audio_cat)/SR:.1f}s total")

    # ffmpeg with audio
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
        "-i", tmp_audio.name,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", args.output,
    ], stdin=subprocess.PIPE)

    for case_i, (sample_idx, cursor_bin, anim_start_ms, anim_end_ms, scroll_frames, pause_frames) in enumerate(tqdm(case_info, desc="Rendering")):
        ci, ei = val_ds.samples[sample_idx]
        chart = val_ds.charts[ci]
        evt = val_ds.events[ci]

        cursor_ms = cursor_bin * BIN_MS
        target_bin = int(targets[sample_idx])
        pred_bin = int(preds[sample_idx])
        score = scores[sample_idx]

        target_ms = (cursor_bin + target_bin) * BIN_MS
        pred_ms = (cursor_bin + pred_bin) * BIN_MS

        past_events_bins = evt[max(0, ei - C_EVENTS):ei].astype(int)
        past_events_ms = past_events_bins * BIN_MS

        mel = np.load(os.path.join(DS_DIR, "mels", chart["mel_file"]), mmap_mode="r")
        total_mel_frames = mel.shape[1]
        mel_min = float(mel.min())
        mel_max = float(mel.max())
        mel_range = max(mel_max - mel_min, 1.0)

        ratio = (pred_bin + 1) / (target_bin + 1) if target_bin > 0 else 0

        for frame_i in range(scroll_frames + pause_frames):
            surface.fill((22, 22, 30))

            # compute now_ms
            if frame_i < scroll_frames:
                t = frame_i / scroll_frames
                now_ms = anim_start_ms + t * (anim_end_ms - anim_start_ms)
            else:
                now_ms = anim_end_ms

            # header
            hdr = font_big.render(
                f"[{case_i+1}/{len(selected)}] score={score:.3f}  "
                f"pred={pred_bin} ({pred_bin*BIN_MS:.0f}ms)  "
                f"target={target_bin} ({target_bin*BIN_MS:.0f}ms)  "
                f"ratio={ratio:.2f}x",
                True, (200, 200, 210))
            surface.blit(hdr, (10, 5))

            info = font_small.render(
                f"{chart.get('artist', '?')} - {chart.get('title', '?')} "
                f"[{chart.get('difficulty', '?')}]  cursor@{cursor_ms:.0f}ms",
                True, (120, 120, 135))
            surface.blit(info, (10, 26))

            # playfield bg
            pygame.draw.rect(surface, (30, 30, 42), (0, PF_TOP, W, PF_H))

            # cursor line
            pygame.draw.line(surface, (255, 255, 255), (HIT_X, PF_TOP), (HIT_X, PF_TOP + PF_H), 2)
            cursor_lbl = font_small.render("CURSOR", True, (180, 180, 180))
            surface.blit(cursor_lbl, (HIT_X - 20, PF_TOP - 12))

            # past events (blue circles, scroll from right to left)
            for e_ms in past_events_ms:
                x = HIT_X + (e_ms - now_ms) * SCROLL_PX_PER_MS
                if -20 < x < W + 20:
                    # dim if past cursor
                    if x < HIT_X:
                        brightness = max(0.2, 1.0 - (HIT_X - x) / 300)
                        color = (int(68 * brightness), int(141 * brightness), int(199 * brightness))
                    else:
                        color = (68, 141, 199)
                    pygame.draw.circle(surface, color, (int(x), PF_CY), 10)
                    pygame.draw.circle(surface, (255, 255, 255), (int(x), PF_CY), 10, 1)

            # target (green ghost) — always visible in future
            tx = HIT_X + (target_ms - now_ms) * SCROLL_PX_PER_MS
            if -20 < tx < W + 20:
                ghost = pygame.Surface((28, 28), pygame.SRCALPHA)
                pygame.draw.circle(ghost, (0, 220, 0, 160), (14, 14), 14)
                pygame.draw.circle(ghost, (255, 255, 255, 200), (14, 14), 14, 2)
                surface.blit(ghost, (int(tx) - 14, PF_CY - 14))
                lbl = font_small.render("TARGET", True, (0, 200, 0))
                surface.blit(lbl, (int(tx) - 18, PF_CY - 26))

            # prediction (red) — always visible in future
            px = HIT_X + (pred_ms - now_ms) * SCROLL_PX_PER_MS
            if -20 < px < W + 20:
                pygame.draw.circle(surface, (235, 69, 44), (int(px), PF_CY), 12)
                pygame.draw.circle(surface, (255, 255, 255), (int(px), PF_CY), 12, 2)
                lbl = font_small.render("PRED", True, (235, 100, 80))
                surface.blit(lbl, (int(px) - 12, PF_CY + 16))

            # mel spectrogram (scrolling with now_ms)
            pygame.draw.rect(surface, (20, 20, 28), (0, MEL_TOP, W, MEL_H))

            # which mel frames are visible?
            left_ms = now_ms - HIT_X / SCROLL_PX_PER_MS
            right_ms = now_ms + (W - HIT_X) / SCROLL_PX_PER_MS
            left_frame = int(left_ms / BIN_MS)
            right_frame = int(right_ms / BIN_MS)

            f0 = max(0, left_frame)
            f1 = min(total_mel_frames, right_frame)
            if f1 > f0:
                mel_slice = mel[:, f0:f1].astype(np.float32)
                mel_norm = np.clip((mel_slice - mel_min) / mel_range * 255, 0, 255).astype(np.uint8)
                mel_norm = mel_norm[::-1, :]  # flip vertically
                mel_rgb = cmap[mel_norm].transpose(1, 0, 2)  # (n_vis, 80, 3)

                mel_surf = pygame.surfarray.make_surface(mel_rgb)

                # position on screen
                screen_x0 = int((f0 * BIN_MS - left_ms) * SCROLL_PX_PER_MS)
                screen_x1 = int((f1 * BIN_MS - left_ms) * SCROLL_PX_PER_MS)
                pixel_w = max(1, screen_x1 - screen_x0)

                scaled = pygame.transform.scale(mel_surf, (pixel_w, MEL_H))
                surface.blit(scaled, (screen_x0, MEL_TOP))

            # cursor line on mel
            pygame.draw.line(surface, (255, 255, 255),
                             (HIT_X, MEL_TOP), (HIT_X, MEL_TOP + MEL_H), 1)

            # event markers on mel bottom
            for e_ms in past_events_ms:
                ex = HIT_X + (e_ms - now_ms) * SCROLL_PX_PER_MS
                if 0 <= ex <= W:
                    pygame.draw.line(surface, (68, 141, 199),
                                     (int(ex), MEL_TOP + MEL_H - 4), (int(ex), MEL_TOP + MEL_H), 2)

            # target/pred lines on mel
            if -20 < tx < W + 20:
                pygame.draw.line(surface, (0, 220, 0),
                                 (int(tx), MEL_TOP), (int(tx), MEL_TOP + MEL_H), 2)
            if -20 < px < W + 20:
                pygame.draw.line(surface, (235, 69, 44),
                                 (int(px), MEL_TOP), (int(px), MEL_TOP + MEL_H), 2)

            # progress bar
            bar_y = H - 15
            pygame.draw.rect(surface, (40, 40, 55), (10, bar_y, W - 20, 6))
            prog = frame_i / (scroll_frames + pause_frames)
            pygame.draw.rect(surface, (80, 120, 220), (10, bar_y, int((W - 20) * prog), 6))

            # write frame
            try:
                proc.stdin.write(pygame.image.tobytes(surface, "RGB"))
            except BrokenPipeError:
                break

    proc.stdin.close()
    proc.wait()
    pygame.quit()

    # cleanup temp audio
    try:
        os.unlink(tmp_audio.name)
    except OSError:
        pass

    print(f"\nSaved: {args.output} ({len(selected)} cases)")


if __name__ == "__main__":
    main()
