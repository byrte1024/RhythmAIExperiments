"""osu!taiko onset visualizer with full controls, stats, and inference analysis.

Controls:
  Space     - Pause / Resume
  Left/Right- Seek -5s / +5s  (hold Shift for 1s)
  Up/Down   - Volume up/down
  +/-       - Speed up/down (0.25x steps)
  R         - Restart
  E         - Export to video (.mp4)
  H         - Toggle help overlay
  T         - Toggle stats panel
  M         - Toggle minimap
  D         - Toggle density graph
  W         - Toggle mel spectrogram + waveform
  Esc/Q     - Quit
"""
import pygame
import glob
import os
import sys
import array
import struct
import math
import random
import json
import argparse
import io
import wave
import subprocess
import tempfile

AUDIO_DIR = "./osu/taiko/audio"
DATA_DIR = "./osu/taiko/data"

WIDTH, HEIGHT = 1200, 700
PLAYFIELD_TOP = 60
PLAYFIELD_H = 160
PLAYFIELD_CENTER = PLAYFIELD_TOP + PLAYFIELD_H // 2
HIT_X = 160
SCROLL_SPEED = 0.5  # pixels per ms
FPS = 120

# Colors
BG_COLOR = (22, 22, 30)
PLAYFIELD_BG = (30, 30, 42)
HIT_LINE_COLOR = (255, 255, 255)
TEXT_COLOR = (200, 200, 210)
DIM_TEXT = (120, 120, 135)
ACCENT = (100, 140, 255)
PANEL_BG = (28, 28, 38)
PANEL_BORDER = (50, 50, 65)
PROGRESS_BG = (40, 40, 55)
PROGRESS_FILL = (80, 120, 220)
PROGRESS_CURSOR = (255, 255, 255)

COLORS = {
    "don": (235, 69, 44),
    "ka": (68, 141, 199),
    "big_don": (255, 90, 60),
    "big_ka": (80, 165, 230),
    "drumroll": (252, 183, 30),
    "spinner": (100, 200, 100),
    "predicted": (180, 120, 255),
}
SIZES = {
    "don": 18, "ka": 18,
    "big_don": 28, "big_ka": 28,
    "drumroll": 14, "spinner": 22,
    "predicted": 18,
}

DENSITY_WINDOW_MS = 1000  # 1-second window for density calc


def parse_args():
    parser = argparse.ArgumentParser(description="osu!taiko onset visualizer")
    parser.add_argument("csv", nargs="?", default=None, help="CSV file to view")
    parser.add_argument("--audio", default=None, help="Override audio file path")
    parser.add_argument("--compare", default=None, help="Second CSV to overlay (e.g. ground truth)")
    parser.add_argument("--stats-json", default=None, help="Inference stats JSON file")
    parser.add_argument("--mel-npy", default=None, help="Mel spectrogram .npy file")
    parser.add_argument("--wave-npy", default=None, help="Waveform envelope .npy file")
    parser.add_argument("--sampling-npy", default=None, help="Sampling timeline .npy file (temperature + metronome)")
    parser.add_argument("--candidates-json", default=None, help="Candidate history JSON (per-prediction candidates)")
    parser.add_argument("--gif", default=None, help="Path to GIF file for beat-synced animation window")
    parser.add_argument("--gif-cycles", type=int, default=1, help="Events per full GIF animation cycle (default: 1)")
    parser.add_argument("--render", default=None, help="Render to video file (e.g. output.mp4) instead of interactive mode")
    parser.add_argument("--render-fps", type=int, default=60, help="Video FPS for render mode (default 60)")
    return parser.parse_args()


def pick_csv():
    csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csvs:
        return None
    pick = random.choice(csvs)
    print(f"Random pick: {os.path.basename(pick)}")
    return pick


def load_csv(path):
    audio_file = None
    onsets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# audio:"):
                audio_file = line.split(":", 1)[1].strip()
                continue
            if line.startswith("time_ms"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                onsets.append((int(parts[0]), parts[1].strip()))
    return audio_file, onsets


def make_tick_sound(duration_ms=40, volume=0.7):
    """Generate a short static/noise tick sound."""
    mixer_freq, mixer_size, mixer_channels = pygame.mixer.get_init()
    n_samples = int(mixer_freq * duration_ms / 1000)
    buf = array.array("h")
    for i in range(n_samples):
        fade = 1.0 - (i / n_samples) ** 0.5  # fast decay
        val = int(volume * 32767 * fade * (random.random() * 2 - 1))
        for _ in range(mixer_channels):
            buf.append(val)
    return pygame.mixer.Sound(buffer=buf)


def find_audio(name):
    if not name:
        return None
    path = os.path.join(AUDIO_DIR, name)
    if os.path.exists(path):
        return path
    if os.path.exists(name):
        return name
    return None


def _has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


HAS_FFMPEG = None  # lazy-checked


def load_audio_at_speed(audio_path, speed):
    """Load audio file at a given speed using ffmpeg atempo filter.
    Returns a temp file path to the resampled WAV, or None on failure."""
    global HAS_FFMPEG
    if HAS_FFMPEG is None:
        HAS_FFMPEG = _has_ffmpeg()
    if not HAS_FFMPEG or not audio_path:
        return None

    # atempo filter only supports 0.5-100.0; chain for slower
    tempo = speed
    atempo_chain = []
    while tempo < 0.5:
        atempo_chain.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        atempo_chain.append("atempo=2.0")
        tempo /= 2.0
    atempo_chain.append(f"atempo={tempo:.4f}")
    af = ",".join(atempo_chain)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-af", af,
             "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", tmp.name],
            capture_output=True, timeout=30,
        )
        if os.path.getsize(tmp.name) > 100:
            return tmp.name
    except Exception:
        pass
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return None


def compute_level_stats(onsets):
    """Compute comprehensive statistics about the onset data."""
    if not onsets:
        return {}

    times = [t for t, _ in onsets]
    types = [k for _, k in onsets]

    duration_ms = max(times) - min(times) if len(times) > 1 else 0
    duration_s = duration_ms / 1000

    # Type counts
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    # Inter-onset intervals
    iois = [times[i+1] - times[i] for i in range(len(times)-1)] if len(times) > 1 else []
    avg_ioi = sum(iois) / len(iois) if iois else 0
    min_ioi = min(iois) if iois else 0
    max_ioi = max(iois) if iois else 0
    median_ioi = sorted(iois)[len(iois)//2] if iois else 0

    # Density over time (events per second, computed in 1s windows)
    density_timeline = []
    if duration_s > 0:
        window_s = 1.0
        for t_start in range(int(min(times)), int(max(times)), int(window_s * 1000)):
            t_end = t_start + int(window_s * 1000)
            count = sum(1 for t in times if t_start <= t < t_end)
            density_timeline.append((t_start, count / window_s))

    densities = [d for _, d in density_timeline]
    mean_density = sum(densities) / len(densities) if densities else 0
    peak_density = max(densities) if densities else 0
    density_std = (sum((d - mean_density)**2 for d in densities) / len(densities))**0.5 if densities else 0

    # BPM estimation from common IOI
    bpm_estimate = 0
    if iois:
        # Round IOIs to nearest 5ms and find most common
        rounded = [round(i / 5) * 5 for i in iois if 100 < i < 1500]
        if rounded:
            from collections import Counter
            common_ioi = Counter(rounded).most_common(1)[0][0]
            if common_ioi > 0:
                bpm_estimate = 60000 / common_ioi

    # IOI distribution buckets for histogram
    ioi_buckets = {}
    for ioi in iois:
        bucket = round(ioi / 10) * 10  # 10ms buckets
        ioi_buckets[bucket] = ioi_buckets.get(bucket, 0) + 1

    is_predicted = "predicted" in type_counts

    stats = {
        "total_events": len(onsets),
        "duration_ms": duration_ms,
        "duration_s": duration_s,
        "type_counts": type_counts,
        "mean_density": mean_density,
        "peak_density": peak_density,
        "density_std": density_std,
        "density_timeline": density_timeline,
        "avg_ioi_ms": avg_ioi,
        "min_ioi_ms": min_ioi,
        "max_ioi_ms": max_ioi,
        "median_ioi_ms": median_ioi,
        "bpm_estimate": bpm_estimate,
        "ioi_buckets": ioi_buckets,
        "is_predicted": is_predicted,
        "first_event_ms": min(times) if times else 0,
        "last_event_ms": max(times) if times else 0,
    }
    return stats


def format_time(ms):
    """Format milliseconds as M:SS.s"""
    s = ms / 1000
    m = int(s // 60)
    s = s % 60
    return f"{m}:{s:05.2f}"


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Must match training constants for mel frame timing
MEL_HOP_LENGTH = 110
MEL_SAMPLE_RATE = 22050
MEL_BIN_MS = MEL_HOP_LENGTH / MEL_SAMPLE_RATE * 1000  # ~4.9887ms

# Mel colormap: black -> blue -> cyan -> yellow -> white
_MEL_CMAP = None

def _get_mel_colormap():
    """Build a 256-entry colormap for mel spectrogram rendering."""
    global _MEL_CMAP
    if _MEL_CMAP is not None:
        return _MEL_CMAP
    cmap = []
    # 5 control points: black, dark blue, cyan, yellow, white
    stops = [
        (0,   (0, 0, 0)),
        (64,  (10, 10, 80)),
        (128, (20, 100, 160)),
        (192, (220, 200, 50)),
        (255, (255, 255, 255)),
    ]
    for i in range(256):
        # Find bounding stops
        lo, hi = stops[0], stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= i <= stops[j+1][0]:
                lo, hi = stops[j], stops[j+1]
                break
        span = hi[0] - lo[0]
        t = (i - lo[0]) / span if span > 0 else 0
        r = int(lo[1][0] + t * (hi[1][0] - lo[1][0]))
        g = int(lo[1][1] + t * (hi[1][1] - lo[1][1]))
        b = int(lo[1][2] + t * (hi[1][2] - lo[1][2]))
        cmap.append((r, g, b))
    _MEL_CMAP = cmap
    return cmap


class GifPlayer:
    """Beat-synced GIF animation rendered as overlay on the main viewer."""

    def __init__(self, gif_path, cycles=1):
        from PIL import Image

        self.cycles = max(1, cycles)
        self.frames = []

        img = Image.open(gif_path)
        try:
            while True:
                frame = img.convert("RGBA")
                surf = pygame.image.fromstring(frame.tobytes(), frame.size, "RGBA")
                self.frames.append(surf)
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if not self.frames:
            raise ValueError(f"No frames found in {gif_path}")

        self.n_frames = len(self.frames)
        self.current_frame = 0
        # pre-scale to a reasonable display size
        fw, fh = self.frames[0].get_size()
        self.display_h = 440
        self.display_w = int(fw * self.display_h / fh)
        self.scaled_frames = [
            pygame.transform.smoothscale(f, (self.display_w, self.display_h))
            for f in self.frames
        ]
        print(f"Loaded GIF: {self.n_frames} frames, display {self.display_w}x{self.display_h}, {cycles} events/cycle")

    def update(self, now_ms, onsets):
        """Update frame based on progress between events."""
        passed = 0
        prev_ms = 0
        next_ms = 0
        for t, _ in onsets:
            if t <= now_ms:
                passed += 1
                prev_ms = t
            else:
                next_ms = t
                break

        cycle_event = passed % self.cycles
        if next_ms > prev_ms:
            inter = (now_ms - prev_ms) / (next_ms - prev_ms)
        else:
            inter = 0.0
        inter = max(0.0, min(1.0, inter))

        progress = (cycle_event + inter) / self.cycles
        self.current_frame = int(progress * self.n_frames) % self.n_frames

    def draw(self, screen, x, y):
        """Render current frame onto the given surface."""
        screen.blit(self.scaled_frames[self.current_frame], (x, y))


class Viewer:
    def __init__(self, csv_path, audio_override=None, compare_csv=None,
                 stats_json_path=None, mel_npy_path=None, wave_npy_path=None,
                 sampling_npy_path=None, candidates_json_path=None,
                 gif_path=None, gif_cycles=1):
        self.csv_path = csv_path
        audio_name, self.onsets = load_csv(csv_path)
        self.stats = compute_level_stats(self.onsets)
        self.song_end_ms = self.stats.get("last_event_ms", 0) + 3000

        # Comparison data (e.g. ground truth vs predicted)
        self.compare_onsets = None
        self.compare_stats = None
        if compare_csv:
            _, self.compare_onsets = load_csv(compare_csv)
            self.compare_stats = compute_level_stats(self.compare_onsets)

        # Inference stats from JSON
        self.inference_stats = None
        if stats_json_path and os.path.exists(stats_json_path):
            with open(stats_json_path, "r") as f:
                self.inference_stats = json.load(f)

        # Mel spectrogram + waveform data (from inference)
        self.mel_data = None  # shape (n_mels, T) numpy array
        self.wave_data = None  # shape (T,) numpy array
        if HAS_NUMPY:
            if mel_npy_path and os.path.exists(mel_npy_path):
                self.mel_data = np.load(mel_npy_path)
                print(f"Loaded mel spectrogram: {self.mel_data.shape}")
            if wave_npy_path and os.path.exists(wave_npy_path):
                self.wave_data = np.load(wave_npy_path)
                print(f"Loaded waveform envelope: {self.wave_data.shape}")

        # Sampling timeline: (N, 3) = [cursor_bin, temperature, closeness]
        self.sampling_data = None
        if HAS_NUMPY:
            if sampling_npy_path and os.path.exists(sampling_npy_path):
                self.sampling_data = np.load(sampling_npy_path)
                print(f"Loaded sampling timeline: {self.sampling_data.shape}")

        # Candidate history: list of [cursor_bin, chosen_bin, [[bin, raw, final], ...]]
        self.candidate_data = None
        self._cand_by_cursor = {}  # cursor_bin → (chosen, abs_candidates) for quick lookup
        if candidates_json_path and os.path.exists(candidates_json_path):
            with open(candidates_json_path, "r", encoding="utf-8") as f:
                self.candidate_data = json.load(f)
            for entry in self.candidate_data:
                cursor_bin, chosen, cands = entry
                abs_cands = [(cursor_bin + c[0], c[1], c[2]) for c in cands]
                self._cand_by_cursor[cursor_bin] = (chosen, abs_cands)
            print(f"Loaded candidate history: {len(self.candidate_data)} predictions, {len(self._cand_by_cursor)} cursor points")

        # Beat-synced GIF
        self.gif_player = None
        if gif_path:
            try:
                self.gif_player = GifPlayer(gif_path, cycles=gif_cycles)
            except Exception as e:
                print(f"Could not load GIF: {e}")

        # Audio
        self.audio_path = audio_override or find_audio(audio_name)
        self.has_audio = False
        self._speed_tmp_file = None  # temp file for speed-adjusted audio

        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        if self.audio_path and os.path.exists(self.audio_path):
            try:
                pygame.mixer.music.load(self.audio_path)
                self.has_audio = True
                print(f"Audio loaded: {self.audio_path}")
            except Exception as e:
                print(f"Could not load audio: {e}")
        else:
            print(f"Audio not found: {self.audio_path or audio_name}")

        self.w = WIDTH
        self.h = HEIGHT
        self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
        pygame.display.set_caption(f"Taiko Viewer - {os.path.basename(csv_path)}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 13)
        self.font_big = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 11)
        # Title font needs CJK support for Japanese song names
        self.font_title = pygame.font.SysFont("meiryoui,meiryo,yugothicui,yugothic,msgothic,msmincho,consolas", 16, bold=True)

        # Tick sounds
        self.tick_don = make_tick_sound(duration_ms=45, volume=0.95)
        self.tick_ka = make_tick_sound(duration_ms=30, volume=0.85)

        # State
        self.playing = False
        self.paused_at = 0
        self.play_start_ticks = 0
        self.now_ms = 0
        self.speed = 1.0
        self.volume = 0.35
        self.next_hit = 0
        self.show_help = False
        self.show_stats = True
        self.show_minimap = True
        self.show_density = True
        self.show_mel = bool(self.mel_data is not None or self.wave_data is not None)
        self.zoom = 1.0  # zoom multiplier for scroll speed
        # Precompute global normalization ranges for consistent rendering
        if HAS_NUMPY and self.mel_data is not None:
            self._mel_global_min = float(self.mel_data.min())
            self._mel_global_max = float(self.mel_data.max())
        else:
            self._mel_global_min = 0.0
            self._mel_global_max = 1.0
        if HAS_NUMPY and self.wave_data is not None:
            self._wave_global_max = float(self.wave_data.max()) if self.wave_data.max() > 0 else 1.0
        else:
            self._wave_global_max = 1.0
        self.scroll_offset = 0  # for stats panel scrolling
        self.recent_hits = []  # (time, kind) for hit flash animation

        # Precompute density graph surface
        self._density_surface = None
        self._compare_density_surface = None
        self._precompute_density_surface()

        # Start playback
        self._start_playback(0)

    def _precompute_density_surface(self):
        """Precompute the density graph as a surface."""
        timeline = self.stats.get("density_timeline", [])
        if not timeline:
            return
        max_d = max(d for _, d in timeline) if timeline else 1
        w = max(len(timeline), 1)
        h = 50
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for i, (t, d) in enumerate(timeline):
            bar_h = int((d / max(max_d, 1)) * (h - 2))
            color = (*ACCENT, 160)
            pygame.draw.line(surf, color, (i, h), (i, h - bar_h), 1)
        self._density_surface = surf

        if self.compare_stats:
            ct = self.compare_stats.get("density_timeline", [])
            if ct:
                cmax = max(d for _, d in ct) if ct else 1
                cw = max(len(ct), 1)
                csurf = pygame.Surface((cw, h), pygame.SRCALPHA)
                for i, (t, d) in enumerate(ct):
                    bar_h = int((d / max(cmax, 1)) * (h - 2))
                    pygame.draw.line(csurf, (255, 180, 80, 120), (i, h), (i, h - bar_h), 1)
                self._compare_density_surface = csurf

    def _start_playback(self, from_ms):
        """Start or resume playback from a given position."""
        self.now_ms = from_ms
        self.paused_at = from_ms
        self.playing = True
        self.play_start_ticks = pygame.time.get_ticks() - int(from_ms / self.speed)

        # Reset hit trigger index
        self.next_hit = 0
        for i, (t, _) in enumerate(self.onsets):
            if t > from_ms:
                self.next_hit = i
                break
        else:
            self.next_hit = len(self.onsets)

        if self.has_audio:
            # The audio file is resampled by ffmpeg atempo so it plays at 1x
            # but represents the song at self.speed. Seek position in the
            # resampled file = from_ms / speed (since the file is shorter/longer).
            audio_seek_s = from_ms / self.speed / 1000
            pygame.mixer.music.play(start=audio_seek_s)
            pygame.mixer.music.set_volume(self.volume)

    def _pause(self):
        self.playing = False
        self.paused_at = self.now_ms
        if self.has_audio:
            pygame.mixer.music.pause()

    def _resume(self):
        self.playing = True
        self.play_start_ticks = pygame.time.get_ticks() - int(self.paused_at / self.speed)
        if self.has_audio:
            pygame.mixer.music.unpause()

    def _seek(self, delta_ms):
        target = max(0, self.now_ms + delta_ms)
        was_playing = self.playing
        if self.has_audio:
            pygame.mixer.music.stop()
        self._start_playback(target)
        if not was_playing:
            self._pause()

    def _set_speed(self, new_speed):
        new_speed = max(0.25, min(4.0, new_speed))
        if abs(new_speed - self.speed) < 0.01:
            return
        current = self.now_ms
        was_playing = self.playing
        self.speed = new_speed
        self._reload_audio_at_speed()
        self._start_playback(current)
        if not was_playing:
            self._pause()

    def _reload_audio_at_speed(self):
        """Reload audio file at current speed using ffmpeg."""
        # Clean up previous temp file
        if self._speed_tmp_file:
            try:
                os.unlink(self._speed_tmp_file)
            except OSError:
                pass
            self._speed_tmp_file = None

        if not self.has_audio or not self.audio_path:
            return

        if abs(self.speed - 1.0) < 0.01:
            # Back to normal speed, use original file
            try:
                pygame.mixer.music.load(self.audio_path)
            except Exception:
                pass
            return

        tmp_path = load_audio_at_speed(self.audio_path, self.speed)
        if tmp_path:
            try:
                pygame.mixer.music.load(tmp_path)
                self._speed_tmp_file = tmp_path
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                shift = mods & pygame.KMOD_SHIFT

                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif ev.key == pygame.K_SPACE:
                    if self.playing:
                        self._pause()
                    else:
                        self._resume()
                elif ev.key == pygame.K_LEFT:
                    self._seek(-1000 if shift else -5000)
                elif ev.key == pygame.K_RIGHT:
                    self._seek(1000 if shift else 5000)
                elif ev.key == pygame.K_UP:
                    self.volume = min(1.0, self.volume + 0.05)
                    if self.has_audio:
                        pygame.mixer.music.set_volume(self.volume)
                elif ev.key == pygame.K_DOWN:
                    self.volume = max(0.0, self.volume - 0.05)
                    if self.has_audio:
                        pygame.mixer.music.set_volume(self.volume)
                elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self._set_speed(self.speed + 0.25)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._set_speed(self.speed - 0.25)
                elif ev.key == pygame.K_r:
                    self._start_playback(0)
                elif ev.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif ev.key == pygame.K_t:
                    self.show_stats = not self.show_stats
                elif ev.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                elif ev.key == pygame.K_d:
                    self.show_density = not self.show_density
                elif ev.key == pygame.K_w:
                    if self.mel_data is not None or self.wave_data is not None:
                        self.show_mel = not self.show_mel
                elif ev.key == pygame.K_e:
                    self._export_video()

            if ev.type == pygame.MOUSEBUTTONDOWN:
                # Click on progress bar to seek
                mx, my = ev.pos
                prog_y = PLAYFIELD_TOP + PLAYFIELD_H + 5
                if prog_y <= my <= prog_y + 20 and self.song_end_ms > 0:
                    frac = max(0, min(1, (mx - 10) / (self.w - 20)))
                    self._seek(frac * self.song_end_ms - self.now_ms)

            if ev.type == pygame.MOUSEWHEEL:
                # Scroll wheel zooms in/out (note highway + mel/wave)
                factor = 1.15 if ev.y > 0 else 1 / 1.15
                self.zoom = max(0.1, min(10.0, self.zoom * factor))

            if ev.type == pygame.VIDEORESIZE:
                self.w, self.h = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)

        return True

    def _export_video(self):
        """Export current view to video file, with file picker for location."""
        import shutil
        import tkinter as tk
        from tkinter import filedialog

        was_playing = self.playing
        if was_playing:
            self._pause()

        # default filename from csv name
        base = os.path.splitext(os.path.basename(self.csv_path))[0]

        # file picker (tkinter dialog, hidden root window)
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        output_path = filedialog.asksaveasfilename(
            title="Export video",
            initialfile=f"{base}.mp4",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        root.destroy()

        if not output_path:
            print("Export cancelled.")
            if was_playing:
                self._resume()
            return

        # show export message on screen
        font = pygame.font.SysFont("consolas", 24)
        msg = font.render(f"Exporting to {os.path.basename(output_path)}...", True, (255, 255, 100))
        rect = msg.get_rect(center=(self.w // 2, self.h // 2))
        self.screen.blit(msg, rect)
        pygame.display.flip()

        self.render_video(output_path)

        # copy CSV alongside the video
        csv_dest = os.path.splitext(output_path)[0] + ".csv"
        try:
            shutil.copy2(self.csv_path, csv_dest)
            print(f"CSV copied to: {csv_dest}")
        except Exception as e:
            print(f"Could not copy CSV: {e}")

        print(f"Export complete: {output_path}")

        if was_playing:
            self._resume()

    def update(self):
        if self.playing:
            self.now_ms = (pygame.time.get_ticks() - self.play_start_ticks) * self.speed

        # Trigger tick sounds
        while self.next_hit < len(self.onsets) and self.onsets[self.next_hit][0] <= self.now_ms:
            kind = self.onsets[self.next_hit][1]
            if "ka" in kind:
                self.tick_ka.play()
            else:
                self.tick_don.play()
            self.recent_hits.append((self.now_ms, kind))
            self.next_hit += 1

        # Expire old hit flashes (keep 300ms)
        self.recent_hits = [(t, k) for t, k in self.recent_hits if self.now_ms - t < 300]

    def draw(self):
        self.screen.fill(BG_COLOR)

        self._draw_header()
        self._draw_playfield()
        self._draw_progress_bar()

        y_below = PLAYFIELD_TOP + PLAYFIELD_H + 30

        if self.show_mel:
            self._draw_mel_view(y_below)
            y_below += 145
            if self.sampling_data is not None:
                y_below += 48

        if self.show_density:
            self._draw_density_graph(y_below)
            y_below += 65

        if self.show_minimap:
            self._draw_minimap(y_below)
            y_below += 35

        if self.show_stats:
            self._draw_stats_panel(y_below)

        if self.show_help:
            self._draw_help_overlay()

        # beat-synced GIF (bottom-right, next to inference stats)
        if self.gif_player:
            self.gif_player.update(self.now_ms, self.onsets)
            gx = self.w - self.gif_player.display_w - 20
            gy = self.h - self.gif_player.display_h - 15
            self.gif_player.draw(self.screen, gx, gy)

        pygame.display.flip()

    def _draw_header(self):
        """Top bar: title, time, speed, volume."""
        # Title
        name = os.path.basename(self.csv_path)
        if len(name) > 60:
            name = name[:57] + "..."
        title = self.font_title.render(name, True, TEXT_COLOR)
        self.screen.blit(title, (10, 8))

        # Time
        total = self.song_end_ms
        time_str = f"{format_time(self.now_ms)} / {format_time(total)}"
        t = self.font_big.render(time_str, True, ACCENT)
        self.screen.blit(t, (10, 32))

        # Status badges
        x = self.w - 10
        badges = []
        if not self.playing:
            badges.append(("PAUSED", (255, 200, 60)))
        badges.append((f"{self.speed:.2f}x", TEXT_COLOR))
        if abs(self.zoom - 1.0) > 0.01:
            badges.append((f"ZOOM {self.zoom:.1f}x", ACCENT))
        badges.append((f"VOL {int(self.volume * 100)}%", TEXT_COLOR))
        if not self.has_audio:
            badges.append(("NO AUDIO", (255, 80, 80)))
        if self.stats.get("is_predicted"):
            badges.append(("INFERENCE", (180, 120, 255)))

        for text, color in reversed(badges):
            surf = self.font.render(text, True, color)
            w = surf.get_width() + 12
            x -= w + 4
            pygame.draw.rect(self.screen, PANEL_BG, (x, 8, w, 22), border_radius=4)
            pygame.draw.rect(self.screen, PANEL_BORDER, (x, 8, w, 22), 1, border_radius=4)
            self.screen.blit(surf, (x + 6, 11))

        # Help hint
        hint = self.font_small.render("H=Help  T=Stats  M=Map  D=Density  W=Mel/Wave", True, DIM_TEXT)
        self.screen.blit(hint, (self.w - hint.get_width() - 10, 36))

    def _draw_playfield(self):
        """Main note highway."""
        # Background
        pygame.draw.rect(self.screen, PLAYFIELD_BG,
                         (0, PLAYFIELD_TOP, self.w, PLAYFIELD_H), border_radius=4)

        cy = PLAYFIELD_CENTER

        # Hit flash
        if self.recent_hits:
            alpha = max(0, min(255, 200 - int((self.now_ms - self.recent_hits[-1][0]) * 0.7)))
            if alpha > 0:
                flash_surf = pygame.Surface((40, PLAYFIELD_H))
                flash_surf.fill((255, 255, 255))
                flash_surf.set_alpha(alpha)
                self.screen.blit(flash_surf, (HIT_X - 20, PLAYFIELD_TOP))

        # Hit line
        pygame.draw.line(self.screen, HIT_LINE_COLOR, (HIT_X, PLAYFIELD_TOP + 10),
                         (HIT_X, PLAYFIELD_TOP + PLAYFIELD_H - 10), 3)

        # Drum circle at hit line
        pygame.draw.circle(self.screen, (60, 60, 80), (HIT_X, cy), 30)
        pygame.draw.circle(self.screen, HIT_LINE_COLOR, (HIT_X, cy), 30, 2)

        # Draw comparison notes first (behind)
        if self.compare_onsets:
            self._draw_notes(self.compare_onsets, cy - 20, alpha_mod=0.4, outline_only=True)

        # Draw main notes
        self._draw_notes(self.onsets, cy)

        # Draw ghost notes above main notes (alternative candidates)
        if self._cand_by_cursor:
            self._draw_ghost_candidates(PLAYFIELD_TOP + 20)

        # Note count near hit line
        passed = self.next_hit
        total = len(self.onsets)
        counter = self.font_small.render(f"{passed}/{total}", True, DIM_TEXT)
        self.screen.blit(counter, (HIT_X - counter.get_width() // 2, PLAYFIELD_TOP + PLAYFIELD_H - 18))

    def _find_next_prediction(self):
        """Find the next prediction cursor point after playback cursor."""
        cursor_bin = int(self.now_ms / MEL_BIN_MS)
        best_cursor = None
        best_dist = float('inf')
        for pred_cursor in self._cand_by_cursor:
            dist = pred_cursor - cursor_bin
            if 0 <= dist < best_dist:
                best_dist = dist
                best_cursor = pred_cursor
        if best_cursor is None:
            return None, None, None
        chosen, cands = self._cand_by_cursor[best_cursor]
        return best_cursor, chosen, cands

    def _draw_ghost_candidates(self, cy):
        """Draw ghost notes for the next prediction's alternative candidates only."""
        pred_cursor, chosen, cands = self._find_next_prediction()
        if cands is None or not cands:
            return

        total_conf = sum(c[2] for c in cands) or 1.0
        # find chosen event's x position
        chosen_bin = pred_cursor + chosen if chosen < 500 else pred_cursor
        chosen_ms = chosen_bin * MEL_BIN_MS
        chosen_x = HIT_X + (chosen_ms - self.now_ms) * SCROLL_SPEED * self.zoom

        for cand_bin, raw_conf, final_conf in cands:
            cand_ms = cand_bin * MEL_BIN_MS
            cand_x = HIT_X + (cand_ms - self.now_ms) * SCROLL_SPEED * self.zoom
            if abs(cand_x - chosen_x) < 3:  # skip the chosen one
                continue
            if cand_x < -40 or cand_x > self.w + 40:
                continue

            # size based on confidence, constant opacity
            pct = final_conf / total_conf
            size = max(4, int(12 * pct * len(cands)))

            ghost_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(ghost_surf, (220, 170, 255, 200), (size, size), size)
            pygame.draw.circle(ghost_surf, (255, 220, 255, 220), (size, size), size, 1)
            self.screen.blit(ghost_surf, (int(cand_x) - size, cy - size))

    def _draw_notes(self, onsets, cy, alpha_mod=1.0, outline_only=False):
        """Draw note circles on the playfield."""
        for time_ms, kind in onsets:
            x = HIT_X + (time_ms - self.now_ms) * SCROLL_SPEED * self.zoom
            if x < -40:
                continue
            if x > self.w + 40:
                break

            color = COLORS.get(kind, (200, 200, 200))
            size = SIZES.get(kind, 16)

            # Dim notes that passed
            if x < HIT_X:
                factor = max(0.15, 1.0 - (HIT_X - x) / 200)
                color = tuple(int(c * factor) for c in color)

            if alpha_mod < 1.0:
                color = tuple(int(c * alpha_mod) for c in color)

            if outline_only:
                pygame.draw.circle(self.screen, color, (int(x), cy), size, 2)
            else:
                pygame.draw.circle(self.screen, color, (int(x), cy), size)
                # White border
                border_color = (255, 255, 255) if x >= HIT_X else tuple(int(180 * (max(0.15, 1.0 - (HIT_X - x) / 200))) for _ in range(3))
                pygame.draw.circle(self.screen, border_color, (int(x), cy), size, 2)

                # Inner circle for don/ka distinction
                if kind in ("don", "big_don"):
                    inner_color = tuple(min(255, c + 40) for c in COLORS.get(kind, color))
                    pygame.draw.circle(self.screen, inner_color, (int(x), cy), max(4, size // 3))

    def _draw_progress_bar(self):
        """Clickable progress bar below playfield."""
        y = PLAYFIELD_TOP + PLAYFIELD_H + 5
        bar_x, bar_w, bar_h = 10, self.w - 20, 16

        pygame.draw.rect(self.screen, PROGRESS_BG, (bar_x, y, bar_w, bar_h), border_radius=3)

        if self.song_end_ms > 0:
            frac = max(0, min(1, self.now_ms / self.song_end_ms))
            fill_w = int(frac * bar_w)
            if fill_w > 0:
                pygame.draw.rect(self.screen, PROGRESS_FILL, (bar_x, y, fill_w, bar_h), border_radius=3)

            # Tick marks for every 30s
            for sec in range(30, int(self.song_end_ms / 1000) + 1, 30):
                tx = bar_x + int((sec * 1000 / self.song_end_ms) * bar_w)
                pygame.draw.line(self.screen, (100, 100, 120), (tx, y), (tx, y + bar_h), 1)

            # Cursor
            cx = bar_x + fill_w
            pygame.draw.rect(self.screen, PROGRESS_CURSOR, (cx - 1, y - 2, 3, bar_h + 4), border_radius=1)

    def _draw_mel_view(self, y_start):
        """Draw scrolling mel spectrogram + waveform aligned to playfield cursor.
        Uses numpy + pygame.surfarray for fast rendering."""
        if not HAS_NUMPY:
            return
        view_w = self.w - 20
        mel_h = 80
        wave_h = 40
        pad_x = 10

        px_per_frame = SCROLL_SPEED * self.zoom * MEL_BIN_MS
        cursor_frame = int(self.now_ms / MEL_BIN_MS)

        frames_left = int((HIT_X - pad_x) / px_per_frame)
        frames_right = int((view_w - HIT_X + pad_x) / px_per_frame)
        frame_start = cursor_frame - frames_left
        frame_end = cursor_frame + frames_right

        label = self.font_small.render(
            "Mel Spectrogram + Waveform (W to toggle)" if self.mel_data is not None else "Waveform (W to toggle)",
            True, DIM_TEXT)
        self.screen.blit(label, (pad_x, y_start))
        y = y_start + 14

        # --- Mel spectrogram (fast path via surfarray) ---
        if self.mel_data is not None:
            pygame.draw.rect(self.screen, PANEL_BG, (pad_x, y, view_w, mel_h), border_radius=3)

            cmap = np.array(_get_mel_colormap(), dtype=np.uint8)  # (256, 3)
            n_mels, total_mel_frames = self.mel_data.shape

            f0 = max(0, frame_start)
            f1 = min(total_mel_frames, frame_end)
            n_total = frame_end - frame_start

            if f1 > f0 and n_total > 0:
                mel_slice = self.mel_data[:, f0:f1]
                mel_range = self._mel_global_max - self._mel_global_min
                if mel_range <= 0:
                    mel_range = 1.0
                mel_norm = np.clip((mel_slice - self._mel_global_min) / mel_range * 255, 0, 255).astype(np.uint8)

                # Flip vertically so low freq is at bottom
                mel_norm = mel_norm[::-1, :]

                # Map to RGB via colormap: (n_mels, n_visible) -> (n_mels, n_visible, 3)
                mel_rgb = cmap[mel_norm]  # (n_mels, n_visible, 3)

                # Create pixel array at native mel resolution, then scale
                # pygame.surfarray wants (width, height, 3) i.e. (cols, rows, 3)
                mel_rgb_t = mel_rgb.transpose(1, 0, 2)  # (n_visible, n_mels, 3)

                # Make surface at mel resolution
                mel_surf = pygame.surfarray.make_surface(mel_rgb_t)

                # Compute where this slice sits in the view
                slice_x_start = int((f0 - frame_start) * px_per_frame)
                slice_pixel_w = max(1, int((f1 - f0) * px_per_frame))

                # Scale to display size
                scaled = pygame.transform.scale(mel_surf, (slice_pixel_w, mel_h))
                self.screen.blit(scaled, (pad_x + slice_x_start, y))

            # Cursor line
            cx = pad_x + int(frames_left * px_per_frame)
            pygame.draw.line(self.screen, (255, 255, 255), (cx, y), (cx, y + mel_h), 1)

            # Onset markers along bottom edge of mel
            for t_ms, kind in self.onsets:
                f = int(t_ms / MEL_BIN_MS)
                if frame_start <= f <= frame_end:
                    ox = pad_x + int((f - frame_start) * px_per_frame)
                    c = COLORS.get(kind, (200, 200, 200))
                    pygame.draw.line(self.screen, c, (ox, y + mel_h - 5), (ox, y + mel_h), 2)

            # Second time labels
            for sec in range(max(0, int(frame_start * MEL_BIN_MS / 1000)),
                             int(frame_end * MEL_BIN_MS / 1000) + 1):
                f = int(sec * 1000 / MEL_BIN_MS)
                if frame_start <= f <= frame_end:
                    lx = pad_x + int((f - frame_start) * px_per_frame)
                    lbl = self.font_small.render(f"{sec}s", True, (200, 200, 200))
                    self.screen.blit(lbl, (lx + 2, y + 1))
                    pygame.draw.line(self.screen, (80, 80, 100), (lx, y), (lx, y + mel_h), 1)

            y += mel_h + 4

        # --- Waveform (fast path via surfarray) ---
        if self.wave_data is not None:
            pygame.draw.rect(self.screen, PANEL_BG, (pad_x, y, view_w, wave_h), border_radius=3)

            total_wave_frames = len(self.wave_data)
            f0 = max(0, frame_start)
            f1 = min(total_wave_frames, frame_end)
            n_total = frame_end - frame_start

            if f1 > f0 and n_total > 0:
                wave_slice = self.wave_data[f0:f1]
                amp = wave_slice / self._wave_global_max  # normalized 0-1 using global max

                # Build RGB image: (n_visible, wave_h, 3)
                n_vis = f1 - f0
                img = np.zeros((n_vis, wave_h, 3), dtype=np.uint8)
                img[:, :] = (28, 28, 38)  # panel bg

                center = wave_h // 2
                for col in range(n_vis):
                    a = amp[col]
                    bar_h = int(a * (wave_h // 2 - 2))
                    if bar_h > 0:
                        r = min(255, int(80 + a * 175))
                        g = min(255, int(140 + a * 60))
                        b = min(255, int(220 - a * 80))
                        img[col, center - bar_h:center + bar_h + 1] = (r, g, b)

                wave_surf = pygame.surfarray.make_surface(img)
                slice_x_start = int((f0 - frame_start) * px_per_frame)
                slice_pixel_w = max(1, int((f1 - f0) * px_per_frame))
                scaled = pygame.transform.scale(wave_surf, (slice_pixel_w, wave_h))
                self.screen.blit(scaled, (pad_x + slice_x_start, y))

            # Cursor line
            cx = pad_x + int(frames_left * px_per_frame)
            pygame.draw.line(self.screen, (255, 255, 255), (cx, y), (cx, y + wave_h), 1)

            # Onset markers along top
            for t_ms, kind in self.onsets:
                f = int(t_ms / MEL_BIN_MS)
                if frame_start <= f <= frame_end:
                    ox = pad_x + int((f - frame_start) * px_per_frame)
                    c = COLORS.get(kind, (200, 200, 200))
                    pygame.draw.line(self.screen, c, (ox, y), (ox, y + 4), 2)

            y += wave_h + 4

        # --- Sampling timeline (temperature + metronome) ---
        if self.sampling_data is not None:
            self._draw_sampling_bar(y, frame_start, frame_end, frames_left, px_per_frame)

    def _draw_sampling_bar(self, y_start, frame_start, frame_end, frames_left, px_per_frame):
        """Draw temperature + metronome closeness bar aligned to mel/wave view."""
        if self.sampling_data is None or not HAS_NUMPY:
            return
        bar_h = 30
        pad_x = 10
        view_w = self.w - 20

        label = self.font_small.render("Temperature (orange) / Metronome (cyan)", True, DIM_TEXT)
        self.screen.blit(label, (pad_x, y_start))
        y = y_start + 14

        pygame.draw.rect(self.screen, PANEL_BG, (pad_x, y, view_w, bar_h), border_radius=3)

        # sampling_data: (N, 3) = [cursor_bin, temperature, closeness]
        bins = self.sampling_data[:, 0]
        temps = self.sampling_data[:, 1]
        closes = self.sampling_data[:, 2]

        # find max temperature for normalization
        temp_max = max(float(temps.max()), 1.0)

        # draw each data point that falls in visible range
        for i in range(len(bins)):
            f = int(bins[i])
            if f < frame_start or f > frame_end:
                continue
            px = pad_x + int((f - frame_start) * px_per_frame)
            if px < pad_x or px > pad_x + view_w:
                continue

            # temperature bar (orange, from bottom)
            t_norm = min(temps[i] / temp_max, 1.0)
            t_h = int(t_norm * (bar_h - 2))
            if t_h > 0:
                r = min(255, int(200 + t_norm * 55))
                g = min(255, int(120 + t_norm * 40))
                pygame.draw.line(self.screen, (r, g, 30),
                                 (px, y + bar_h - 1), (px, y + bar_h - 1 - t_h), 1)

            # metronome closeness (cyan, from top)
            c_h = int(closes[i] * (bar_h - 2))
            if c_h > 0:
                pygame.draw.line(self.screen, (30, 200, 220),
                                 (px, y + 1), (px, y + 1 + c_h), 1)

        # cursor line
        cx = pad_x + int(frames_left * px_per_frame)
        pygame.draw.line(self.screen, (255, 255, 255), (cx, y), (cx, y + bar_h), 1)

    def _draw_density_graph(self, y_start):
        """Density over time graph."""
        if not self._density_surface:
            return

        label = self.font_small.render("Density (events/sec)", True, DIM_TEXT)
        self.screen.blit(label, (10, y_start))

        graph_y = y_start + 14
        graph_w = self.w - 20
        graph_h = 45

        pygame.draw.rect(self.screen, PANEL_BG, (10, graph_y, graph_w, graph_h), border_radius=3)

        # Scale density surface to fit
        if self._compare_density_surface:
            scaled = pygame.transform.smoothscale(self._compare_density_surface, (graph_w, graph_h))
            self.screen.blit(scaled, (10, graph_y))

        scaled = pygame.transform.smoothscale(self._density_surface, (graph_w, graph_h))
        self.screen.blit(scaled, (10, graph_y))

        # Current position marker
        if self.song_end_ms > 0:
            frac = max(0, min(1, self.now_ms / self.song_end_ms))
            cx = 10 + int(frac * graph_w)
            pygame.draw.line(self.screen, (255, 255, 255, 180), (cx, graph_y), (cx, graph_y + graph_h), 1)

        # Peak/mean labels
        peak = self.stats.get("peak_density", 0)
        mean = self.stats.get("mean_density", 0)
        info = self.font_small.render(f"mean={mean:.1f}/s  peak={peak:.1f}/s", True, DIM_TEXT)
        self.screen.blit(info, (self.w - info.get_width() - 14, y_start))

    def _draw_minimap(self, y_start):
        """Horizontal minimap showing all onset positions."""
        label = self.font_small.render("Timeline", True, DIM_TEXT)
        self.screen.blit(label, (10, y_start))

        map_y = y_start + 14
        map_w = self.w - 20
        map_h = 16

        pygame.draw.rect(self.screen, PANEL_BG, (10, map_y, map_w, map_h), border_radius=2)

        if self.song_end_ms > 0:
            # Draw onset ticks
            for time_ms, kind in self.onsets:
                frac = time_ms / self.song_end_ms
                x = 10 + int(frac * map_w)
                color = COLORS.get(kind, (150, 150, 150))
                color = tuple(min(255, c + 30) for c in color)
                pygame.draw.line(self.screen, color, (x, map_y + 2), (x, map_y + map_h - 2), 1)

            # Comparison onsets
            if self.compare_onsets:
                for time_ms, kind in self.compare_onsets:
                    frac = time_ms / self.song_end_ms
                    x = 10 + int(frac * map_w)
                    pygame.draw.line(self.screen, (255, 180, 80), (x, map_y + 1), (x, map_y + 5), 1)

            # Cursor
            frac = max(0, min(1, self.now_ms / self.song_end_ms))
            cx = 10 + int(frac * map_w)
            pygame.draw.rect(self.screen, (255, 255, 255), (cx - 1, map_y - 1, 3, map_h + 2), border_radius=1)

    def _compute_local_metronome(self):
        """Compute metronome % around cursor: what % of recent gaps match the dominant gap."""
        # gather onsets in the last 2s before cursor
        cursor_ms = self.now_ms
        recent = [t for t, _ in self.onsets if cursor_ms - 2000 <= t <= cursor_ms]
        if len(recent) < 4:
            return None, None

        gaps = []
        for i in range(1, len(recent)):
            g = recent[i] - recent[i - 1]
            if g > 0:
                gaps.append(g)
        if len(gaps) < 3:
            return None, None

        # cluster gaps within 5% to find dominant
        sorted_gaps = sorted(gaps)
        clusters = []
        cluster_vals = [sorted_gaps[0]]
        for i in range(1, len(sorted_gaps)):
            centroid = sum(cluster_vals) / len(cluster_vals)
            if centroid > 0 and abs(sorted_gaps[i] - centroid) / centroid <= 0.05:
                cluster_vals.append(sorted_gaps[i])
            else:
                clusters.append((sum(cluster_vals) / len(cluster_vals), len(cluster_vals)))
                cluster_vals = [sorted_gaps[i]]
        clusters.append((sum(cluster_vals) / len(cluster_vals), len(cluster_vals)))
        clusters.sort(key=lambda x: x[1], reverse=True)

        dominant_gap = clusters[0][0]
        dominant_count = clusters[0][1]
        total = len(gaps)
        in_peak_pct = dominant_count / total * 100
        dominant_bpm = 60000 / dominant_gap if dominant_gap > 0 else 0

        # color based on how metronomic
        if in_peak_pct >= 80:
            color = (255, 80, 80)   # red = very metronomic
        elif in_peak_pct >= 50:
            color = (255, 200, 60)  # yellow = moderate
        else:
            color = (100, 220, 100) # green = varied

        return f"{in_peak_pct:.0f}% in peak ({dominant_gap:.0f}ms / {dominant_bpm:.0f}BPM)  {len(gaps)} gaps  {len(clusters)} clusters", color

    def _draw_candidate_bars(self, x, y, max_w):
        """Draw horizontal confidence bars for the nearest prediction's candidates."""
        pred_cursor, chosen, cands = self._find_next_prediction()
        if cands is None or not cands:
            return

        label = self.font_small.render("Next candidates:", True, DIM_TEXT)
        self.screen.blit(label, (x, y))
        y += 14

        chosen_bin = pred_cursor + chosen if chosen < 500 else None
        total_conf = sum(c[2] for c in cands) or 1.0
        bar_h = 6
        for i, (cand_bin, raw_conf, final_conf) in enumerate(cands[:5]):
            pct = final_conf / total_conf
            bar_w = max(2, int(pct * max_w))
            is_chosen = chosen_bin is not None and abs(cand_bin - chosen_bin) < 3
            color = (100, 220, 100) if is_chosen else (180, 120, 255)
            pygame.draw.rect(self.screen, color, (x, y, bar_w, bar_h))

            cand_ms = cand_bin * MEL_BIN_MS
            lbl = self.font_small.render(f"{cand_ms:.0f}ms {pct*100:.0f}%", True, TEXT_COLOR)
            self.screen.blit(lbl, (x + bar_w + 4, y - 2))
            y += bar_h + 2

    def _draw_stats_panel(self, y_start):
        """Statistics panel at the bottom."""
        panel_x = 10
        panel_w = self.w - 20
        panel_h = self.h - y_start - 10

        if panel_h < 30:
            return

        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, y_start, panel_w, panel_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (panel_x, y_start, panel_w, panel_h), 1, border_radius=4)

        s = self.stats
        if not s:
            return

        col_x = panel_x + 12
        y = y_start + 8

        def text(label_str, value_str, color=TEXT_COLOR, label_color=DIM_TEXT):
            nonlocal y
            if y > y_start + panel_h - 16:
                return
            lbl = self.font_small.render(label_str, True, label_color)
            val = self.font.render(str(value_str), True, color)
            self.screen.blit(lbl, (col_x, y))
            self.screen.blit(val, (col_x + lbl.get_width() + 6, y - 1))
            y += 17

        def section(title_str):
            nonlocal y
            if y > y_start + panel_h - 16:
                return
            t = self.font_big.render(title_str, True, ACCENT)
            self.screen.blit(t, (col_x, y))
            y += 20

        # --- Column 1: Basic stats ---
        section("Level Stats")
        text("Events:", f"{s['total_events']}")
        text("Duration:", f"{s['duration_s']:.1f}s ({format_time(s['duration_ms'])})")

        # Type breakdown
        type_parts = []
        for kind in ["don", "ka", "big_don", "big_ka", "drumroll", "spinner", "predicted"]:
            c = s["type_counts"].get(kind, 0)
            if c > 0:
                pct = c / s["total_events"] * 100
                type_parts.append(f"{kind}={c} ({pct:.0f}%)")
        text("Types:", "  ".join(type_parts[:3]))
        if len(type_parts) > 3:
            text("", "  ".join(type_parts[3:]))

        text("Density:", f"mean={s['mean_density']:.1f}/s  peak={s['peak_density']:.1f}/s  std={s['density_std']:.1f}")
        if s["bpm_estimate"] > 0:
            text("Est. BPM:", f"{s['bpm_estimate']:.0f}")
        text("IOI:", f"avg={s['avg_ioi_ms']:.0f}ms  med={s['median_ioi_ms']:.0f}ms  min={s['min_ioi_ms']:.0f}ms  max={s['max_ioi_ms']:.0f}ms")

        # Local density around cursor
        local_count = sum(1 for t, _ in self.onsets if abs(t - self.now_ms) < 500)
        text("Local density:", f"{local_count:.0f} events/s (around cursor)")

        # Metronome analysis around cursor (last 2s of events before cursor)
        met_str, met_color = self._compute_local_metronome()
        if met_str:
            text("Metronome:", met_str, color=met_color)

        # Candidate confidence bars for the next prediction near cursor
        if self._cand_by_cursor:
            self._draw_candidate_bars(col_x, y, panel_w // 2 - 24)
            y += 50

        # --- Column 2: Inference stats (if available) ---
        col2_x = panel_x + panel_w // 2
        if self.inference_stats or s.get("is_predicted"):
            y_save = y
            y = y_start + 8
            col_x_save = col_x
            col_x = col2_x

            section("Inference Stats")

            if self.inference_stats:
                ist = self.inference_stats
                text("Checkpoint:", os.path.basename(ist.get("checkpoint", "?")))
                text("Epoch:", f"{ist.get('epoch', '?')}")
                text("Val loss:", f"{ist.get('val_loss', 0):.4f}")
                text("Val acc:", f"{ist.get('val_accuracy', 0):.3f}")
                text("Device:", ist.get("device", "?"))
                text("Inference time:", f"{ist.get('inference_time_s', 0):.1f}s")
                text("Events/sec (real):", f"{ist.get('events_per_sec_realtime', 0):.1f}")

                cond = ist.get("conditioning", {})
                text("Conditioning:", f"mean={cond.get('mean', 0):.1f}  peak={cond.get('peak', 0):.1f}  std={cond.get('std', 0):.1f}")

                text("STOP predictions:", f"{ist.get('stop_count', 0)}")
                text("Avg STOP gap:", f"{ist.get('avg_stop_gap_ms', 0):.0f}ms")
                text("Total model calls:", f"{ist.get('total_model_calls', 0)}")

                # Timing stats
                if "timing" in ist:
                    timing = ist["timing"]
                    text("Audio load:", f"{timing.get('audio_load_s', 0):.2f}s")
                    text("Model load:", f"{timing.get('model_load_s', 0):.2f}s")
                    text("Inference:", f"{timing.get('inference_s', 0):.2f}s")
                    text("ms/event:", f"{timing.get('ms_per_event', 0):.1f}")
                    text("ms/model call:", f"{timing.get('ms_per_call', 0):.1f}")

                # Temperature stats
                if "temperature" in ist:
                    ts = ist["temperature"]
                    text("Temperature:", f"mean={ts['mean']:.2f}  med={ts['median']:.2f}  [{ts['min']:.2f}, {ts['max']:.2f}]")

                # Metronome detection stats
                if "metronome" in ist:
                    ms = ist["metronome"]
                    text("Met closeness:", f"mean={ms['closeness_mean']:.2f}  med={ms['closeness_median']:.2f}  p90={ms['closeness_p90']:.2f}")
                    text("Met multiplier:", f"mean={ms['multiplier_mean']:.2f}  max={ms['multiplier_max']:.2f}")

                # Event distribution
                if "event_distribution" in ist:
                    ed = ist["event_distribution"]
                    text("Offset range:", f"{ed.get('min_offset', 0)}-{ed.get('max_offset', 0)} bins")
                    text("Avg offset:", f"{ed.get('mean_offset', 0):.1f} bins ({ed.get('mean_offset_ms', 0):.1f}ms)")
                    text("Median offset:", f"{ed.get('median_offset', 0)} bins")

            else:
                text("(Run with --stats-json for", "full inference stats)")

            col_x = col_x_save
            y = max(y, y_save)

        # --- Comparison stats ---
        if self.compare_stats:
            y += 5
            col_x_save = col_x
            col_x = col2_x if not (self.inference_stats or s.get("is_predicted")) else panel_x + 12

            section("Comparison")
            cs = self.compare_stats
            text("Compare events:", f"{cs['total_events']}")
            text("Compare density:", f"mean={cs['mean_density']:.1f}/s  peak={cs['peak_density']:.1f}/s")
            diff = s['total_events'] - cs['total_events']
            text("Event diff:", f"{diff:+d} ({diff/max(cs['total_events'],1)*100:+.1f}%)",
                 color=(100, 255, 100) if abs(diff) < cs['total_events'] * 0.1 else (255, 100, 100))
            col_x = col_x_save

        # IOI histogram (small, inline)
        ioi_buckets = s.get("ioi_buckets", {})
        if ioi_buckets and y + 50 < y_start + panel_h:
            y += 4
            hist_x = panel_x + 12
            hist_label = self.font_small.render("IOI Distribution (ms)", True, DIM_TEXT)
            self.screen.blit(hist_label, (hist_x, y))
            y += 14

            # Filter to reasonable range and draw
            filtered = {k: v for k, v in ioi_buckets.items() if 0 < k <= 2000}
            if filtered:
                max_count = max(filtered.values())
                hist_w = min(panel_w - 24, 500)
                hist_h = 35
                max_bucket = max(filtered.keys())
                min_bucket = min(filtered.keys())
                bucket_range = max(max_bucket - min_bucket, 1)

                pygame.draw.rect(self.screen, (35, 35, 48), (hist_x, y, hist_w, hist_h), border_radius=2)
                for bucket, count in sorted(filtered.items()):
                    bx = hist_x + int((bucket - min_bucket) / bucket_range * hist_w)
                    bh = max(1, int(count / max_count * (hist_h - 2)))
                    color = ACCENT if bucket < 500 else (150, 150, 180)
                    pygame.draw.line(self.screen, color, (bx, y + hist_h), (bx, y + hist_h - bh), 1)

                # Axis labels
                for ms in range(0, int(max_bucket) + 1, 250):
                    if ms >= min_bucket:
                        lx = hist_x + int((ms - min_bucket) / bucket_range * hist_w)
                        lbl = self.font_small.render(f"{ms}", True, DIM_TEXT)
                        self.screen.blit(lbl, (lx - 8, y + hist_h + 1))

    def _draw_help_overlay(self):
        """Semi-transparent help overlay."""
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        lines = [
            ("CONTROLS", None),
            ("", ""),
            ("Space", "Pause / Resume"),
            ("Left/Right", "Seek -5s / +5s"),
            ("Shift+Left/Right", "Seek -1s / +1s"),
            ("Up/Down", "Volume up / down"),
            ("+/-", "Speed up / down (0.25x)"),
            ("R", "Restart from beginning"),
            ("E", "Export to video (.mp4)"),
            ("H", "Toggle this help"),
            ("T", "Toggle stats panel"),
            ("M", "Toggle minimap"),
            ("D", "Toggle density graph"),
            ("W", "Toggle mel spectrogram + waveform"),
            ("Scroll wheel", "Zoom in / out"),
            ("Click progress bar", "Seek to position"),
            ("Esc / Q", "Quit"),
            ("", ""),
            ("LEGEND", None),
        ]

        y = 80
        cx = self.w // 2
        for key, desc in lines:
            if desc is None:
                t = self.font_title.render(key, True, ACCENT)
                self.screen.blit(t, (cx - t.get_width() // 2, y))
                y += 28
            elif key == "":
                y += 8
            else:
                kt = self.font_big.render(key, True, (255, 255, 255))
                dt = self.font.render(desc, True, TEXT_COLOR)
                self.screen.blit(kt, (cx - 180, y))
                self.screen.blit(dt, (cx + 40, y + 2))
                y += 24

        # Legend
        y += 5
        for kind, color in [("don", COLORS["don"]), ("ka", COLORS["ka"]),
                             ("big_don", COLORS["big_don"]), ("big_ka", COLORS["big_ka"]),
                             ("drumroll", COLORS["drumroll"]), ("spinner", COLORS["spinner"]),
                             ("predicted", COLORS["predicted"])]:
            pygame.draw.circle(self.screen, color, (cx - 160, y + 8), SIZES.get(kind, 16) // 2 + 2)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx - 160, y + 8), SIZES.get(kind, 16) // 2 + 2, 1)
            t = self.font.render(kind, True, TEXT_COLOR)
            self.screen.blit(t, (cx - 140, y + 2))
            y += 22

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        # Clean up temp audio file
        if self._speed_tmp_file:
            try:
                os.unlink(self._speed_tmp_file)
            except OSError:
                pass

    def render_video(self, output_path, fps=60):
        """Render the taiko view to a video file with audio + hit sounds."""
        import numpy as np

        print(f"Rendering to {output_path} at {fps}fps...")

        # generate tick sounds as numpy arrays for mixing (white noise clap, matches viewer)
        sr = 44100
        tick_dur = 0.04  # 40ms
        n_tick = int(sr * tick_dur)

        def make_noise_tick(vol=0.7):
            noise = np.random.uniform(-1, 1, n_tick)
            fade = (1.0 - (np.arange(n_tick) / n_tick) ** 0.5)  # fast decay
            return (noise * fade * vol * 32767).astype(np.int16)

        don_tick = make_noise_tick(vol=0.7)
        ka_tick = make_noise_tick(vol=0.5)

        # load source audio as wav for mixing
        audio_data = None
        audio_sr = sr
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_streams", self.audio_path],
                    capture_output=True, text=True
                )
                # convert to raw pcm via ffmpeg
                pcm_result = subprocess.run(
                    ["ffmpeg", "-i", self.audio_path, "-f", "s16le",
                     "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-"],
                    capture_output=True, timeout=60
                )
                if pcm_result.returncode == 0:
                    audio_data = np.frombuffer(pcm_result.stdout, dtype=np.int16).astype(np.float64)
                    print(f"  Audio loaded: {len(audio_data)/sr:.1f}s")
            except Exception as e:
                print(f"  Could not load audio for mixing: {e}")

        # determine video duration
        duration_ms = self.song_end_ms + 2000
        n_frames = int(duration_ms / 1000 * fps)
        ms_per_frame = 1000.0 / fps

        # mix audio with tick sounds
        if audio_data is not None:
            total_samples = max(len(audio_data), int(duration_ms / 1000 * sr))
            mixed = np.zeros(total_samples, dtype=np.float64)
            mixed[:len(audio_data)] = audio_data
            # add tick sounds at onset positions
            for onset_ms, kind in self.onsets:
                sample_pos = int(onset_ms / 1000 * sr)
                tick = ka_tick if "ka" in kind else don_tick
                end = min(sample_pos + len(tick), total_samples)
                if sample_pos >= 0 and sample_pos < total_samples:
                    mixed[sample_pos:end] += tick[:end - sample_pos]
            # normalize
            peak = np.abs(mixed).max()
            if peak > 32767:
                mixed = mixed * (32767 / peak)
            mixed_pcm = mixed.astype(np.int16).tobytes()
        else:
            mixed_pcm = None

        # set up pygame surface (offscreen, dimensions must be even for h264)
        render_w, render_h = 1200, 300
        surface = pygame.Surface((render_w, render_h))

        # start ffmpeg video pipe
        tmp_audio = None
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{render_w}x{render_h}",
            "-r", str(fps),
            "-i", "-",  # video from stdin
        ]

        if mixed_pcm:
            # write mixed audio to temp file
            tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp_audio.name, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(mixed_pcm)
            ffmpeg_cmd += ["-i", tmp_audio.name]  # audio input
            ffmpeg_cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
                          "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-b:a", "192k",
                          "-shortest", output_path]
        else:
            ffmpeg_cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
                          "-pix_fmt", "yuv420p",
                          output_path]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # render frames
        hit_circle_x = 120
        scroll_px_per_ms = 0.5
        font = pygame.font.SysFont("consolas", 16)
        font_big = pygame.font.SysFont("consolas", 22, bold=True)

        recent_hits = []  # (time_ms, kind) for flash effect
        next_hit = 0

        from tqdm import tqdm
        for frame_i in tqdm(range(n_frames), desc="Rendering"):
            now_ms = frame_i * ms_per_frame

            # trigger hits
            while next_hit < len(self.onsets) and self.onsets[next_hit][0] <= now_ms:
                recent_hits.append((self.onsets[next_hit][0], self.onsets[next_hit][1]))
                next_hit += 1
            recent_hits = [(t, k) for t, k in recent_hits if now_ms - t < 200]

            # draw
            surface.fill((22, 22, 30))

            # header
            time_str = f"{int(now_ms/60000):02d}:{int(now_ms/1000)%60:02d}"
            total_str = f"{int(duration_ms/60000):02d}:{int(duration_ms/1000)%60:02d}"
            header = font.render(f"{os.path.basename(self.csv_path)}  {time_str}/{total_str}  "
                                f"Notes: {len(self.onsets)}", True, (200, 200, 210))
            surface.blit(header, (10, 8))

            # playfield background
            pf_top = 40
            pf_h = 120
            pf_center = pf_top + pf_h // 2
            pygame.draw.rect(surface, (30, 30, 42), (0, pf_top, render_w, pf_h))

            # hit line
            pygame.draw.line(surface, (255, 255, 255), (hit_circle_x, pf_top), (hit_circle_x, pf_top + pf_h), 2)

            # draw approaching notes
            for onset_ms, kind in self.onsets:
                dx = (onset_ms - now_ms) * scroll_px_per_ms
                x = hit_circle_x + dx
                if x < -40 or x > render_w + 40:
                    continue
                is_big = "big" in kind
                radius = 28 if is_big else 20
                if "ka" in kind:
                    color = (80, 165, 230) if is_big else (68, 141, 199)
                elif "drumroll" in kind:
                    color = (252, 183, 30)
                    radius = 18
                else:
                    color = (255, 90, 60) if is_big else (235, 69, 44)
                pygame.draw.circle(surface, color, (int(x), pf_center), radius)
                # white inner circle
                if "drumroll" not in kind:
                    pygame.draw.circle(surface, (255, 255, 255), (int(x), pf_center), radius // 3)

            # hit flash
            for t, k in recent_hits:
                age = now_ms - t
                alpha = max(0, 1.0 - age / 200)
                flash_r = int(40 * (1 + age / 100))
                flash_surface = pygame.Surface((flash_r * 2, flash_r * 2), pygame.SRCALPHA)
                flash_color = (68, 141, 199) if "ka" in k else (235, 69, 44)
                pygame.draw.circle(flash_surface, (*flash_color, int(alpha * 150)),
                                  (flash_r, flash_r), flash_r)
                surface.blit(flash_surface, (hit_circle_x - flash_r, pf_center - flash_r))

            # progress bar
            bar_y = pf_top + pf_h + 10
            bar_h = 6
            pygame.draw.rect(surface, (40, 40, 55), (10, bar_y, render_w - 20, bar_h))
            if duration_ms > 0:
                progress = min(now_ms / duration_ms, 1.0)
                pygame.draw.rect(surface, (80, 120, 220),
                               (10, bar_y, int((render_w - 20) * progress), bar_h))

            # bottom info
            info_y = bar_y + 14
            passed = sum(1 for t, _ in self.onsets if t <= now_ms)
            info = font.render(f"Passed: {passed}/{len(self.onsets)}", True, (120, 120, 135))
            surface.blit(info, (10, info_y))

            # write frame to ffmpeg
            frame_bytes = pygame.image.tobytes(surface, "RGB")
            try:
                proc.stdin.write(frame_bytes)
            except BrokenPipeError:
                print("ffmpeg pipe broken!")
                break

        proc.stdin.close()
        proc.wait()

        # cleanup
        if tmp_audio:
            try:
                os.unlink(tmp_audio.name)
            except OSError:
                pass

        print(f"Done! Saved to {output_path}")


def main():
    args = parse_args()
    csv_path = args.csv or pick_csv()
    if not csv_path:
        print("No file selected.")
        return
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    if args.render:
        # render mode: no interactive window needed, just init pygame for drawing
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        viewer = Viewer(csv_path, audio_override=args.audio, compare_csv=args.compare,
                        stats_json_path=args.stats_json,
                        mel_npy_path=getattr(args, 'mel_npy', None),
                        wave_npy_path=getattr(args, 'wave_npy', None),
                        sampling_npy_path=getattr(args, 'sampling_npy', None),
                        candidates_json_path=getattr(args, 'candidates_json', None),
                        gif_path=args.gif, gif_cycles=args.gif_cycles)
        viewer.render_video(args.render, fps=args.render_fps)
        pygame.quit()
    else:
        viewer = Viewer(csv_path, audio_override=args.audio, compare_csv=args.compare,
                        stats_json_path=args.stats_json,
                        mel_npy_path=getattr(args, 'mel_npy', None),
                        wave_npy_path=getattr(args, 'wave_npy', None),
                        sampling_npy_path=getattr(args, 'sampling_npy', None),
                        candidates_json_path=getattr(args, 'candidates_json', None),
                        gif_path=args.gif, gif_cycles=args.gif_cycles)
        viewer.run()


if __name__ == "__main__":
    main()
