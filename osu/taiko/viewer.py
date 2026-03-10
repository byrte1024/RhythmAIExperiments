"""osu!taiko onset visualizer with full controls, stats, and inference analysis.

Controls:
  Space     - Pause / Resume
  Left/Right- Seek -5s / +5s  (hold Shift for 1s)
  Up/Down   - Volume up/down
  +/-       - Speed up/down (0.25x steps)
  R         - Restart
  H         - Toggle help overlay
  T         - Toggle stats panel
  M         - Toggle minimap
  D         - Toggle density graph
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


class Viewer:
    def __init__(self, csv_path, audio_override=None, compare_csv=None, stats_json_path=None):
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
            except Exception as e:
                print(f"Could not load audio: {e}")
        else:
            if audio_name:
                print(f"Audio not found: {audio_name}")

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

            if ev.type == pygame.MOUSEBUTTONDOWN:
                # Click on progress bar to seek
                mx, my = ev.pos
                prog_y = PLAYFIELD_TOP + PLAYFIELD_H + 5
                if prog_y <= my <= prog_y + 20 and self.song_end_ms > 0:
                    frac = max(0, min(1, (mx - 10) / (self.w - 20)))
                    self._seek(frac * self.song_end_ms - self.now_ms)

            if ev.type == pygame.VIDEORESIZE:
                self.w, self.h = ev.w, ev.h
                self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)

        return True

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
        hint = self.font_small.render("H=Help  T=Stats  M=Map  D=Density", True, DIM_TEXT)
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

        # Note count near hit line
        passed = self.next_hit
        total = len(self.onsets)
        counter = self.font_small.render(f"{passed}/{total}", True, DIM_TEXT)
        self.screen.blit(counter, (HIT_X - counter.get_width() // 2, PLAYFIELD_TOP + PLAYFIELD_H - 18))

    def _draw_notes(self, onsets, cy, alpha_mod=1.0, outline_only=False):
        """Draw note circles on the playfield."""
        for time_ms, kind in onsets:
            x = HIT_X + (time_ms - self.now_ms) * SCROLL_SPEED
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
            ("H", "Toggle this help"),
            ("T", "Toggle stats panel"),
            ("M", "Toggle minimap"),
            ("D", "Toggle density graph"),
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


def main():
    args = parse_args()
    csv_path = args.csv or pick_csv()
    if not csv_path:
        print("No file selected.")
        return
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    viewer = Viewer(csv_path, audio_override=args.audio, compare_csv=args.compare,
                    stats_json_path=args.stats_json)
    viewer.run()


if __name__ == "__main__":
    main()
