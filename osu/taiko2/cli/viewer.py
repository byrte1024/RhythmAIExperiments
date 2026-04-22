"""osu!taiko chart viewer — game-view playback with mel / density / stats.

Faithful port of the taiko1 viewer adapted to the taiko2 `Chart` data
model. Plays a chart against its audio and overlays inference
diagnostics loaded from the sidecar files `cli.infer --debug` produces.

Loads:
  - ``.osu``                     — single chart. No audio unless paired
                                    externally; playback silent.
  - ``.osz``                     — shows an index picker for the charts
                                    inside, plays with the embedded audio.
  - taiko2 bundle (``.zip``)     — direct round-trip of Chart.save output.

Controls:
  Space      - Pause / Resume
  Left/Right - Seek -5s / +5s   (hold Shift for 1s)
  Up/Down    - Volume up / down
  +/-        - Zoom in / out
  Scroll     - Zoom
  Click bar  - Seek
  R          - Restart
  H          - Toggle help overlay
  T          - Toggle tick synth
  I          - Toggle inline stats panel
  M          - Toggle minimap
  D          - Toggle density graph
  W          - Toggle mel spectrogram
  G          - Toggle ghost candidates   (requires --steps-log)
  N          - Toggle mute
  Esc / Q    - Quit
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..domain.beatmap import OnsetKind
from ..domain.chart import Chart
from ..parsing.osz import load_pack

# ─────────────────────────── layout (taiko1 parity) ───────────────────
WIDTH, HEIGHT = 1200, 760
PLAYFIELD_TOP = 60
PLAYFIELD_H = 170
PLAYFIELD_CENTER = PLAYFIELD_TOP + PLAYFIELD_H // 2
HIT_X = 160
SCROLL_SPEED = 0.5                # pixels per ms at zoom=1
FPS = 120

# Must match training constants for mel frame timing.
MEL_SAMPLE_RATE = 22000
MEL_HOP_DIVISOR = 200
MEL_BIN_MS = 1000.0 / MEL_HOP_DIVISOR     # 5.000 ms/frame

# ─────────────────────────── colors (taiko1 parity) ───────────────────
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

COLORS: dict[OnsetKind, tuple[int, int, int]] = {
    OnsetKind.DON:      (235,  69,  44),
    OnsetKind.KA:       ( 68, 141, 199),
    OnsetKind.BIG_DON:  (255,  90,  60),
    OnsetKind.BIG_KA:   ( 80, 165, 230),
    OnsetKind.DRUMROLL: (252, 183,  30),
    OnsetKind.SPINNER:  (100, 200, 100),
    OnsetKind.UNKNOWN:  (160, 160, 160),
}
SIZES: dict[OnsetKind, int] = {
    OnsetKind.DON:      18,
    OnsetKind.KA:       18,
    OnsetKind.BIG_DON:  28,
    OnsetKind.BIG_KA:   28,
    OnsetKind.DRUMROLL: 14,
    OnsetKind.SPINNER:  22,
    OnsetKind.UNKNOWN:  16,
}
GHOST_COLOR = (220, 170, 255)
GHOST_HIGHLIGHT = (255, 220, 255)

# ─────────────────────────── tick voicing ─────────────────────────────

TICK_VOICE: dict[OnsetKind, tuple[int, float]] = {
    OnsetKind.DON:      (50, 1.00),
    OnsetKind.KA:       (30, 0.85),
    OnsetKind.BIG_DON:  (65, 1.30),
    OnsetKind.BIG_KA:   (45, 1.15),
    OnsetKind.DRUMROLL: (20, 0.60),
    OnsetKind.SPINNER:  (25, 0.60),
    OnsetKind.UNKNOWN:  (25, 0.50),
}


# ─────────────────────────── helpers ──────────────────────────────────

def _format_time(ms: int) -> str:
    s = max(0, ms) / 1000.0
    m = int(s // 60)
    s = s - m * 60
    return f"{m}:{s:05.2f}"


_MEL_CMAP: list[tuple[int, int, int]] | None = None


def _get_mel_colormap() -> list[tuple[int, int, int]]:
    """256-entry colormap: black → dark blue → cyan → yellow → white.
    Matches the taiko1 viewer's mel rendering for visual parity."""
    global _MEL_CMAP
    if _MEL_CMAP is not None:
        return _MEL_CMAP
    stops = [
        (0,   (0, 0, 0)),
        (64,  (10, 10, 80)),
        (128, (20, 100, 160)),
        (192, (220, 200, 50)),
        (255, (255, 255, 255)),
    ]
    cmap: list[tuple[int, int, int]] = []
    for i in range(256):
        lo, hi = stops[0], stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= i <= stops[j + 1][0]:
                lo, hi = stops[j], stops[j + 1]
                break
        span = hi[0] - lo[0]
        t = (i - lo[0]) / span if span > 0 else 0
        r = int(lo[1][0] + t * (hi[1][0] - lo[1][0]))
        g = int(lo[1][1] + t * (hi[1][1] - lo[1][1]))
        b = int(lo[1][2] + t * (hi[1][2] - lo[1][2]))
        cmap.append((r, g, b))
    _MEL_CMAP = cmap
    return cmap


# ─────────────────────────── file picking ────────────────────────────

def _pick_file() -> Path | None:
    """Open a native file dialog. Returns None if the user cancels."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a chart",
        filetypes=[
            ("osu! taiko chart", "*.osu"),
            ("osu! pack",        "*.osz"),
            ("taiko2 bundle",    "*.zip"),
            ("All files",        "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def _pick_osz_index(pack_path: Path) -> int | None:
    """Simple pygame-window picker. Returns the selected index or None."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((720, 420))
    pygame.display.set_caption(f"Pick chart — {pack_path.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    small = pygame.font.SysFont("consolas", 11)
    title_font = pygame.font.SysFont("consolas", 18, bold=True)

    pack = load_pack(pack_path)
    if not pack.tracks:
        pygame.quit()
        return None

    hovered = 0
    scroll = 0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return None
                if ev.key == pygame.K_UP:
                    hovered = max(0, hovered - 1)
                elif ev.key == pygame.K_DOWN:
                    hovered = min(len(pack.tracks) - 1, hovered + 1)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    pygame.quit()
                    return hovered
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                idx = (my + scroll - 80) // 44
                if 0 <= idx < len(pack.tracks):
                    pygame.quit()
                    return int(idx)
            if ev.type == pygame.MOUSEWHEEL:
                scroll = max(0, scroll - ev.y * 40)

        screen.fill(BG_COLOR)
        title_surf = title_font.render(
            f"{pack.tracks[0].artist} — {pack.tracks[0].title}", True, TEXT_COLOR,
        )
        screen.blit(title_surf, (24, 24))
        screen.blit(small.render(
            f"{len(pack.tracks)} charts · click / ↑↓ + Enter",
            True, DIM_TEXT,
        ), (24, 56))

        y0 = 80 - scroll
        for i, t in enumerate(pack.tracks):
            row_top = y0 + i * 44
            if row_top < 72 or row_top > screen.get_height() - 12:
                continue
            rect = (12, row_top, screen.get_width() - 24, 40)
            bg = PANEL_BG if i != hovered else (48, 52, 80)
            pygame.draw.rect(screen, bg, rect, border_radius=6)
            pygame.draw.rect(screen, PANEL_BORDER, rect, width=1, border_radius=6)
            label = (
                f"[{i}] {t.difficulty.version}  ·  "
                f"OD {t.difficulty.overall_difficulty:g}  ·  "
                f"{t.density.total_events} events"
            )
            star = t.difficulty.star_rating
            if star is not None:
                label += f"  ·  {star:.2f}★"
            screen.blit(font.render(label[:120], True, TEXT_COLOR), (24, row_top + 10))

        pygame.display.flip()
        clock.tick(60)


# ─────────────────────────── audio handling ──────────────────────────

def _write_audio_tmp(chart: Chart) -> Path | None:
    if chart.audio is None:
        return None
    ext = chart.track.audio.format or "mp3"
    fd, name = tempfile.mkstemp(prefix="taiko2_audio_", suffix=f".{ext}")
    os.close(fd)
    path = Path(name)
    path.write_bytes(chart.audio)
    return path


def _bisect_onsets_ge(onsets, t_ms: int) -> int:
    lo, hi = 0, len(onsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if onsets[mid].time_ms < t_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _synth_tick(
    duration_ms: int, amp: float,
    mixer_freq: int, mixer_channels: int, *, volume: float = 0.9,
):
    import pygame
    n_samples = int(mixer_freq * duration_ms / 1000)
    buf = array.array("h")
    peak = int(volume * amp * 32767)
    peak = max(-32768, min(32767, peak))
    for i in range(n_samples):
        fade = 1.0 - (i / n_samples) ** 0.5
        val = int(peak * fade * (random.random() * 2 - 1))
        val = max(-32768, min(32767, val))
        for _ in range(mixer_channels):
            buf.append(val)
    return pygame.mixer.Sound(buffer=buf)


# ─────────────────────────── mel computation ─────────────────────────

def _compute_mel(chart: Chart) -> "tuple[Any, float] | None":
    """Decode chart audio bytes and compute the canonical (n_mels, T) mel.
    Returns (mel, frame_ms) or None on any failure. Uses taiko2's
    `MelSampler` so the spectrogram matches what the trained model sees."""
    if chart.audio is None:
        return None
    try:
        import numpy as np
        import librosa
    except ImportError:
        return None
    from ..samplers.mel import MelSampler
    from ..domain.dataset import MelSamplerConfig

    ext = chart.track.audio.format or "mp3"
    fd, tmp_name = tempfile.mkstemp(prefix="taiko2_viewer_mel_", suffix=f".{ext}")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_bytes(chart.audio)
        cfg = MelSamplerConfig(
            sample_rate=MEL_SAMPLE_RATE, n_fft=2048,
            hop_divisor=MEL_HOP_DIVISOR,
            n_mels=80, f_min=20.0, f_max=8000.0,
            power=2.0, top_db=80.0,
        )
        sampler = MelSampler(cfg)
        waveform, sr = librosa.load(str(tmp), sr=cfg.sample_rate, mono=True)
        mel = sampler.sample_waveform(waveform, int(sr))
        return np.asarray(mel, dtype=np.float32), sampler.frame_ms
    except Exception as exc:  # pragma: no cover — diagnostic only
        print(f"[viewer] mel compute failed: {exc}", file=sys.stderr)
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ─────────────────────────── ghost candidate log ─────────────────────

def _parse_steps_log(
    path: Path, *, b_pred: int = 500,
) -> "dict[int, tuple[int, list[tuple[int, float, float]]]] | None":
    """Parse a `cli.infer --debug` step log into the `_cand_by_cursor`
    shape the draw code expects:

        {cursor_bin: (chosen_abs_bin, [(abs_bin, prob, prob), ...])}

    - ``cursor_bin`` is absolute (from the log row).
    - ``top{i}_bin`` / ``top{i}_prob`` from the decoder's extras are
      converted to absolute bin via ``cursor_bin + top{i}_bin`` and
      filtered to non-STOP entries (``top{i}_bin != b_pred``).
    - ``chosen_abs_bin`` = first non-STOP of the bin_offsets list or, if
      the row was STOP, the cursor's own bin so ghosts still render.
    """
    if not path.exists():
        return None
    out: dict[int, tuple[int, list[tuple[int, float, float]]]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cursor_bin = d.get("cursor_bin")
                if cursor_bin is None:
                    continue
                cursor_bin = int(cursor_bin)
                # Top-K candidates → absolute bins, skip STOP.
                cands: list[tuple[int, float, float]] = []
                for i in range(1, 10):
                    bin_key, prob_key = f"top{i}_bin", f"top{i}_prob"
                    if bin_key not in d or prob_key not in d:
                        break
                    rel = int(d[bin_key])
                    if rel == b_pred:
                        continue
                    prob = float(d[prob_key])
                    cands.append((cursor_bin + rel, prob, prob))
                if not cands:
                    continue
                # Chosen = top1. Overwrite if multiple rows share a cursor
                # (early runs with tiny hop_on_stop can double-visit).
                chosen = cands[0][0]
                out[cursor_bin] = (chosen, cands)
    except OSError as exc:
        print(f"[viewer] could not read steps log: {exc}", file=sys.stderr)
        return None
    return out or None


# ─────────────────────────── main viewer ──────────────────────────────

class Viewer:
    def __init__(
        self,
        chart: Chart,
        audio_path: Path | None,
        *,
        steps_log_path: Path | None = None,
        compute_mel: bool = True,
    ):
        import pygame
        self.pygame = pygame
        self.chart = chart
        self.onsets = list(chart.track.onsets)
        self.audio_path = audio_path

        pygame.init()
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        self.w, self.h = WIDTH, HEIGHT
        self.screen = pygame.display.set_mode(
            (self.w, self.h), pygame.RESIZABLE,
        )
        pygame.display.set_caption(
            f"taiko2 viewer — {chart.track.artist} — "
            f"{chart.track.title} [{chart.track.difficulty.version}]"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 13)
        self.font_big = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 11)
        self.font_title = pygame.font.SysFont(
            "meiryoui,meiryo,yugothicui,yugothic,msgothic,consolas",
            16, bold=True,
        )

        # ── audio ─────────────────────────────────────────────────────
        self.has_audio = False
        self.volume = 0.35
        if audio_path is not None and audio_path.exists():
            try:
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.set_volume(self.volume)
                self.has_audio = True
            except pygame.error:
                self.has_audio = False

        # ── playback state ────────────────────────────────────────────
        self.playing = False
        self.paused_at = 0
        self.play_start_ticks = 0
        self.now_ms = 0
        self.next_hit = 0
        self.zoom = 1.0
        self.muted = False
        self.tick_enabled = True
        self.tick_volume = 0.95

        # ── toggles (all on by default, like taiko1) ──────────────────
        self.show_help = False
        self.show_stats = False
        self.show_minimap = True
        self.show_density = True
        self.show_mel = compute_mel
        self.show_ghosts = False   # enabled only when a log is loaded

        # ── precomputed chart metrics + end time ──────────────────────
        self.metrics = chart.calculate_metrics()
        last_ms = int(self.onsets[-1].time_ms) if self.onsets else 0
        self.song_end_ms = max(last_ms + 3000, 5000)

        # ── ticks ─────────────────────────────────────────────────────
        self._ticks: dict[OnsetKind, object] = {}
        mixer_freq, _, mixer_ch = pygame.mixer.get_init() or (44100, -16, 2)
        for kind, (dur, amp) in TICK_VOICE.items():
            self._ticks[kind] = _synth_tick(
                duration_ms=dur, amp=amp,
                mixer_freq=mixer_freq, mixer_channels=mixer_ch,
                volume=self.tick_volume,
            )
        self._sorted_onsets = sorted(self.onsets, key=lambda o: o.time_ms)
        self._next_onset_i = 0
        self._last_frame_ms = 0

        # ── hit flash (recent passes through HIT_X) ───────────────────
        self.recent_hits: list[tuple[int, OnsetKind]] = []

        # ── mel ───────────────────────────────────────────────────────
        self.mel_data: Any = None
        self._mel_global_min = 0.0
        self._mel_global_max = 1.0
        if compute_mel and chart.audio is not None:
            got = _compute_mel(chart)
            if got is not None:
                self.mel_data, _ = got
                self._mel_global_min = float(self.mel_data.min())
                self._mel_global_max = float(self.mel_data.max())
                print(f"[viewer] mel loaded: {self.mel_data.shape}")

        # ── ghost candidate log ───────────────────────────────────────
        self._cand_by_cursor: dict[
            int, tuple[int, list[tuple[int, float, float]]]
        ] = {}
        if steps_log_path is not None:
            parsed = _parse_steps_log(steps_log_path)
            if parsed:
                self._cand_by_cursor = parsed
                self.show_ghosts = True
                print(
                    f"[viewer] loaded {len(parsed):,} ghost-candidate "
                    f"cursors from {steps_log_path}"
                )

        # ── density surface ───────────────────────────────────────────
        self._density_surface = self._precompute_density_surface()

        # ── start playback ────────────────────────────────────────────
        self._start_playback(0)

    # ── density precompute ────────────────────────────────────────────

    def _precompute_density_surface(self):
        import pygame
        timeline = self.metrics.density_timeline
        if not timeline:
            return None
        max_d = max(timeline) if timeline else 1
        w = max(len(timeline), 1)
        h = 50
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        for i, d in enumerate(timeline):
            bar_h = int((d / max(max_d, 1)) * (h - 2))
            color = (*ACCENT, 160)
            pygame.draw.line(surf, color, (i, h), (i, h - bar_h), 1)
        return surf

    # ── playback helpers ──────────────────────────────────────────────

    def _start_playback(self, from_ms: int) -> None:
        self.now_ms = from_ms
        self.paused_at = from_ms
        self.playing = True
        self.play_start_ticks = self.pygame.time.get_ticks() - int(from_ms)
        # Tick pointer
        self._last_frame_ms = from_ms
        self._next_onset_i = _bisect_onsets_ge(self._sorted_onsets, from_ms)
        # Note-counter pointer for HUD
        self.next_hit = 0
        for i, o in enumerate(self._sorted_onsets):
            if o.time_ms > from_ms:
                self.next_hit = i
                break
        else:
            self.next_hit = len(self._sorted_onsets)
        # Hit flashes are cleared on any seek.
        self.recent_hits = []
        if self.has_audio:
            try:
                self.pygame.mixer.music.play(start=from_ms / 1000.0)
                self.pygame.mixer.music.set_volume(
                    0.0 if self.muted else self.volume,
                )
            except self.pygame.error:
                pass

    def _pause(self) -> None:
        self.playing = False
        self.paused_at = self.now_ms
        if self.has_audio:
            self.pygame.mixer.music.pause()

    def _resume(self) -> None:
        self.playing = True
        self.play_start_ticks = self.pygame.time.get_ticks() - int(self.paused_at)
        if self.has_audio:
            self.pygame.mixer.music.unpause()

    def _seek(self, delta_ms: int) -> None:
        target = max(0, min(self.song_end_ms, self.now_ms + delta_ms))
        was_playing = self.playing
        if self.has_audio:
            self.pygame.mixer.music.stop()
        self._start_playback(target)
        if not was_playing:
            self._pause()

    def _seek_absolute(self, target_ms: int) -> None:
        self._seek(target_ms - self.now_ms)

    def _toggle_mute(self) -> None:
        self.muted = not self.muted
        if self.has_audio:
            self.pygame.mixer.music.set_volume(
                0.0 if self.muted else self.volume,
            )

    def _set_volume(self, delta: float) -> None:
        self.volume = max(0.0, min(1.0, self.volume + delta))
        if self.has_audio and not self.muted:
            self.pygame.mixer.music.set_volume(self.volume)

    def _toggle_pause(self) -> None:
        if self.playing:
            self._pause()
        else:
            self._resume()

    def _trigger_ticks(self, now_ms: int) -> None:
        if not self.playing or not self.tick_enabled:
            self._last_frame_ms = now_ms
            return
        onsets = self._sorted_onsets
        i = self._next_onset_i
        n = len(onsets)
        while i < n and onsets[i].time_ms <= now_ms:
            sound = self._ticks.get(onsets[i].kind)
            if sound is not None:
                try:
                    sound.play()
                except self.pygame.error:
                    pass
            self.recent_hits.append((onsets[i].time_ms, onsets[i].kind))
            # Keep only the last few flash events.
            if len(self.recent_hits) > 8:
                self.recent_hits = self.recent_hits[-8:]
            i += 1
        self._next_onset_i = i
        self.next_hit = i
        self._last_frame_ms = now_ms

    # ── main frame ────────────────────────────────────────────────────

    def _update_now_ms(self) -> None:
        if self.playing:
            self.now_ms = self.pygame.time.get_ticks() - self.play_start_ticks
            if self.now_ms >= self.song_end_ms:
                self._pause()
                self.now_ms = self.song_end_ms

    def draw(self) -> None:
        self.screen.fill(BG_COLOR)

        self._draw_header()
        self._draw_playfield()
        self._draw_progress_bar()

        y_below = PLAYFIELD_TOP + PLAYFIELD_H + 30

        if self.show_mel and self.mel_data is not None:
            self._draw_mel_view(y_below)
            y_below += 14 + 90 + 6   # label + panel + gap

        if self.show_density and self._density_surface is not None:
            self._draw_density_graph(y_below)
            y_below += 65

        if self.show_minimap:
            self._draw_minimap(y_below)
            y_below += 35

        if self.show_stats:
            self._draw_stats_panel(y_below)

        if self.show_help:
            self._draw_help_overlay()

        self.pygame.display.flip()

    # ── header ────────────────────────────────────────────────────────

    def _draw_header(self) -> None:
        pygame = self.pygame
        t = self.chart.track
        title = f"{t.artist} — {t.title}  [{t.difficulty.version}]"
        if len(title) > 90:
            title = title[:87] + "..."
        surf = self.font_title.render(title, True, TEXT_COLOR)
        self.screen.blit(surf, (10, 8))

        time_str = f"{_format_time(self.now_ms)} / {_format_time(self.song_end_ms)}"
        t_surf = self.font_big.render(time_str, True, ACCENT)
        self.screen.blit(t_surf, (10, 32))

        # Status badges (right-to-left)
        x = self.w - 10
        badges: list[tuple[str, tuple[int, int, int]]] = []
        if not self.playing:
            badges.append(("PAUSED", (255, 200, 60)))
        if abs(self.zoom - 1.0) > 0.01:
            badges.append((f"ZOOM {self.zoom:.2f}x", ACCENT))
        vol_pct = 0 if self.muted else int(self.volume * 100)
        badges.append((
            f"VOL {vol_pct}%{' (M)' if self.muted else ''}", TEXT_COLOR,
        ))
        badges.append((
            f"TICKS {'ON' if self.tick_enabled else 'off'}", DIM_TEXT,
        ))
        if not self.has_audio:
            badges.append(("NO AUDIO", (255, 80, 80)))
        if self._cand_by_cursor:
            if self.show_ghosts:
                badges.append(("GHOSTS", (220, 170, 255)))
            else:
                badges.append(("ghosts off", DIM_TEXT))

        for text, color in reversed(badges):
            surf = self.font.render(text, True, color)
            w = surf.get_width() + 12
            x -= w + 4
            pygame.draw.rect(self.screen, PANEL_BG, (x, 8, w, 22),
                             border_radius=4)
            pygame.draw.rect(self.screen, PANEL_BORDER, (x, 8, w, 22),
                             1, border_radius=4)
            self.screen.blit(surf, (x + 6, 11))

        # Help hint
        hint = self.font_small.render(
            "H=Help  I=Stats  M=Map  D=Density  W=Mel  G=Ghosts  T=Ticks  N=Mute",
            True, DIM_TEXT,
        )
        self.screen.blit(hint, (self.w - hint.get_width() - 10, 36))

    # ── playfield ─────────────────────────────────────────────────────

    def _draw_playfield(self) -> None:
        pygame = self.pygame
        cy = PLAYFIELD_CENTER
        scroll = SCROLL_SPEED * self.zoom

        pygame.draw.rect(
            self.screen, PLAYFIELD_BG,
            (0, PLAYFIELD_TOP, self.w, PLAYFIELD_H), border_radius=4,
        )

        # Hit flash — brightest right after the most recent onset crossing.
        if self.recent_hits:
            last_t = self.recent_hits[-1][0]
            alpha = max(0, min(255, 200 - int((self.now_ms - last_t) * 0.7)))
            if alpha > 0:
                flash = pygame.Surface((40, PLAYFIELD_H))
                flash.fill((255, 255, 255))
                flash.set_alpha(alpha)
                self.screen.blit(flash, (HIT_X - 20, PLAYFIELD_TOP))

        # Hit line + drum circle.
        pygame.draw.line(
            self.screen, HIT_LINE_COLOR,
            (HIT_X, PLAYFIELD_TOP + 10),
            (HIT_X, PLAYFIELD_TOP + PLAYFIELD_H - 10), 3,
        )
        pygame.draw.circle(self.screen, (60, 60, 80), (HIT_X, cy), 30)
        pygame.draw.circle(self.screen, HIT_LINE_COLOR, (HIT_X, cy), 30, 2)

        # Real notes.
        self._draw_notes(self._sorted_onsets, cy, scroll)

        # Ghosts ABOVE the main notes (alternative candidates).
        if self.show_ghosts and self._cand_by_cursor:
            self._draw_ghost_candidates(PLAYFIELD_TOP + 20, scroll)

        # Note counter (passed / total) near the hit line.
        counter = self.font_small.render(
            f"{self.next_hit}/{len(self._sorted_onsets)}", True, DIM_TEXT,
        )
        self.screen.blit(
            counter,
            (HIT_X - counter.get_width() // 2,
             PLAYFIELD_TOP + PLAYFIELD_H - 18),
        )

    def _draw_notes(self, onsets, cy: int, scroll: float) -> None:
        pygame = self.pygame
        for o in onsets:
            x = HIT_X + (o.time_ms - self.now_ms) * scroll
            if x < -40:
                continue
            if x > self.w + 40:
                break
            base = COLORS.get(o.kind, (200, 200, 200))
            size = SIZES.get(o.kind, 16)

            # Dim notes that have passed the judgement ring.
            if x < HIT_X:
                factor = max(0.15, 1.0 - (HIT_X - x) / 200)
                color = tuple(int(c * factor) for c in base)
                border = tuple(int(180 * factor) for _ in range(3))
            else:
                color = base
                border = (255, 255, 255)

            pygame.draw.circle(self.screen, color, (int(x), cy), size)
            pygame.draw.circle(self.screen, border, (int(x), cy), size, 2)

            # Inner circle makes don/ka visually distinct beyond hue.
            if o.kind in (OnsetKind.DON, OnsetKind.BIG_DON):
                inner = tuple(min(255, c + 40) for c in base)
                pygame.draw.circle(
                    self.screen, inner, (int(x), cy), max(4, size // 3),
                )

    def _find_next_prediction(self):
        """Return (cursor_bin, chosen_abs_bin, candidate_list) for the
        first prediction at or after the playhead. None if no
        candidates are loaded or the playhead is past the last one."""
        if not self._cand_by_cursor:
            return None, None, None
        cursor_bin = int(self.now_ms / MEL_BIN_MS)
        best_cursor = None
        best_dist = float("inf")
        for pred_cursor in self._cand_by_cursor:
            dist = pred_cursor - cursor_bin
            if 0 <= dist < best_dist:
                best_dist = dist
                best_cursor = pred_cursor
        if best_cursor is None:
            return None, None, None
        chosen, cands = self._cand_by_cursor[best_cursor]
        return best_cursor, chosen, cands

    def _draw_ghost_candidates(self, cy: int, scroll: float) -> None:
        pygame = self.pygame
        pred_cursor, chosen, cands = self._find_next_prediction()
        if cands is None or not cands:
            return
        total = sum(c[2] for c in cands) or 1.0
        chosen_ms = chosen * MEL_BIN_MS
        chosen_x = HIT_X + (chosen_ms - self.now_ms) * scroll

        for abs_bin, _raw, final in cands:
            cand_ms = abs_bin * MEL_BIN_MS
            x = HIT_X + (cand_ms - self.now_ms) * scroll
            if abs(x - chosen_x) < 3:
                continue          # chosen is drawn as the real note
            if x < -40 or x > self.w + 40:
                continue
            pct = final / total
            size = max(4, int(12 * pct * len(cands)))
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*GHOST_COLOR, 200), (size, size), size)
            pygame.draw.circle(surf, (*GHOST_HIGHLIGHT, 220),
                               (size, size), size, 1)
            self.screen.blit(surf, (int(x) - size, cy - size))

    # ── progress bar ──────────────────────────────────────────────────

    def _draw_progress_bar(self) -> None:
        pygame = self.pygame
        y = PLAYFIELD_TOP + PLAYFIELD_H + 5
        bar_x, bar_w, bar_h = 10, self.w - 20, 16
        pygame.draw.rect(self.screen, PROGRESS_BG,
                         (bar_x, y, bar_w, bar_h), border_radius=3)
        if self.song_end_ms > 0:
            frac = max(0, min(1, self.now_ms / self.song_end_ms))
            fill_w = int(frac * bar_w)
            if fill_w > 0:
                pygame.draw.rect(
                    self.screen, PROGRESS_FILL,
                    (bar_x, y, fill_w, bar_h), border_radius=3,
                )
            # 30-second ticks.
            for sec in range(30, int(self.song_end_ms / 1000) + 1, 30):
                tx = bar_x + int((sec * 1000 / self.song_end_ms) * bar_w)
                pygame.draw.line(
                    self.screen, (100, 100, 120),
                    (tx, y), (tx, y + bar_h), 1,
                )
            cx = bar_x + fill_w
            pygame.draw.rect(
                self.screen, PROGRESS_CURSOR,
                (cx - 1, y - 2, 3, bar_h + 4), border_radius=1,
            )

    # ── mel view ──────────────────────────────────────────────────────

    def _draw_mel_view(self, y_start: int) -> None:
        import numpy as np
        pygame = self.pygame
        if self.mel_data is None:
            return
        pad_x = 10
        view_w = self.w - 20
        mel_h = 90
        px_per_frame = SCROLL_SPEED * self.zoom * MEL_BIN_MS
        cursor_frame = int(self.now_ms / MEL_BIN_MS)
        frames_left = int((HIT_X - pad_x) / max(px_per_frame, 0.001))
        frames_right = int((view_w - HIT_X + pad_x) / max(px_per_frame, 0.001))
        frame_start = cursor_frame - frames_left
        frame_end = cursor_frame + frames_right

        label = self.font_small.render(
            "Mel Spectrogram (W to toggle)", True, DIM_TEXT,
        )
        self.screen.blit(label, (pad_x, y_start))
        y = y_start + 14

        pygame.draw.rect(self.screen, PANEL_BG,
                         (pad_x, y, view_w, mel_h), border_radius=3)

        cmap = np.array(_get_mel_colormap(), dtype=np.uint8)
        n_mels, total = self.mel_data.shape
        f0 = max(0, frame_start)
        f1 = min(total, frame_end)
        if f1 > f0:
            mel_slice = self.mel_data[:, f0:f1]
            mel_range = max(self._mel_global_max - self._mel_global_min, 1e-6)
            mel_norm = np.clip(
                (mel_slice - self._mel_global_min) / mel_range * 255, 0, 255,
            ).astype(np.uint8)
            mel_norm = mel_norm[::-1, :]                    # low freq at bottom
            mel_rgb = cmap[mel_norm]                        # (n_mels, n_vis, 3)
            mel_rgb_t = mel_rgb.transpose(1, 0, 2)          # (n_vis, n_mels, 3)
            mel_surf = pygame.surfarray.make_surface(mel_rgb_t)
            slice_x = int((f0 - frame_start) * px_per_frame)
            slice_w = max(1, int((f1 - f0) * px_per_frame))
            scaled = pygame.transform.scale(mel_surf, (slice_w, mel_h))
            self.screen.blit(scaled, (pad_x + slice_x, y))

        # Cursor line.
        cx = pad_x + int(frames_left * px_per_frame)
        pygame.draw.line(
            self.screen, (255, 255, 255), (cx, y), (cx, y + mel_h), 1,
        )

        # Onset markers along the bottom edge.
        for o in self._sorted_onsets:
            f = int(o.time_ms / MEL_BIN_MS)
            if frame_start <= f <= frame_end:
                ox = pad_x + int((f - frame_start) * px_per_frame)
                c = COLORS.get(o.kind, (200, 200, 200))
                pygame.draw.line(self.screen, c,
                                 (ox, y + mel_h - 5), (ox, y + mel_h), 2)

        # Second-labelled time grid.
        sec_start = max(0, int(frame_start * MEL_BIN_MS / 1000))
        sec_end = int(frame_end * MEL_BIN_MS / 1000) + 1
        for sec in range(sec_start, sec_end):
            f = int(sec * 1000 / MEL_BIN_MS)
            if frame_start <= f <= frame_end:
                lx = pad_x + int((f - frame_start) * px_per_frame)
                lbl = self.font_small.render(f"{sec}s", True, (200, 200, 200))
                self.screen.blit(lbl, (lx + 2, y + 1))
                pygame.draw.line(self.screen, (80, 80, 100),
                                 (lx, y), (lx, y + mel_h), 1)

    # ── density graph ─────────────────────────────────────────────────

    def _draw_density_graph(self, y_start: int) -> None:
        pygame = self.pygame
        if self._density_surface is None:
            return
        label = self.font_small.render(
            "Density (events/sec)", True, DIM_TEXT,
        )
        self.screen.blit(label, (10, y_start))
        graph_y = y_start + 14
        graph_w = self.w - 20
        graph_h = 45
        pygame.draw.rect(
            self.screen, PANEL_BG,
            (10, graph_y, graph_w, graph_h), border_radius=3,
        )
        scaled = pygame.transform.smoothscale(
            self._density_surface, (graph_w, graph_h),
        )
        self.screen.blit(scaled, (10, graph_y))
        if self.song_end_ms > 0:
            frac = max(0, min(1, self.now_ms / self.song_end_ms))
            cx = 10 + int(frac * graph_w)
            pygame.draw.line(
                self.screen, (255, 255, 255),
                (cx, graph_y), (cx, graph_y + graph_h), 1,
            )
        info = self.font_small.render(
            f"mean={self.metrics.density_mean:.1f}/s  "
            f"peak={self.metrics.density_peak}/s  "
            f"std={self.metrics.density_std:.1f}",
            True, DIM_TEXT,
        )
        self.screen.blit(info, (self.w - info.get_width() - 14, y_start))

    # ── minimap ───────────────────────────────────────────────────────

    def _draw_minimap(self, y_start: int) -> None:
        pygame = self.pygame
        label = self.font_small.render("Timeline", True, DIM_TEXT)
        self.screen.blit(label, (10, y_start))
        map_y = y_start + 14
        map_w = self.w - 20
        map_h = 16
        pygame.draw.rect(self.screen, PANEL_BG,
                         (10, map_y, map_w, map_h), border_radius=2)
        if self.song_end_ms > 0:
            for o in self._sorted_onsets:
                frac = o.time_ms / self.song_end_ms
                x = 10 + int(frac * map_w)
                color = COLORS.get(o.kind, (150, 150, 150))
                color = tuple(min(255, c + 30) for c in color)
                pygame.draw.line(
                    self.screen, color,
                    (x, map_y + 2), (x, map_y + map_h - 2), 1,
                )
            frac = max(0, min(1, self.now_ms / self.song_end_ms))
            cx = 10 + int(frac * map_w)
            pygame.draw.rect(
                self.screen, (255, 255, 255),
                (cx - 1, map_y - 1, 3, map_h + 2), border_radius=1,
            )

    # ── stats panel ───────────────────────────────────────────────────

    def _compute_local_metronome(self):
        """Return (summary_str, color) characterizing the metronome grip
        of the last ~2 s of onsets; None if not enough data."""
        cursor = self.now_ms
        recent = [o.time_ms for o in self._sorted_onsets
                  if cursor - 2000 <= o.time_ms <= cursor]
        if len(recent) < 4:
            return None
        gaps = [recent[i] - recent[i - 1] for i in range(1, len(recent))
                if recent[i] > recent[i - 1]]
        if len(gaps) < 3:
            return None
        # Greedy 5%-tolerance clustering, same as taiko1.
        sorted_gaps = sorted(gaps)
        clusters: list[tuple[float, int]] = []
        cur: list[int] = [sorted_gaps[0]]
        for g in sorted_gaps[1:]:
            centroid = sum(cur) / len(cur)
            if centroid > 0 and abs(g - centroid) / centroid <= 0.05:
                cur.append(g)
            else:
                clusters.append((sum(cur) / len(cur), len(cur)))
                cur = [g]
        clusters.append((sum(cur) / len(cur), len(cur)))
        clusters.sort(key=lambda c: c[1], reverse=True)
        dominant_gap, dominant_count = clusters[0]
        total = len(gaps)
        pct = dominant_count / total * 100
        bpm = 60000 / dominant_gap if dominant_gap > 0 else 0
        if pct >= 80:
            color = (255, 80, 80)
        elif pct >= 50:
            color = (255, 200, 60)
        else:
            color = (100, 220, 100)
        return (
            f"{pct:.0f}% in peak ({dominant_gap:.0f}ms / {bpm:.0f}BPM)  "
            f"{len(gaps)} gaps  {len(clusters)} clusters",
            color,
        )

    def _draw_stats_panel(self, y_start: int) -> None:
        pygame = self.pygame
        m = self.metrics
        panel_x = 10
        panel_w = self.w - 20
        panel_h = self.h - y_start - 10
        if panel_h < 40:
            return
        pygame.draw.rect(self.screen, PANEL_BG,
                         (panel_x, y_start, panel_w, panel_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         (panel_x, y_start, panel_w, panel_h), 1, border_radius=4)

        col_x = panel_x + 12
        y = y_start + 8

        def text(label: str, value: str, color=TEXT_COLOR, lcolor=DIM_TEXT):
            nonlocal y
            if y > y_start + panel_h - 18:
                return
            lsurf = self.font_small.render(label, True, lcolor)
            vsurf = self.font.render(value, True, color)
            self.screen.blit(lsurf, (col_x, y))
            self.screen.blit(vsurf, (col_x + lsurf.get_width() + 6, y - 1))
            y += 17

        def section(title: str):
            nonlocal y
            if y > y_start + panel_h - 18:
                return
            surf = self.font_big.render(title, True, ACCENT)
            self.screen.blit(surf, (col_x, y))
            y += 20

        section("Level Stats")
        text("Events:", f"{m.total_events:,}")
        text("Duration:", f"{m.duration_s:.1f}s ({_format_time(int(m.duration_s * 1000))})")

        # Per-kind breakdown
        type_parts: list[str] = []
        for kind, count in (
            ("don",      m.count_don),
            ("ka",       m.count_ka),
            ("big_don",  m.count_big_don),
            ("big_ka",   m.count_big_ka),
            ("drumroll", m.count_drumroll),
            ("spinner",  m.count_spinner),
        ):
            if count > 0:
                pct = count / max(m.total_events, 1) * 100
                type_parts.append(f"{kind}={count} ({pct:.0f}%)")
        text("Types:", "  ".join(type_parts[:3]))
        if len(type_parts) > 3:
            text("", "  ".join(type_parts[3:]))

        text("Density:",
             f"mean={m.density_mean:.1f}/s  peak={m.density_peak}/s  "
             f"std={m.density_std:.1f}")
        if m.estimated_bpm:
            text("Est. BPM:", f"{m.estimated_bpm:.0f}")
        text("IOI:",
             f"avg={m.ioi_mean_ms:.0f}ms  med={m.ioi_median_ms:.0f}ms  "
             f"min={m.ioi_min_ms:.0f}ms  max={m.ioi_max_ms:.0f}ms  "
             f"p95={m.ioi_p95_ms:.0f}  p99={m.ioi_p99_ms:.0f}")
        text("Short IOI:",
             f"{m.short_ioi_count} (<20ms, {m.short_ioi_pct:.2f}%)")
        text("Long gaps (>2s):", f"{m.long_gap_count}")

        # Local density around cursor
        local = sum(
            1 for o in self._sorted_onsets
            if abs(o.time_ms - self.now_ms) < 500
        )
        text("Local density:", f"{local} events/s (around cursor)")

        met = self._compute_local_metronome()
        if met is not None:
            msg, color = met
            text("Metronome:", msg, color=color)

        # Move to second column if we still have room.
        col2_x = panel_x + panel_w // 2
        if y_start + 8 + 17 * 8 < y_start + panel_h - 20:
            col_x_save = col_x
            y_save = y
            col_x = col2_x
            y = y_start + 8
            section("Streaks")
            text("Longest streak:", f"{m.longest_streak}")
            text("Mean streak len:", f"{m.mean_streak_len:.2f}")
            text("Streak event frac:", f"{m.streak_event_fraction:.2%}")

            section("Density detail")
            text("Min/sec:", f"{m.density_min}")
            text("CV:", f"{m.density_cv:.2f}")
            text("Silence regions:", f"{len(m.silence_regions)} (>=2 s @ 0 ev/s)")
            text("Dense regions:", f"{len(m.dense_regions)} (>2x mean for >=3 s)")

            section("Pattern")
            text("Don ratio:", f"{m.don_ratio:.2f}")
            text("Events/sec:", f"{m.events_per_sec:.2f}")
            text("IOI std:", f"{m.ioi_std_ms:.0f} ms")
            if m.dominant_ioi_ms is not None:
                text("Dominant IOI:", f"{m.dominant_ioi_ms:.0f} ms")
            text("Over-pspace (self):", f"{m.over_pspace_self:.2f}")
            col_x = col_x_save
            y = max(y, y_save)

    # ── help overlay ──────────────────────────────────────────────────

    def _draw_help_overlay(self) -> None:
        pygame = self.pygame
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        lines = [
            ("CONTROLS", None),
            ("", ""),
            ("Space",            "Pause / Resume"),
            ("Left / Right",     "Seek -5s / +5s"),
            ("Shift+Left/Right", "Seek -1s / +1s"),
            ("Up / Down",        "Volume up / down"),
            ("+ / -",            "Zoom in / out"),
            ("Scroll wheel",     "Zoom"),
            ("Click progress bar", "Seek"),
            ("R",                "Restart"),
            ("T",                "Toggle tick synth"),
            ("N",                "Mute music"),
            ("W",                "Toggle mel spectrogram"),
            ("D",                "Toggle density graph"),
            ("M",                "Toggle minimap"),
            ("I",                "Toggle stats panel"),
            ("G",                "Toggle ghost candidates"),
            ("H",                "Toggle this help"),
            ("Esc / Q",          "Quit"),
        ]
        y = 80
        cx = self.w // 2
        for key, desc in lines:
            if desc is None:
                surf = self.font_title.render(key, True, ACCENT)
                self.screen.blit(surf, (cx - surf.get_width() // 2, y))
                y += 28
            elif key == "":
                y += 8
            else:
                ksurf = self.font_big.render(key, True, (255, 255, 255))
                dsurf = self.font.render(desc, True, TEXT_COLOR)
                self.screen.blit(ksurf, (cx - 220, y))
                self.screen.blit(dsurf, (cx - 80, y + 2))
                y += 22

    # ── event loop ────────────────────────────────────────────────────

    def handle_events(self) -> bool:
        pygame = self.pygame
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.VIDEORESIZE:
                self.w, self.h = ev.w, ev.h
                self.screen = pygame.display.set_mode(
                    (self.w, self.h), pygame.RESIZABLE,
                )
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                bar_y = PLAYFIELD_TOP + PLAYFIELD_H + 5
                if bar_y <= my <= bar_y + 20 and self.song_end_ms > 0:
                    frac = max(0, min(1, (mx - 10) / max(self.w - 20, 1)))
                    self._seek_absolute(int(frac * self.song_end_ms))
            if ev.type == pygame.MOUSEWHEEL:
                factor = 1.15 if ev.y > 0 else 1 / 1.15
                self.zoom = max(0.1, min(10.0, self.zoom * factor))
            if ev.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                shift = mods & pygame.KMOD_SHIFT
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif ev.key == pygame.K_SPACE:
                    self._toggle_pause()
                elif ev.key == pygame.K_LEFT:
                    self._seek(-1000 if shift else -5000)
                elif ev.key == pygame.K_RIGHT:
                    self._seek(1000 if shift else 5000)
                elif ev.key == pygame.K_UP:
                    self._set_volume(+0.05)
                elif ev.key == pygame.K_DOWN:
                    self._set_volume(-0.05)
                elif ev.key in (
                    pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
                ):
                    self.zoom = min(10.0, self.zoom * 1.25)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.zoom = max(0.1, self.zoom / 1.25)
                elif ev.key == pygame.K_r:
                    self._start_playback(0)
                elif ev.key == pygame.K_t:
                    self.tick_enabled = not self.tick_enabled
                elif ev.key == pygame.K_n:
                    self._toggle_mute()
                elif ev.key == pygame.K_w:
                    if self.mel_data is not None:
                        self.show_mel = not self.show_mel
                elif ev.key == pygame.K_d:
                    self.show_density = not self.show_density
                elif ev.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                elif ev.key == pygame.K_i:
                    self.show_stats = not self.show_stats
                elif ev.key == pygame.K_g:
                    if self._cand_by_cursor:
                        self.show_ghosts = not self.show_ghosts
                elif ev.key == pygame.K_h:
                    self.show_help = not self.show_help
        return True

    def run(self) -> None:
        pygame = self.pygame
        running = True
        while running:
            running = self.handle_events()
            self._update_now_ms()
            self._trigger_ticks(self.now_ms)
            self.draw()
            self.clock.tick(FPS)
        pygame.mixer.music.stop()
        pygame.quit()


# ─────────────────────────── CLI ─────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Taiko2 game-view chart viewer.")
    p.add_argument(
        "path", nargs="?", default=None,
        help="Path to .osu / .osz / .zip. If omitted, opens a file dialog.",
    )
    p.add_argument(
        "--index", type=int, default=None,
        help="Chart index inside an .osz (skips the picker).",
    )
    p.add_argument(
        "--steps-log", type=Path, default=None,
        help=(
            "Path to a .steps.jsonl written by `cli.infer --debug`. "
            "Enables ghost-candidate rendering (toggle with G). If "
            "omitted, auto-searches for a sibling `{chart_stem}.steps.jsonl`."
        ),
    )
    p.add_argument(
        "--no-mel", action="store_true",
        help="Skip mel computation entirely (avoid ~1-3s startup cost).",
    )
    return p.parse_args(argv)


def _auto_steps_log(chart_path: Path) -> Path | None:
    candidate = chart_path.with_suffix(".steps.jsonl")
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.path).resolve() if args.path else _pick_file()
    if path is None:
        print("No file selected.", file=sys.stderr)
        return 2
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2

    suffix = path.suffix.lower()
    tmp_audio: Path | None = None

    try:
        if suffix == ".osz":
            index = args.index
            if index is None:
                index = _pick_osz_index(path)
                if index is None:
                    print("Cancelled.", file=sys.stderr)
                    return 0
            chart = Chart.load(path, index=index)
        else:
            chart = Chart.load(path)

        tmp_audio = _write_audio_tmp(chart)
        steps_log = args.steps_log or _auto_steps_log(path)
        Viewer(
            chart, tmp_audio,
            steps_log_path=steps_log,
            compute_mel=not args.no_mel,
        ).run()
        return 0
    finally:
        if tmp_audio is not None:
            try:
                tmp_audio.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
