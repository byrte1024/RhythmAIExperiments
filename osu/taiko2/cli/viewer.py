"""osu!taiko game-view viewer — plays a Chart against its audio.

Tick (T to toggle) plays a short white-noise burst when each onset
crosses the judgement ring. Per-kind shaping only — duration and
amplitude vary by don/ka/big — no synth/sine component. Tick volume
sits above the music so beats are clearly audible.

Loads:
  - ``.osu``                     — single taiko chart. No audio unless
                                    paired externally; playback silent.
  - ``.osz``                     — shows an index picker for the charts
                                    inside, plays with the embedded audio.
  - taiko2 bundle (``.zip``)     — direct round-trip of Chart.save output.

Usage::

    python -m osu.taiko2.cli.viewer                     # opens a file dialog
    python -m osu.taiko2.cli.viewer path/to/chart.osu
    python -m osu.taiko2.cli.viewer path/to/pack.osz    # asks for index
    python -m osu.taiko2.cli.viewer path/to/pack.osz --index 2

Controls:
  Space        — pause / resume
  Left/Right   — seek -5 s / +5 s   (Shift = ±1 s)
  Up/Down      — volume up / down
  R            — restart
  T            — toggle tick synth
  Esc / Q      — quit
"""
from __future__ import annotations

import argparse
import array
import os
import random
import sys
import tempfile
from pathlib import Path

from ..domain.beatmap import OnsetKind
from ..domain.chart import Chart
from ..parsing.osz import load_pack

# ───────────────────────────── layout ────────────────────────────────
WIDTH, HEIGHT = 1200, 700
PLAYFIELD_TOP = 220
PLAYFIELD_H = 180
PLAYFIELD_CENTER = PLAYFIELD_TOP + PLAYFIELD_H // 2
HIT_X = 160                       # x-position of the judgement ring
SCROLL_SPEED = 0.5                # pixels per millisecond
FPS = 120

# ───────────────────────────── colors ────────────────────────────────
BG = (22, 22, 30)
PLAYFIELD_BG = (30, 30, 42)
HIT_LINE = (255, 255, 255)
TEXT = (210, 210, 220)
DIM = (120, 120, 135)
ACCENT = (100, 140, 255)
PANEL = (28, 28, 38)
PANEL_BORDER = (55, 55, 75)
PROGRESS_BG = (40, 40, 55)
PROGRESS_FILL = (80, 120, 220)

KIND_COLORS: dict[OnsetKind, tuple[int, int, int]] = {
    OnsetKind.DON:      (235,  69,  44),
    OnsetKind.KA:       ( 68, 141, 199),
    OnsetKind.BIG_DON:  (255,  90,  60),
    OnsetKind.BIG_KA:   ( 80, 165, 230),
    OnsetKind.DRUMROLL: (252, 183,  30),
    OnsetKind.SPINNER:  (100, 200, 100),
    OnsetKind.UNKNOWN:  (160, 160, 160),
}
KIND_RADIUS: dict[OnsetKind, int] = {
    OnsetKind.DON:      22,
    OnsetKind.KA:       22,
    OnsetKind.BIG_DON:  34,
    OnsetKind.BIG_KA:   34,
    OnsetKind.DRUMROLL: 18,
    OnsetKind.SPINNER:  28,
    OnsetKind.UNKNOWN:  18,
}

# Per-kind white-noise shaping: `(duration_ms, amp)`. Big variants are
# longer + louder; ka is a bit shorter than don so the timbre still reads
# even without pitch. Pure noise — no sine, no synth.
TICK_VOICE: dict[OnsetKind, tuple[int, float]] = {
    OnsetKind.DON:      (50, 1.00),
    OnsetKind.KA:       (30, 0.85),
    OnsetKind.BIG_DON:  (65, 1.30),
    OnsetKind.BIG_KA:   (45, 1.15),
    OnsetKind.DRUMROLL: (20, 0.60),
    OnsetKind.SPINNER:  (25, 0.60),
    OnsetKind.UNKNOWN:  (25, 0.50),
}


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
    """Simple pygame-window picker. Returns the selected index or None if
    the user closes the window.
    """
    pack = load_pack(pack_path)
    if pack is None or not pack.tracks:
        raise ValueError(f"{pack_path} contains no taiko charts")
    if len(pack.tracks) == 1:
        return 0

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((720, 520))
    pygame.display.set_caption(f"Select chart — {pack_path.name}")
    font = pygame.font.SysFont("Segoe UI", 18)
    small = pygame.font.SysFont("Segoe UI", 14)
    title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    clock = pygame.time.Clock()

    hovered = 0
    scroll = 0

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.display.quit()
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.display.quit()
                    return None
                if ev.key == pygame.K_UP:
                    hovered = (hovered - 1) % len(pack.tracks)
                elif ev.key == pygame.K_DOWN:
                    hovered = (hovered + 1) % len(pack.tracks)
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    pygame.display.quit()
                    return hovered
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                idx = (my - 80 + scroll) // 44
                if 0 <= idx < len(pack.tracks):
                    pygame.display.quit()
                    return int(idx)
            if ev.type == pygame.MOUSEMOTION:
                idx = (ev.pos[1] - 80 + scroll) // 44
                if 0 <= idx < len(pack.tracks):
                    hovered = int(idx)
            if ev.type == pygame.MOUSEWHEEL:
                scroll = max(0, scroll - ev.y * 44)

        screen.fill(BG)
        head = pack.tracks[0]
        title_surf = title_font.render(
            f"{head.artist} — {head.title}"[:60], True, TEXT,
        )
        screen.blit(title_surf, (24, 24))
        screen.blit(small.render(
            f"{len(pack.tracks)} charts · click / ↑↓ + Enter",
            True, DIM,
        ), (24, 56))

        y0 = 80 - scroll
        for i, t in enumerate(pack.tracks):
            row_top = y0 + i * 44
            if row_top < 72 or row_top > screen.get_height() - 12:
                continue
            rect = (12, row_top, screen.get_width() - 24, 40)
            bg = PANEL if i != hovered else (48, 52, 80)
            pygame.draw.rect(screen, bg, rect, border_radius=6)
            pygame.draw.rect(screen, PANEL_BORDER, rect, width=1, border_radius=6)
            label = f"[{i}] {t.difficulty.version}  ·  OD {t.difficulty.overall_difficulty:g}  ·  {t.density.total_events} events"
            star = t.difficulty.star_rating
            if star is not None:
                label += f"  ·  {star:.2f}★"
            screen.blit(font.render(label[:120], True, TEXT), (24, row_top + 10))

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


# ─────────────────────────── tick synth ──────────────────────────────

def _bisect_onsets_ge(onsets, t_ms: int) -> int:
    """First index `i` in `onsets` where `onsets[i].time_ms >= t_ms`."""
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
    """Generate a short white-noise burst with square-root fade-out.

    No sine, no synth — pure uniform noise through an amplitude envelope.
    Noise is seeded deterministically so the sample bytes are identical
    across sessions and don't need re-randomization.

    Pre-synthesized once per kind at viewer init; `.play()`'d directly on
    each hit.
    """
    import pygame

    n = max(1, int(mixer_freq * duration_ms / 1000))
    rng = random.Random(duration_ms * 1000 + int(amp * 1000))
    buf = array.array("h")
    peak = 32767
    for i in range(n):
        fade = 1.0 - (i / n) ** 0.5               # fast decay
        noise = rng.random() * 2 - 1
        val = int(volume * amp * peak * fade * noise)
        val = max(-peak, min(peak, val))
        for _ in range(mixer_channels):
            buf.append(val)
    return pygame.mixer.Sound(buffer=buf)


# ─────────────────────────── game viewer ─────────────────────────────

class Viewer:
    def __init__(self, chart: Chart, audio_path: Path | None):
        import pygame
        self.pygame = pygame
        self.chart = chart
        self.onsets = chart.track.onsets
        self.audio_path = audio_path

        self.volume = 0.7
        self.paused = False
        self.tick_enabled = True
        # Ticks sit ABOVE the music so onsets cut through clearly.
        self.tick_volume = 0.95

        pygame.init()
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(
            f"taiko2 viewer — {chart.track.artist} — {chart.track.title} [{chart.track.difficulty.version}]"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 16)
        self.font_small = pygame.font.SysFont("Segoe UI", 13)
        self.font_big = pygame.font.SysFont("Segoe UI", 22, bold=True)

        self._audio_ok = False
        if audio_path is not None and audio_path.exists():
            try:
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play()
                self._audio_ok = True
            except pygame.error:
                self._audio_ok = False

        self.start_tick = pygame.time.get_ticks()
        self.offset_ms = 0              # current seek offset relative to start
        self.last_paused_at_ms = 0
        self._last_frame_ms = 0         # for crossing-based tick triggering
        self._next_onset_i = 0          # next onset to watch, sorted by time

        # Precompute per-kind tick sounds. One pygame.mixer.Sound each —
        # replayed via play() on each hit without re-synthesizing.
        self._ticks: dict[OnsetKind, object] = {}
        mixer_freq, _, mixer_channels = pygame.mixer.get_init() or (44100, -16, 2)
        for kind, (dur, amp) in TICK_VOICE.items():
            self._ticks[kind] = _synth_tick(
                duration_ms=dur, amp=amp,
                mixer_freq=mixer_freq, mixer_channels=mixer_channels,
                volume=self.tick_volume,
            )

        # Ensure onsets are time-sorted (they should be from parsing, but
        # don't trust the input silently).
        self._sorted_onsets = sorted(
            self.onsets, key=lambda o: o.time_ms,
        )

        # End time for progress/layout: either last onset + 3s or duration_s
        last_ms = (
            int(self.onsets[-1].time_ms) if self.onsets
            else int(chart.track.density.duration_s * 1000)
        )
        self.end_ms = max(last_ms + 3000, 5000)

        # Precomputed metrics panel data
        self.metrics = chart.calculate_metrics()

    # ── Time helpers ──────────────────────────────────────────────────

    def _now_ms(self) -> int:
        if self.paused:
            return self.last_paused_at_ms
        return self.pygame.time.get_ticks() - self.start_tick + self.offset_ms

    def _seek_relative(self, delta_ms: int) -> None:
        cur = self._now_ms()
        target = max(0, min(cur + delta_ms, self.end_ms))
        self._seek_absolute(target)

    def _seek_absolute(self, target_ms: int) -> None:
        target_ms = max(0, target_ms)
        self.offset_ms = target_ms
        self.start_tick = self.pygame.time.get_ticks()
        if self.paused:
            self.last_paused_at_ms = target_ms
        self._last_frame_ms = target_ms
        self._next_onset_i = _bisect_onsets_ge(self._sorted_onsets, target_ms)
        if self._audio_ok:
            try:
                self.pygame.mixer.music.play(start=target_ms / 1000.0)
                if self.paused:
                    self.pygame.mixer.music.pause()
            except self.pygame.error:
                pass

    def _toggle_pause(self) -> None:
        if self.paused:
            self.paused = False
            self.start_tick = self.pygame.time.get_ticks()
            self.offset_ms = self.last_paused_at_ms
            self._last_frame_ms = self.last_paused_at_ms
            if self._audio_ok:
                self.pygame.mixer.music.unpause()
        else:
            self.last_paused_at_ms = self._now_ms()
            self.paused = True
            if self._audio_ok:
                self.pygame.mixer.music.pause()

    def _set_volume(self, delta: float) -> None:
        self.volume = max(0.0, min(1.0, self.volume + delta))
        if self._audio_ok:
            self.pygame.mixer.music.set_volume(self.volume)

    def _restart(self) -> None:
        self._seek_absolute(0)

    def _toggle_ticks(self) -> None:
        self.tick_enabled = not self.tick_enabled

    def _trigger_ticks(self, now_ms: int) -> None:
        """Fire a tick sound for every onset whose time falls inside the
        interval `(self._last_frame_ms, now_ms]`. Safe across frames where
        multiple onsets are crossed; pauses and seeks reset the pointer so
        no burst fires after a jump.
        """
        if self.paused or not self.tick_enabled:
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
            i += 1
        self._next_onset_i = i
        self._last_frame_ms = now_ms

    # ── Drawing ───────────────────────────────────────────────────────

    def _draw_header(self) -> None:
        t = self.chart.track
        title = f"{t.artist} — {t.title}"
        sub = (
            f"[{t.difficulty.version}]  ·  OD {t.difficulty.overall_difficulty:g}"
        )
        if t.difficulty.star_rating is not None:
            sub += f"  ·  {t.difficulty.star_rating:.2f}★"
        sub += f"  ·  {len(self.onsets)} events  ·  {t.density.duration_s:.1f} s"

        self.screen.blit(self.font_big.render(title[:100], True, TEXT), (20, 16))
        self.screen.blit(self.font.render(sub[:120], True, DIM), (20, 50))

        # BPM + density on the right
        bpm_txt = (
            f"BPM: {self.metrics.estimated_bpm:g}"
            if self.metrics.estimated_bpm else "BPM: —"
        )
        right = (
            f"{bpm_txt}   "
            f"density {self.metrics.density_mean:.1f}/s   "
            f"peak {self.metrics.density_peak}"
        )
        surf = self.font.render(right, True, DIM)
        self.screen.blit(surf, (WIDTH - surf.get_width() - 20, 50))

    def _draw_playfield(self, now_ms: int) -> None:
        pf_rect = (0, PLAYFIELD_TOP, WIDTH, PLAYFIELD_H)
        self.pygame.draw.rect(self.screen, PLAYFIELD_BG, pf_rect)
        self.pygame.draw.line(
            self.screen, (40, 40, 55),
            (0, PLAYFIELD_CENTER), (WIDTH, PLAYFIELD_CENTER), 2,
        )

        # Judgement ring
        self.pygame.draw.circle(
            self.screen, (60, 60, 80), (HIT_X, PLAYFIELD_CENTER), 38, 3,
        )
        self.pygame.draw.line(
            self.screen, HIT_LINE,
            (HIT_X, PLAYFIELD_TOP + 10),
            (HIT_X, PLAYFIELD_TOP + PLAYFIELD_H - 10), 2,
        )

        # Draw onsets whose time is within the visible window
        window_ms_right = (WIDTH - HIT_X) / SCROLL_SPEED
        window_ms_left = HIT_X / SCROLL_SPEED
        t_lo = now_ms - window_ms_left
        t_hi = now_ms + window_ms_right
        for onset in self.onsets:
            ot = onset.time_ms
            if ot < t_lo or ot > t_hi:
                continue
            x = int(HIT_X + (ot - now_ms) * SCROLL_SPEED)
            color = KIND_COLORS.get(onset.kind, (200, 200, 200))
            r = KIND_RADIUS.get(onset.kind, 18)
            self.pygame.draw.circle(self.screen, color, (x, PLAYFIELD_CENTER), r)
            self.pygame.draw.circle(
                self.screen, (0, 0, 0), (x, PLAYFIELD_CENTER), r, 2,
            )

    def _draw_progress(self, now_ms: int) -> None:
        bar_y = HEIGHT - 58
        bar_h = 10
        self.pygame.draw.rect(
            self.screen, PROGRESS_BG, (20, bar_y, WIDTH - 40, bar_h),
            border_radius=4,
        )
        ratio = min(1.0, max(0.0, now_ms / max(self.end_ms, 1)))
        self.pygame.draw.rect(
            self.screen, PROGRESS_FILL,
            (20, bar_y, int((WIDTH - 40) * ratio), bar_h),
            border_radius=4,
        )

        def _fmt(ms: int) -> str:
            s = max(0, ms // 1000)
            return f"{s // 60:d}:{s % 60:02d}"

        t_text = f"{_fmt(now_ms)} / {_fmt(self.end_ms)}"
        tsurf = self.font.render(t_text, True, TEXT)
        self.screen.blit(tsurf, (20, bar_y + bar_h + 6))

        status = "PAUSED" if self.paused else "PLAYING"
        if not self._audio_ok:
            status += " (no audio)"
        status += f"   vol {int(self.volume * 100)}%"
        status += f"   ticks {'on' if self.tick_enabled else 'off'}"
        ssurf = self.font.render(status, True, DIM)
        self.screen.blit(ssurf, (WIDTH - ssurf.get_width() - 20, bar_y + bar_h + 6))

    def _draw_help(self) -> None:
        lines = [
            "Space: pause/resume",
            "←/→: seek ±5s (Shift = ±1s)",
            "↑/↓: volume",
            "R: restart",
            "T: toggle tick synth",
            "Esc/Q: quit",
        ]
        x, y = WIDTH - 230, 100
        for line in lines:
            surf = self.font_small.render(line, True, DIM)
            self.screen.blit(surf, (x, y))
            y += 18

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        pygame = self.pygame
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self._toggle_pause()
                    elif ev.key == pygame.K_LEFT:
                        self._seek_relative(-1000 if shift else -5000)
                    elif ev.key == pygame.K_RIGHT:
                        self._seek_relative(1000 if shift else 5000)
                    elif ev.key == pygame.K_UP:
                        self._set_volume(+0.05)
                    elif ev.key == pygame.K_DOWN:
                        self._set_volume(-0.05)
                    elif ev.key == pygame.K_r:
                        self._restart()
                    elif ev.key == pygame.K_t:
                        self._toggle_ticks()

            now_ms = self._now_ms()
            self._trigger_ticks(now_ms)
            self.screen.fill(BG)
            self._draw_header()
            self._draw_playfield(now_ms)
            self._draw_progress(now_ms)
            self._draw_help()
            pygame.display.flip()
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
    return p.parse_args(argv)


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
        elif suffix == ".osu":
            chart = Chart.load(path)
        else:
            chart = Chart.load(path)

        tmp_audio = _write_audio_tmp(chart)
        Viewer(chart, tmp_audio).run()
        return 0
    finally:
        if tmp_audio is not None:
            try:
                tmp_audio.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
