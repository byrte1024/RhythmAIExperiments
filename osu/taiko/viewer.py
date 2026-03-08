"""Simple osu!taiko onset visualizer. Pick a CSV, watch it play."""
import pygame
import glob
import os
import sys
import array
import math
import random

AUDIO_DIR = "./osu/taiko/audio"
DATA_DIR = "./osu/taiko/data"

WIDTH, HEIGHT = 1000, 300
HIT_X = 120
SCROLL_SPEED = 0.5  # pixels per ms
FPS = 120

COLORS = {
    "don": (235, 69, 44),
    "ka": (68, 141, 199),
    "big_don": (235, 69, 44),
    "big_ka": (68, 141, 199),
    "drumroll": (252, 183, 30),
    "spinner": (100, 200, 100),
}
SIZES = {
    "don": 18, "ka": 18,
    "big_don": 26, "big_ka": 26,
    "drumroll": 14, "spinner": 22,
}


def pick_csv():
    """Use CLI arg if provided, otherwise pick a random CSV."""
    if len(sys.argv) > 1:
        return sys.argv[1]
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
            if len(parts) == 2:
                onsets.append((int(parts[0]), parts[1]))
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
    sound = pygame.mixer.Sound(buffer=buf)
    return sound


def find_audio(name):
    if not name:
        return None
    path = os.path.join(AUDIO_DIR, name)
    if os.path.exists(path):
        return path
    if os.path.exists(name):
        return name
    return None


def main():
    csv_path = pick_csv()
    if not csv_path:
        print("No file selected.")
        return

    audio_name, onsets = load_csv(csv_path)
    audio_path = find_audio(audio_name)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(os.path.basename(csv_path))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    tick_don = make_tick_sound(duration_ms=45, volume=0.95)
    tick_ka = make_tick_sound(duration_ms=30, volume=0.85)

    if audio_path:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(0.35)
    else:
        print(f"Audio not found: {audio_name}")

    start_ticks = pygame.time.get_ticks()
    next_hit = 0  # index of next onset to trigger
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        now_ms = pygame.time.get_ticks() - start_ticks

        # trigger tick sounds as notes cross the hit line
        while next_hit < len(onsets) and onsets[next_hit][0] <= now_ms:
            kind = onsets[next_hit][1]
            if "ka" in kind:
                tick_ka.play()
            else:
                tick_don.play()
            next_hit += 1

        screen.fill((30, 30, 30))

        # hit line
        pygame.draw.line(screen, (255, 255, 255, 80), (HIT_X, 40), (HIT_X, HEIGHT - 40), 2)

        # draw onsets
        center_y = HEIGHT // 2
        for time_ms, kind in onsets:
            x = HIT_X + (time_ms - now_ms) * SCROLL_SPEED
            if x < -40:
                continue
            if x > WIDTH + 40:
                break

            color = COLORS.get(kind, (200, 200, 200))
            size = SIZES.get(kind, 16)

            # dim notes that already passed
            if x < HIT_X:
                color = tuple(c // 3 for c in color)

            pygame.draw.circle(screen, color, (int(x), center_y), size)
            pygame.draw.circle(screen, (255, 255, 255), (int(x), center_y), size, 2)

        # time display
        secs = now_ms / 1000
        label = font.render(f"{secs:.1f}s", True, (180, 180, 180))
        screen.blit(label, (10, 10))

        # legend
        ly = 10
        for kind, color in [("don", COLORS["don"]), ("ka", COLORS["ka"]),
                             ("drumroll", COLORS["drumroll"])]:
            pygame.draw.circle(screen, color, (WIDTH - 100, ly + 7), 6)
            t = font.render(kind, True, (180, 180, 180))
            screen.blit(t, (WIDTH - 88, ly))
            ly += 20

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
