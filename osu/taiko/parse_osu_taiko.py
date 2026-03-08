"""Parse osu!taiko .osu files from .osz archives into beat onset CSVs."""
import zipfile
import glob
import os
import sys

CHARTS_DIR = "./osu/taiko/charts"
DATA_DIR = "./osu/taiko/data"
AUDIO_DIR = "./osu/taiko/audio"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


def parse_hit_objects(osu_text):
    """Extract onset times and note types from an osu!taiko file."""
    lines = osu_text.split("\n")
    in_hit_objects = False
    onsets = []

    for line in lines:
        line = line.strip()
        if line == "[HitObjects]":
            in_hit_objects = True
            continue
        if line.startswith("[") and in_hit_objects:
            break
        if not in_hit_objects or not line:
            continue

        parts = line.split(",")
        time_ms = int(parts[2])
        obj_type = int(parts[3])
        hit_sound = int(parts[4])

        # type bitmask: bit0=circle, bit1=slider(drumroll), bit3=spinner(denden)
        if obj_type & 1:
            kind = "don" if (hit_sound & 0x0A) == 0 else "ka"
            if hit_sound & 4:
                kind = "big_" + kind
        elif obj_type & 2:
            kind = "drumroll"
        elif obj_type & 8:
            kind = "spinner"
        else:
            kind = "unknown"

        onsets.append((time_ms, kind))

    return onsets


def get_metadata(osu_text):
    """Extract artist, title, difficulty from osu file."""
    meta = {}
    for line in osu_text.split("\n"):
        line = line.strip()
        if line.startswith("Artist:"):
            meta["artist"] = line.split(":", 1)[1]
        elif line.startswith("Title:"):
            meta["title"] = line.split(":", 1)[1]
        elif line.startswith("Version:"):
            meta["difficulty"] = line.split(":", 1)[1]
        elif line.startswith("Mode:"):
            meta["mode"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("AudioFilename:"):
            meta["audio"] = line.split(":", 1)[1].strip()
    return meta


def process_osz(osz_path):
    """Process one .osz archive, output one CSV per taiko difficulty."""
    basename = os.path.splitext(os.path.basename(osz_path))[0]
    count = 0

    try:
        with zipfile.ZipFile(osz_path) as z:
            osu_files = [n for n in z.namelist() if n.endswith(".osu")]
            audio_extracted = {}

            for osu_name in osu_files:
                text = z.read(osu_name).decode("utf-8", errors="replace")
                meta = get_metadata(text)

                if meta.get("mode") != 1:
                    continue

                onsets = parse_hit_objects(text)
                if not onsets:
                    continue

                # Extract audio once per unique audio file
                audio_src = meta.get("audio", "")
                if audio_src and audio_src not in audio_extracted:
                    audio_ext = os.path.splitext(audio_src)[1]
                    audio_dst = f"{basename}{audio_ext}"
                    for ch in r'<>:"/\|?*':
                        audio_dst = audio_dst.replace(ch, "_")
                    audio_out = os.path.join(AUDIO_DIR, audio_dst)
                    if audio_src in z.namelist():
                        with z.open(audio_src) as src, open(audio_out, "wb") as dst:
                            dst.write(src.read())
                    audio_extracted[audio_src] = audio_dst

                audio_filename = audio_extracted.get(audio_src, "")

                diff = meta.get("difficulty", "unknown")
                for ch in r'<>:"/\|?*':
                    diff = diff.replace(ch, "_")
                csv_name = f"{basename} [{diff}].csv"
                csv_path = os.path.join(DATA_DIR, csv_name)

                with open(csv_path, "w") as f:
                    f.write(f"# audio: {audio_filename}\n")
                    f.write("time_ms,type\n")
                    for time_ms, kind in onsets:
                        f.write(f"{time_ms},{kind}\n")

                count += 1
    except zipfile.BadZipFile:
        print(f"  SKIP (bad zip): {osz_path}", file=sys.stderr)
    except Exception as e:
        print(f"  ERROR: {osz_path}: {e}", file=sys.stderr)

    return count


if __name__ == "__main__":
    osz_files = sorted(glob.glob(os.path.join(CHARTS_DIR, "*.osz")))
    print(f"Found {len(osz_files)} .osz files")

    total = 0
    for i, osz in enumerate(osz_files, 1):
        n = process_osz(osz)
        total += n
        if i % 100 == 0:
            print(f"  [{i}/{len(osz_files)}] {total} charts so far...")

    print(f"Done. {total} taiko onset CSVs written to {DATA_DIR}")
