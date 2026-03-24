"""Create side-by-side comparison images for exp 46 hard_alpha sweep."""
from PIL import Image
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "experiments", "experiment_46")

configs = [
    ("detect_experiment_46a", "a=0.0"),
    ("detect_experiment_46b", "a=0.25"),
    ("detect_experiment_44",  "a=0.5"),
    ("detect_experiment_46c", "a=0.75"),
    ("detect_experiment_46d", "a=1.0"),
]

graphs = [
    "heatmap",           # pred vs target scatter heatmap
    "entropy_heatmap",   # entropy heatmap
    "ratio_confusion",   # ratio confusion histogram
    "ratio_heatmap",     # ratio heatmap
    "metronome_scatter", # metronome scatter
    "metronome_heatmap", # metronome RGB heatmap
]

EVAL = 2

for graph in graphs:
    images = []
    labels = []
    for run, label in configs:
        path = os.path.join(SCRIPT_DIR, "runs", run, "evals", f"eval_{EVAL:03d}_{graph}.png")
        if os.path.exists(path):
            images.append(Image.open(path))
            labels.append(label)
        else:
            print(f"  Missing: {path}")

    if not images:
        print(f"No images for {graph}")
        continue

    # create side-by-side
    w = images[0].width
    h = images[0].height
    label_h = 30
    combined = Image.new("RGB", (w * len(images), h + label_h), (22, 22, 30))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("consola.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        combined.paste(img, (i * w, label_h))
        # draw label
        text_x = i * w + w // 2
        draw.text((text_x, 5), label, fill=(200, 200, 210), font=font, anchor="mt")

    out_path = os.path.join(OUT_DIR, f"compare_{graph}.png")
    combined.save(out_path, quality=95)
    print(f"Saved: {out_path} ({combined.width}x{combined.height})")

print("Done!")
