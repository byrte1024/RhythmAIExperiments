import subprocess
import glob
import os

CHARTS_DIR = "./osu/taiko/charts"
UNRAR = r"C:\Program Files\WinRAR\UnRAR.exe"

rars = sorted(glob.glob(os.path.join(CHARTS_DIR, "*.rar")))
print(f"Found {len(rars)} RAR archives")

for i, rar in enumerate(rars, 1):
    name = os.path.basename(rar)
    print(f"[{i}/{len(rars)}] Extracting: {name}")
    subprocess.run(
        [UNRAR, "e", "-o+", rar, CHARTS_DIR + "/"],
        stdout=subprocess.DEVNULL,
        check=True,
    )

print("Done.")
