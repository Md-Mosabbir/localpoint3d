"""Scan gallery for the mid-project report: what the model sees (grey, 8192 points), everyday + local objects.

usage: python reports/build/mid_figs.py <everyday scans dir> <local preview dir from notebook 03> <out.png>
The local pictures are the scan_*.png files notebook 03 saves; the everyday ones are made here.
"""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pointllm_helpers import preprocess, preview  # noqa: E402

everyday_dir, local_dir, out = sys.argv[1:4]
tiles = []
for name in ["Chair_01", "Cup_1", "Table_1", "Keyboard_1"]:
    pts, _ = preprocess(f"{everyday_dir}/{name}_cleaned.ply", colour="grey", specks="removed")
    tiles.append((preview(pts), f"{name.split('_')[0]} (everyday)"))
for p in sorted(glob.glob(f"{local_dir}/scan_*.png")):
    obj = os.path.basename(p)[5:-4].replace("_", " ").replace("benched", "bench")
    tiles.append((Image.open(p).convert("RGB"), f"{obj} (local)"))

W, H, cols = 760, 300, 2
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, (H + 34) * rows), "white")
draw = ImageDraw.Draw(sheet)
for i, (img, label) in enumerate(tiles):
    x, y = (i % cols) * W, (i // cols) * (H + 34)
    img = img.copy()
    w, h = img.size
    img = img.crop((0, int(h * 0.12), w, h))  # drop preview()'s own title line
    img.thumbnail((W - 10, H))
    sheet.paste(img, (x + 5, y + 34))
    draw.text((x + 12, y + 4), label, fill="#1b1b1a", font=font)
sheet.save(out)
print(out, sheet.size)
