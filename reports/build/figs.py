"""Make the figures + scan stats for the progress-report PDF."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "src"))
from pc_io import load_points
from pointllm_helpers import preprocess

VERDICT = {"Chair_1": 1, "Chair_2": 0, "Chair_3": 0, "Chair_4": 0, "Chair_5": 1, "Chair_6": 1,
           "Chair_7": 1, "Chair_8": 1, "Cup_1": 1, "Cup_2": 1, "Cup_3": 1, "Cup_4": 0,
           "Keyboard_1": 0, "Keyboard_2": 1, "Keyboard_3": 0, "Table_1": 0, "Table_2": 0}
order = ["Chair", "Cup", "Keyboard", "Table"]
files = sorted(glob.glob(f"{REPO}/data/Models/*/*.ply"),
               key=lambda p: (order.index(os.path.basename(os.path.dirname(p))),
                              int(os.path.splitext(p)[0].rsplit("_", 1)[1])))
rng = np.random.default_rng(0)


def scatter3d(ax, xyz, rgb, s=0.3):
    # these Scaniverse PLY exports are Z-up (floor lies in the x-y plane)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=s, linewidths=0)
    lo, hi = xyz.min(0), xyz.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=-60); ax.set_axis_off()


stats = []
fig = plt.figure(figsize=(12, 13.2))
for i, p in enumerate(files):
    name = os.path.splitext(os.path.basename(p))[0]
    xyz, rgb, info = load_points(p)
    size = xyz.max(0) - xyz.min(0)
    # biggest flat surface (floor / desk) via a tiny RANSAC, on 50k points
    sub = xyz[rng.choice(len(xyz), min(50000, len(xyz)), replace=False)]
    flat = 0.0
    for _ in range(300):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9: continue
        flat = max(flat, float(np.mean(np.abs((sub - a) @ (n / L)) < 0.01)))
    stats.append({"name": name, "points": int(len(xyz)), "colour": rgb is not None,
                  "w": float(size[0]), "h": float(size[1]), "d": float(size[2]),
                  "mb": os.path.getsize(p) / 1e6, "plane_frac": flat, "ok": VERDICT[name]})
    idx = rng.choice(len(xyz), min(30000, len(xyz)), replace=False)
    ax = fig.add_subplot(5, 4, i + 1, projection="3d")
    scatter3d(ax, xyz[idx], rgb[idx] if rgb is not None else "grey")
    mark = "correct" if VERDICT[name] else "wrong"
    ax.set_title(f"{name}  ({mark})", fontsize=10, color="#1a7f37" if VERDICT[name] else "#b42318")
    print(name, len(xyz), size.round(2), round(flat, 2), flush=True)
plt.tight_layout()
fig.savefig(f"{OUT}/gallery.png", dpi=130)
plt.close(fig)

# raw scan vs what PointLLM receives, for one good and one bad scan
fig = plt.figure(figsize=(12, 6))
for row, name in enumerate(["Chair_5", "Table_1"]):
    p = [f for f in files if os.path.basename(f).startswith(name + ".")][0]
    xyz, rgb, _ = load_points(p)
    idx = rng.choice(len(xyz), min(30000, len(xyz)), replace=False)
    ax = fig.add_subplot(2, 2, row * 2 + 1, projection="3d")
    scatter3d(ax, xyz[idx], rgb[idx])
    ax.set_title(f"{name}: raw PLY ({len(xyz):,} points, metres)", fontsize=10)
    pts, _ = preprocess(p)
    ax = fig.add_subplot(2, 2, row * 2 + 2, projection="3d")
    scatter3d(ax, pts[:, :3], pts[:, 3:], s=1.2)
    ax.set_title(f"{name}: model input (8,192 points, unit sphere)", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUT}/pipeline.png", dpi=130)
plt.close(fig)

json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
