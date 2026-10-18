import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

# 1. PointLLM: share of scans named correctly, per object
objs = ["Chair", "Cup", "Keyboard", "Table"]; ok = [5, 3, 1, 0]; n = [8, 4, 3, 2]
fig, ax = plt.subplots(figsize=(6.4, 2.4))
pct = [100 * a / b for a, b in zip(ok, n)]
ax.barh(objs[::-1], pct[::-1], color=BLUE, height=0.55)
for y, (p, a, b) in enumerate(zip(pct[::-1], ok[::-1], n[::-1])):
    ax.text(p + 1.5, y, f"{a} / {b}", va="center", color=INK)
ax.set_xlim(0, 105); ax.set_xlabel("scans named correctly (%)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/by_object.png", dpi=200); plt.close(fig)

# 2. clutter: how much bigger the scan is than the object, vs result
expected = {"Chair": 1.0, "Cup": 0.12, "Keyboard": 0.45, "Table": 1.0}  # rough real sizes (m)
stats = json.load(open(f"{OUT}/stats.json"))
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for s in stats:
    s["ratio"] = max(s["w"], s["h"], s["d"]) / expected[s["name"].split("_")[0]]
for good, col, mk, lab in [(1, BLUE, "o", "named correctly"), (0, ORANGE, "X", "wrong")]:
    pts = [s for s in stats if s["ok"] == good]
    ax.scatter([s["ratio"] for s in pts], [100 * s["plane_frac"] for s in pts], s=70, c=col,
               marker=mk, label=lab, edgecolors="white", linewidths=1.5, zorder=3)
    for s in pts:
        off = {"Chair_8": (8, -4), "Table_2": (7, 2), "Table_1": (4, -12), "Chair_4": (-14, 7)}.get(s["name"], (6, 3))
        ax.annotate(s["name"], (s["ratio"], 100 * s["plane_frac"]), xytext=off,
                    textcoords="offset points", fontsize=7.5, color=MUTED)
ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1x", "2x", "4x", "8x", "16x"])
ax.set_xlabel("scan size / real object size (log scale)")
ax.set_ylabel("points on biggest flat\nsurface (%)")
ax.axvline(4, color=MUTED, lw=1, ls="--", zorder=1)
ax.text(4.15, 21, "4x or more: 0 of 5 correct", fontsize=7.5, color=MUTED)
ax.set_xlim(0.85, 18); ax.set_ylim(15, 82)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/clutter.png", dpi=200); plt.close(fig)
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)

# 3. CLIP zero-shot, redrawn from the team's printed output
labels = ["a 3D scan of a chair", "a three-dimensional photo of a coffee Cup",
          "a 3D scan of a bottle", "a photo of a coffee Cup", "a 3D scan of a building"]
vals = [47.99, 37.09, 9.26, 3.01, 2.65]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
cols = [ORANGE if "Cup" not in l else BLUE for l in labels]
ax.barh(labels[::-1], vals[::-1], color=cols[::-1], height=0.55)
for y, v in enumerate(vals[::-1]):
    ax.text(v + 0.8, y, f"{v:.1f} %", va="center", color=INK)
ax.set_xlim(0, 56); ax.set_xlabel("confidence (%)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="correct object (cup)"), Patch(color=ORANGE, label="wrong object")],
          frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/clip.png", dpi=200); plt.close(fig)
for s in sorted(stats, key=lambda s: s["ratio"]):
    print(f'{s["name"]:11s} ratio {s["ratio"]:5.1f}  plane {s["plane_frac"]:.0%}  ok {s["ok"]}')
