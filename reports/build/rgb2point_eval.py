"""Run RGB2Point on every reviewed web picture and measure whether its clouds tell our objects apart.

    python reports/build/rgb2point_eval.py            (paths below; CPU is fine, ~1 s per picture)

Inputs : data/web_images/<object>/*.jpg          (our 9 objects, reviewed by eye)
         data/web_images_control/<class>/*.jpg   (chair, airplane: classes RGB2Point WAS trained on)
Weights: ~/.cache/localpoint3d/RGB2point + pc1024_three.pth (the released checkpoint)
Outputs: data/rgb2point_clouds.npz (all clouds, gitignored)
         results/rgb2point/*.csv and reports/<run>/figures/*.png

The test: for every cloud, find the most similar other cloud (smallest Chamfer distance).
If the clouds carry the object's shape, that neighbour is usually the same object.
"""
import argparse
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.path.expanduser("~/.cache/localpoint3d"))
BLUE, ORANGE, AQUA, NEUTRAL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#1b1b1a", "#5b5a56"
CONTROLS = ["chair", "airplane"]


# ---------------------------------------------------------------- model

def load_model():
    """The released checkpoint, built exactly like the authors' inference.py."""
    import torch
    model_file = WORK / "RGB2point" / "model.py"
    source = model_file.read_text()
    # Don't download separate ImageNet weights: the checkpoint already holds the ViT weights.
    patched = re.sub(r'("vit_base_patch16_224"\s*,\s*pretrained\s*=\s*)True', r"\1False", source)
    if patched != source:
        model_file.write_text(patched)
    sys.path.insert(0, str(WORK / "RGB2point"))
    from model import PointCloudNet
    model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
    model.load_state_dict(torch.load(WORK / "pc1024_three.pth", map_location="cpu")["model"])
    return model.eval()


def predict(model, image_path):
    """1 picture -> 1,024 xyz points. Same preprocessing as the authors' utils.predict."""
    import torch
    from PIL import Image
    from torchvision import transforms
    to_input = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = to_input(Image.open(image_path).convert("RGB")).reshape(1, 1, 3, 224, 224)
    with torch.no_grad():
        return model(x)[0].numpy().astype(np.float32)


def run_all(model, folders):
    """{(group, label, file): cloud} for every picture in every class folder."""
    clouds = {}
    for group, root in folders:
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_")):
            pictures = sorted(folder.glob("*.jpg"))
            for i, picture in enumerate(pictures, 1):
                clouds[(group, folder.name, picture.name)] = predict(model, picture)
                print(f"\r{folder.name:16s} {i:4d}/{len(pictures)}", end="", flush=True)
            print()
    return clouds


# ---------------------------------------------------------------- maths

def chamfer(tree_a, a, tree_b, b):
    """Average distance from each point to the nearest point of the other cloud, both ways."""
    return float(tree_b.query(a)[0].mean() + tree_a.query(b)[0].mean()) / 2


def distance_matrix(clouds):
    trees = [cKDTree(c) for c in clouds]
    n = len(clouds)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = chamfer(trees[i], clouds[i], trees[j], clouds[j])
        print(f"\rdistances {i + 1}/{n}", end="", flush=True)
    print()
    return d


def nearest_neighbour_labels(d, labels, allowed=None):
    """For each cloud: the label of its most similar other cloud (only among `allowed` indices)."""
    idx = np.arange(len(labels)) if allowed is None else np.asarray(allowed)
    sub = d[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, np.inf)
    return [labels[idx[j]] for j in sub.argmin(axis=1)], idx


def two_class_accuracy(d, labels, a, b):
    """1-NN accuracy when only clouds of class a and class b are compared. Guessing = about 50 %."""
    allowed = [i for i, lab in enumerate(labels) if lab in (a, b)]
    predicted, idx = nearest_neighbour_labels(d, labels, allowed)
    return float(np.mean([p == labels[i] for p, i in zip(predicted, idx)]))


def chance_1nn(labels):
    """Chance of a random other cloud having the same label (the guessing line for 1-NN)."""
    counts = pd.Series(labels).value_counts().to_numpy()
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def shortcut_checks(d, labels, paths, allowed):
    """Is the 1-NN score just near-copies of the same photo, or plain photo look (colours, layout)?

    a) near-copies: skip any partner whose *picture* is almost the same (dhash differs in <= 12 of 64 bits)
    b) photo look: the same 1-NN test on the pictures themselves (32x32 colour thumbnails), no 3D at all
    """
    from PIL import Image
    sys.path.insert(0, str(REPO / "src"))
    from web_images import dhash_64, different_bits

    idx = np.asarray(allowed)
    sub_labels = [labels[i] for i in idx]
    hashes, thumbs = [], []
    for i in idx:
        with Image.open(paths[i]) as im:
            hashes.append(dhash_64(im))
            thumbs.append(np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32).ravel() / 255)
    near_copy = np.array([[different_bits(a, b) <= 12 for b in hashes] for a in hashes])

    def score(dist, skip=None):
        dist = dist.copy()
        np.fill_diagonal(dist, np.inf)
        if skip is not None:
            dist[skip] = np.inf
        nearest = dist.argmin(axis=1)
        return float(np.mean([sub_labels[j] == sub_labels[i] for i, j in enumerate(nearest)])), nearest

    cloud_d = d[np.ix_(idx, idx)]
    plain, nearest = score(cloud_d)
    without_copies, _ = score(cloud_d, skip=near_copy)
    thumbs = np.stack(thumbs)
    thumb_d = np.sqrt(((thumbs[:, None, :] - thumbs[None, :, :]) ** 2).sum(-1))
    photo_only, _ = score(thumb_d)
    partner_is_copy = float(np.mean([near_copy[i, j] for i, j in enumerate(nearest)]))
    return pd.DataFrame([
        {"test": "RGB2Point clouds, nearest other cloud", "accuracy": plain},
        {"test": "same, but near-copy pictures not allowed as partner", "accuracy": without_copies},
        {"test": "no 3D: nearest picture by 32x32 colour thumbnail", "accuracy": photo_only},
        {"test": "share of nearest-cloud partners that are near-copy pictures", "accuracy": partner_is_copy},
        {"test": "guessing", "accuracy": chance_1nn(sub_labels)},
    ])


# ---------------------------------------------------------------- figures

def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def title(fig, main, sub):
    fig.text(0.01, 0.985, main, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.935, sub, fontsize=9.5, color=MUTED, va="top")


def fig_pairs(pairs, chance, path):
    import matplotlib.pyplot as plt
    ours = pairs[pairs.kind == "two of our objects"].sort_values("accuracy")
    ctrl = pairs[pairs.kind == "chair vs airplane (control)"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.25, right=0.97)
    y_ours, y_ctrl = 0, 1
    ax.scatter(ours.accuracy * 100, np.full(len(ours), y_ours) + np.linspace(-0.18, 0.18, len(ours)),
               s=22, color=BLUE, alpha=0.75, label="one pair of our objects")
    ax.plot([ours.accuracy.median() * 100] * 2, [y_ours - 0.3, y_ours + 0.3], color=INK, lw=2)
    ax.text(ours.accuracy.median() * 100 + 1, y_ours + 0.33, f"median {ours.accuracy.median():.0%}", fontsize=9, color=INK)
    ax.scatter(ctrl.accuracy * 100, [y_ctrl], s=80, color=ORANGE, zorder=3)
    ax.text(ctrl.accuracy.iloc[0] * 100 - 1, y_ctrl + 0.22, f"{ctrl.accuracy.iloc[0]:.0%}", fontsize=9, color=INK, ha="right")
    ax.axvline(chance * 100, color=MUTED, ls="--", lw=1)
    ax.text(chance * 100 + 0.8, 1.45, "guessing", fontsize=8.5, color=MUTED)
    ax.set_yticks([y_ours, y_ctrl], ["two of our objects\n(36 pairs)", "chair vs airplane\n(trained classes)"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(40, 100)
    ax.set_xlabel("how often a cloud's most similar cloud is the same object (%)", color=MUTED, fontsize=9)
    style(ax)
    title(fig, "Two objects at a time, the nearest cloud is usually the same object",
          "Each cloud matched to its most similar other cloud. Chair vs airplane: classes RGB2Point was trained on.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_confusion(cm, chance, accuracy, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    share = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.2, right=0.97)
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE])
    ax.imshow(share.to_numpy(), cmap=cmap, vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = share.iat[i, j]
            ax.text(j, i, f"{v:.0%}" if v >= 0.005 else "", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else INK)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    ax.set_xticks(range(len(cm)), cm.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cm)), cm.index, fontsize=9)
    ax.set_xlabel("object of the most similar cloud", color=MUTED, fontsize=9)
    ax.set_ylabel("true object", color=MUTED, fontsize=9)
    title(fig, f"All 9 objects: nearest cloud is the same object {accuracy:.0%} of the time",
          f"Up to 40 photos per object. Guessing: {chance:.0%}. Outline = correct; rows add up to 100 %.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_side_views(examples, path):
    """x-z side view of 2 clouds per class; controls first."""
    import matplotlib.pyplot as plt
    names = list(examples)
    fig, axes = plt.subplots(2, len(names), figsize=(1.75 * len(names), 4.3))
    fig.subplots_adjust(top=0.8, bottom=0.02, left=0.01, right=0.99, wspace=0.05, hspace=0.05)
    for col, name in enumerate(names):
        for row in range(2):
            ax = axes[row, col]
            p = examples[name][row]
            ax.scatter(p[:, 0], p[:, 2], s=1.2, color=ORANGE if name in CONTROLS else BLUE)
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(name.replace("_", " "), fontsize=9, color=INK)
    title(fig, "RGB2Point output seen from the side",
          "Orange: classes it was trained on. Blue: our objects, two random pictures each. Ours mostly get the chair's L-shaped profile.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=REPO / "data/web_images")
    parser.add_argument("--controls", default=REPO / "data/web_images_control")
    parser.add_argument("--clouds", default=REPO / "data/rgb2point_clouds.npz")
    parser.add_argument("--tables", default=REPO / "results/rgb2point")
    parser.add_argument("--figures", default=REPO / "reports/2026-11-17_rgb2point_web_images/figures")
    parser.add_argument("--per-class", type=int, default=40, help="pictures per class in the distance test")
    args = parser.parse_args()
    Path(args.tables).mkdir(parents=True, exist_ok=True)
    Path(args.figures).mkdir(parents=True, exist_ok=True)

    # 1. clouds for every picture (cached, so the analysis can be re-run quickly)
    if Path(args.clouds).exists():
        data = np.load(args.clouds, allow_pickle=True)
        keys, stack = [tuple(k) for k in data["keys"]], data["clouds"]
        clouds = dict(zip(keys, stack))
        print(f"loaded {len(clouds)} clouds from {args.clouds}")
    else:
        clouds = run_all(load_model(), [("ours", args.images), ("control", args.controls)])
        np.savez_compressed(args.clouds, keys=np.array(list(clouds), dtype=object),
                            clouds=np.stack(list(clouds.values())))
    counts = pd.Series([lab for _, lab, _ in clouds]).value_counts()
    print(counts.to_string())

    # 2. the same number of pictures per class for the distance test (random, fixed seed)
    random.seed(0)
    chosen = []
    for label in counts.index:
        keys = sorted(k for k in clouds if k[1] == label)
        chosen += random.sample(keys, min(args.per_class, len(keys)))
    labels = [k[1] for k in chosen]
    cache = Path(args.clouds).with_name("rgb2point_distances.npz")
    if cache.exists() and [tuple(k) for k in np.load(cache, allow_pickle=True)["keys"]] == chosen:
        d = np.load(cache, allow_pickle=True)["d"]
    else:
        d = distance_matrix([clouds[k] for k in chosen])
        np.savez_compressed(cache, keys=np.array(chosen, dtype=object), d=d)
    ours = [lab for lab in dict.fromkeys(labels) if lab not in CONTROLS]

    # 3a. every pair of classes, two at a time
    rows = [{"class_a": "chair", "class_b": "airplane", "kind": "chair vs airplane (control)",
             "accuracy": two_class_accuracy(d, labels, "chair", "airplane")}] if all(c in labels for c in CONTROLS) else []
    for i, a in enumerate(sorted(ours)):
        for b in sorted(ours)[i + 1:]:
            rows.append({"class_a": a, "class_b": b, "kind": "two of our objects",
                         "accuracy": two_class_accuracy(d, labels, a, b)})
    pairs = pd.DataFrame(rows)
    pairs.round(3).to_csv(Path(args.tables) / "two_class_1nn_2026-11-17.csv", index=False)

    # 3b. all 9 of our objects at once
    allowed = [i for i, lab in enumerate(labels) if lab in ours]
    predicted, idx = nearest_neighbour_labels(d, labels, allowed)
    truth = [labels[i] for i in idx]
    accuracy = float(np.mean([p == t for p, t in zip(predicted, truth)]))
    chance = chance_1nn(truth)
    order = sorted(ours)
    cm = pd.crosstab(pd.Categorical(truth, order), pd.Categorical(predicted, order), dropna=False)
    cm.index.name, cm.columns.name = "true object", "object of most similar cloud"
    cm.to_csv(Path(args.tables) / "nine_class_1nn_confusion_2026-11-17.csv")

    # 3c. within vs between, per class (the earlier test, now on more pictures + controls)
    per_class = []
    for label in dict.fromkeys(labels):
        mine = [i for i, lab in enumerate(labels) if lab == label]
        # compare our objects with our other objects, and chair with airplane
        others = [i for i, lab in enumerate(labels) if lab != label and (lab in CONTROLS) == (label in CONTROLS)]
        within = d[np.ix_(mine, mine)][np.triu_indices(len(mine), 1)]
        between = d[np.ix_(mine, others)].ravel()
        per_class.append({"class": label, "group": "control" if label in CONTROLS else "ours",
                          "pictures_total": int(counts[label]), "pictures_in_test": len(mine),
                          "chamfer_within": within.mean(), "chamfer_between": between.mean(),
                          "between_over_within": between.mean() / within.mean()})
    per_class = pd.DataFrame(per_class)
    per_class.round(4).to_csv(Path(args.tables) / "within_vs_between_all_2026-11-17.csv", index=False)

    summary = {"pictures": len(clouds), "our_pictures": int(counts[ours].sum()), "test_clouds": len(chosen),
               "nine_class_accuracy": accuracy, "nine_class_chance": chance,
               "pair_median": float(pairs[pairs.kind == "two of our objects"].accuracy.median()),
               "pair_min": float(pairs[pairs.kind == "two of our objects"].accuracy.min()),
               "pair_max": float(pairs[pairs.kind == "two of our objects"].accuracy.max()),
               "control_pair": float(pairs[pairs.kind != "two of our objects"].accuracy.iloc[0]) if len(pairs) and
               (pairs.kind != "two of our objects").any() else float("nan")}
    pd.Series(summary).to_csv(Path(args.tables) / "summary_2026-11-17.csv", header=["value"])
    print(pd.Series(summary).to_string())
    print(per_class.round(3).to_string(index=False))

    # 3d. two cheap explanations for a good nearest-neighbour score, checked
    roots = {"ours": Path(args.images), "control": Path(args.controls)}
    paths = [roots[g] / lab / f for g, lab, f in chosen]
    checks = shortcut_checks(d, labels, paths, allowed)
    checks.round(3).to_csv(Path(args.tables) / "shortcut_checks_2026-11-17.csv", index=False)
    print(checks.round(3).to_string(index=False))

    # 4. figures (model output only; the web pictures themselves are not reproduced)
    fig_pairs(pairs, 0.5, Path(args.figures) / "1_two_class_test.png")
    fig_confusion(cm, chance, accuracy, Path(args.figures) / "2_nine_class_confusion.png")
    random.seed(1)
    examples = {}
    for label in CONTROLS + order:
        keys = sorted(k for k in clouds if k[1] == label)
        if len(keys) >= 2:
            examples[label] = [clouds[k] for k in random.sample(keys, 2)]
    fig_side_views(examples, Path(args.figures) / "3_side_views.png")


if __name__ == "__main__":
    main()
