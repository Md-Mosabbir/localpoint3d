"""Build reports/2026-11-17_rgb2point_web_images/report.md + report.pdf from results/rgb2point/*.csv.

    python reports/build/build_rgb2point_report.py      (run reports/build/rgb2point_eval.py first)
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from report_writer import Report  # noqa: E402

RUN = REPO / "reports/2026-11-17_rgb2point_web_images"
T = REPO / "results/rgb2point"


def pct(x):
    return f"{x:.0%}"


def main():
    s = pd.read_csv(T / "summary_2026-11-17.csv", index_col=0)["value"]
    per = pd.read_csv(T / "within_vs_between_all_2026-11-17.csv")
    pairs = pd.read_csv(T / "two_class_1nn_2026-11-17.csv")
    checks = pd.read_csv(T / "shortcut_checks_2026-11-17.csv")
    cm = pd.read_csv(T / "nine_class_1nn_confusion_2026-11-17.csv", index_col=0)
    ours = pairs[pairs.kind == "two of our objects"].sort_values("accuracy")
    diag = pd.Series({o: cm.loc[o, o] / cm.loc[o].sum() for o in cm.index}).sort_values()
    check = dict(zip(checks.test, checks.accuracy))

    r = Report("RGB2Point on web photos of Bangladeshi objects",
               "LocalPoint3D · Track 3, Group 2 · 17 Nov 2026 · run on a laptop CPU with the released checkpoint")

    r.heading("1. Summary")
    r.bullets([
        f"We collected **2,250** web photos of 9 Bangladeshi objects and kept **{int(s.our_pictures)}** after checking them by eye. "
        "We also added 93 photos of chairs and airplanes, two classes RGB2Point was trained on, as a control.",
        f"RGB2Point turned all **{int(s.pictures)}** photos into 1,024-point clouds. Seen from the side, most clouds of our objects "
        "look like a **chair**: the checkpoint only knows chairs, airplanes and cars.",
        f"Still, the clouds are not random. With all 9 objects together, a cloud's most similar other cloud is the same object "
        f"**{pct(s.nine_class_accuracy)}** of the time. Guessing would give {pct(s.nine_class_chance)}. With two objects at a time it is "
        f"{pct(s.pair_min)}–{pct(s.pair_max)} (median {pct(s.pair_median)}), and chair vs airplane is {pct(s.control_pair)}.",
        f"This is not just near-copies of the same web photo: without them it is still {pct(check['same, but near-copy pictures not allowed as partner'])}. "
        f"Nor is it plain photo colours: comparing tiny thumbnails of the photos gives only {pct(check['no 3D: nearest picture by 32x32 colour thumbnail'])}.",
        "**What it means:** the clouds keep a trace of what the photo showed, enough to match photos of the same object, "
        "but they do **not** show the object's real shape. They are a fingerprint of the photo, not 3D data of a bodna or a rehal.",
    ])

    r.heading("2. What we did")
    r.bullets([
        "**Photos:** Bing and DuckDuckGo image search with several phrases per object (`data/web_image_queries.csv`, script `src/web_images.py`). "
        "Google Images does not answer scripts. Every photo was checked by eye; about 1 in 3 was usable (other objects, adverts, collages removed).",
        "**Model:** RGB2Point (Lee et al.), released checkpoint `pc1024_three.pth`, trained on ShapeNet chairs, airplanes and cars. "
        "One photo in, 1,024 xyz points out, no colour. No training or fine-tuning by us.",
        "**Chamfer distance:** how far apart two clouds are. For every point, take the distance to the nearest point of the other cloud, "
        "then average. 0 means identical.",
        "**Nearest-neighbour test:** for each cloud, find the most similar other cloud (smallest Chamfer distance) and check whether it shows "
        "the same object. Up to 40 photos per object (patil 34, piri 19), chosen at random with a fixed seed.",
    ])
    r.table(per.rename(columns={"class": "object", "pictures_total": "photos", "pictures_in_test": "in test",
                                "chamfer_within": "within object", "chamfer_between": "to other objects",
                                "between_over_within": "ratio"})[["object", "group", "photos", "in test", "within object",
                                                                  "to other objects", "ratio"]], round_to=3)
    r.text("Average Chamfer distance between two clouds of the same object, and to clouds of other objects. "
           "For our objects the ratio is about 1.0–1.2: on average, two bodna clouds are almost as far apart as a bodna and a muura. "
           "For the control classes it is 1.3.")

    r.heading("3. What the clouds look like")
    r.figure("figures/3_side_views.png", "Side view of RGB2Point's output. Orange: classes it was trained on. Blue: two random photos of each "
             "of our objects. Most of ours get the chair's L-shaped profile (seat and back).")

    r.heading("4. Can the clouds tell our objects apart?")
    r.figure("figures/1_two_class_test.png", f"Two objects at a time. Guessing = 50 %. Hardest pairs: "
             + ", ".join(f"{a.replace('_', ' ')} vs {b.replace('_', ' ')} {pct(acc)}" for a, b, acc in
                         ours[["class_a", "class_b", "accuracy"]].head(3).itertuples(index=False)) + ".")
    r.figure("figures/2_nine_class_confusion.png", "All 9 objects at once. Rows: true object; columns: object of the most similar cloud. "
             f"Best: {diag.index[-1].replace('_', ' ')} {pct(diag.iloc[-1])}; worst: {diag.index[0].replace('_', ' ')} {pct(diag.iloc[0])}.")

    r.heading("5. Two cheap explanations, checked")
    r.table(checks.rename(columns={"accuracy": "result"}), round_to=3)
    r.bullets([
        "**Near-copies:** the web has many copies of the same product photo. Only "
        f"{pct(check['share of nearest-cloud partners that are near-copy pictures'])} of the matches were near-copies, and excluding them "
        "barely changes the score.",
        "**Photo colours and layout:** matching the photos themselves as 32×32 thumbnails is much worse than matching the clouds. "
        "So RGB2Point's image encoder passes on more than colours. A stronger image model (e.g. CLIP) would probably match the photos even "
        "better, which is the point of the next step.",
    ])

    r.heading("6. What this means for the project")
    r.bullets([
        "RGB2Point clouds of our objects should **not** be used as 3D scans or fed to PointLLM as if they were our objects: "
        "PointLLM would mostly see a chair.",
        "The reviewed photos are useful as they are: real photos of the 9 objects for **CLIP / MetaCLIP** zero-shot tests and for captions.",
        "Getting 3D from photos would need a model trained on many kinds of objects. We tried TripoSR on a laptop CPU, "
        "but it ran out of resources, so it would need Colab with a GPU.",
    ])

    r.heading("7. Limits")
    r.bullets([
        "Web photos are not checked for licences. They are kept out of git and out of this report; only model output is shown.",
        "Patil (34) and piri (19) have few usable photos; their results are less certain.",
        "Same-object photos come from different angles. A model that gives views in the camera's frame would make even good clouds of the "
        "same object differ, so the nearest-neighbour test is a strict test.",
        "Numbers come from one random sample of up to 40 photos per object (seed 0).",
    ])
    r.text("Files: tables in `results/rgb2point/`, code in `reports/build/rgb2point_eval.py`, notes in `journal/06_rgb2point_web_images.md`.")
    print(r.save(RUN))


if __name__ == "__main__":
    main()
