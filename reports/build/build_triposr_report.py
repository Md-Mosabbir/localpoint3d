"""Build reports/2026-11-21_triposr_web_images/report.md + report.pdf from results/triposr/*.csv.

    python reports/build/build_triposr_report.py

The numbers come from the batch log the TripoSR notebook wrote on Colab (batch_log.pdf in the same folder),
parsed into results/triposr/. Figure 2 shows only point clouds cropped from that log, never the web photos.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from report_writer import Report  # noqa: E402

RUN = REPO / "reports/2026-11-21_triposr_web_images"
T = REPO / "results/triposr"

# Our reading of the 40 previews in batch_log.pdf (all bodna: the log only previews the first 40 photos)
PREVIEWS = {
    "one object on a light background (most still have a pale halo)": 27,
    "coloured or scene background rebuilt as a slab (0008, 0036, 0037, 0041, 0076)": 5,
    "a hand holding the bodna, rebuilt with it (0001, 0005, 0006, 0057)": 4,
    "two bodnas in one photo, rebuilt as one blob (0003, 0014, 0026, 0028)": 4,
}


def main():
    per_image = pd.read_csv(T / "triposr_batch_per_image_2026-11-21.csv")
    per_object = pd.read_csv(T / "triposr_batch_per_object_2026-11-21.csv")
    n = len(per_image)
    watertight = int((per_image.watertight == "yes").sum())
    new = per_image[per_image.mesh_status == "success"]
    per_object["seconds_median"] = per_object.seconds_median.map(lambda s: "–" if pd.isna(s) else f"{s:.1f}")

    r = Report("TripoSR on web photos of Bangladeshi objects",
               "LocalPoint3D · Track 3, Group 2 · 21 Nov 2026 · Colab T4 GPU, released TripoSR checkpoint")

    r.heading("1. Summary")
    r.bullets([
        f"TripoSR turned all **{n}** web photos of 8 Bangladeshi objects into 3D meshes and then into coloured "
        "**8,192-point clouds**, the size PointLLM expects. Nothing failed.",
        "Unlike RGB2Point (17 Nov), which made every object look like a chair, TripoSR's clouds **look like the object in the photo**: "
        "a bodna comes out with its round body, spout and handle, in its own colour.",
        "But the run kept the photo backgrounds (`REMOVE_BACKGROUND = False`). So TripoSR also rebuilt the background: most clouds have "
        "a pale slab behind the object, coloured backgrounds become coloured slabs, and hands or a second bodna in the photo become 3D too.",
        "TripoSR sees one side only, so the back of every object is a guess.",
        "**What it means:** TripoSR is the first photo-to-3D model that gives usable-looking shapes of our objects. Before the clouds are used "
        "(e.g. in PointLLM), the batch should be **re-run with background removal on**, and we should measure whether the clouds really carry "
        "the object's shape.",
    ])

    r.heading("2. What we did")
    r.bullets([
        "**Photos:** the web photos collected for the RGB2Point test (Bing and DuckDuckGo, checked by eye). "
        "Same counts as on 17 Nov for 7 objects; see section 6 for rehal and the wooden bench.",
        "**Model:** TripoSR (Stability AI and Tripo), released checkpoint `stabilityai/TripoSR`, trained on Objaverse, which has many kinds "
        "of objects. One photo in, a coloured 3D mesh out. No training or fine-tuning by us.",
        "**Notebook:** `notebooks/triposr.ipynb`, run on Colab with a T4 GPU.",
        "**Mesh → cloud:** 8,192 points sampled evenly over each mesh's surface, each point coloured like the surface under it. "
        "Clouds are not centred or scaled (`NORMALIZE_POINTCLOUD = False`).",
        "**Settings:** marching-cubes resolution 192 (TripoSR's default is 256), background removal **off**, model turned upright the way "
        "TripoSR's own demo does, random seed 0.",
    ])

    r.heading("3. Every photo became a 3D model and a cloud")
    r.figure("figures/1_photos_per_object.png", f"Photos per object. All {n} gave a mesh and an 8,192-point cloud; "
             f"{watertight} meshes ({watertight / n:.0%}) are watertight, i.e. closed with no holes.")
    r.table(per_object.rename(columns={"photos": "photos", "watertight_pct": "watertight %",
                                       "vertices_median": "mesh vertices (median)",
                                       "nn_distance_median": "point spacing (median)",
                                       "seconds_median": "seconds per photo"})
            [["object", "photos", "watertight %", "mesh vertices (median)", "point spacing (median)", "seconds per photo"]],
            round_to=4)
    r.bullets([
        f"**Speed:** about {new.seconds.median():.0f} s per photo on the T4; the whole batch took 27.6 min. "
        "Bodna and hurricane lamp were made in an earlier session, so the log only has their reload time (shown as – above).",
        "**Point spacing** (average distance from a point to its nearest neighbour) is almost the same for every object, 0.008–0.009. "
        "So the points are spread evenly; there are no empty or crowded clouds.",
        "These numbers show the batch **ran correctly**. They do not show whether the shapes are right.",
    ])

    r.heading("4. What the clouds look like")
    r.figure("figures/2_examples.png", "Point clouds made by TripoSR from bodna photos (the photos themselves are not shown). Top: a plain "
             "photo gives a recognisable bodna with a pale halo. Middle: a coloured or scene background becomes part of the cloud. "
             "Bottom: a hand or a second bodna is rebuilt too.")
    r.text("The log previews the first 40 photos, all bodna. Our reading of them:")
    r.table(pd.DataFrame({"what the cloud shows": list(PREVIEWS), "previews": list(PREVIEWS.values())}))
    r.bullets([
        "**Shape:** in the clean cases the body, neck, spout and handle are all there. That is a big step up from RGB2Point's chair outline.",
        "**Background:** even on a white background most clouds have a pale halo or slab, because TripoSR treats the whole picture as one "
        "object. These background points shift the cloud's centre and size, so they would distort PointLLM's input after unit-sphere scaling.",
        "**Clutter:** hands, a second object, a black or green backdrop and room scenes all end up in the cloud.",
    ])

    r.heading("5. What this means for the project")
    r.bullets([
        "**Re-run with `REMOVE_BACKGROUND = True`.** TripoSR's own pipeline cuts the object out first (rembg) and places it on grey. "
        "That should remove the slabs and most hands; photos with two objects should still be dropped by hand.",
        "**Then test the clouds like the RGB2Point ones:** nearest-neighbour test (is a cloud's most similar cloud the same object?) "
        "and side views. If they pass, they can be fed to PointLLM's multiple choice next to our phone scans.",
        "**Compare with our own scans:** TripoSR clouds of a bodna should be closer to our Bodna_1/2 scans than to other objects. "
        "That would show web photos can add training or test data for local objects.",
        "TripoSR guesses the back from one view, so clouds are best for shape-from-the-front questions, not exact 3D measurements.",
    ])

    r.heading("6. Limits and open questions")
    r.bullets([
        "**Rehal has 219 photos, but only 98 rehal photos passed the 17 Nov review**, and the wooden bench (142 reviewed photos) is "
        "missing. The rehal files run from 0001 to 0249, close to the 250 downloaded. So this batch probably includes rehal photos that "
        "were not reviewed (or were rejected). To check before using the rehal clouds.",
        "bodna_0038 and bodna_0038_1 are the same photo (identical mesh), so bodna has 88 different photos, not 89.",
        "No quality measure yet: everything in section 4 is from looking at 40 previews of one object.",
        "Web photos have unverified licences. The batch log (`batch_log.pdf`) contains the 40 bodna input photos; this report shows only "
        "model output.",
        "The meshes and clouds stay on Google Drive (`MyDrive/TripoSR_Batches/`), not in git.",
    ])
    r.text("Files: per-photo and per-object tables in `results/triposr/`, the original batch log in `batch_log.pdf` next to this report, "
           "code in `notebooks/triposr.ipynb`, notes in `journal/06_rgb2point_web_images.md`.")
    print(r.save(RUN))


if __name__ == "__main__":
    main()
