# 01 · Scaniverse: LiDAR vs photogrammetry, how close are they?

**Why we care:** if both capture methods give about the same point cloud, we can use either one (and teammates without a LiDAR iPhone can still help). If they differ, we need to know *where*, because thin parts like rickshaw spokes may be exactly what a model needs to recognise the object.

**Our tool:** Scaniverse (latest version), Mesh export as PLY/OBJ.
**How we measure:** `notebooks/02_lidar_vs_photogrammetry.ipynb`. Results go in `results/lidar_vs_photo.csv`.

---

## 2026-10-17 · literature review (done with an AI research agent; sources below)

### Short answer (expected)

- **Overall shape: similar.** After alignment, expect a few cm of difference on big flat parts (cart bed, CNG body).
- **Fine detail: different.** Thin rods, spokes, ropes, hood frames and shiny metal are where the two methods disagree most. That disagreement is a result worth reporting.

### What the sources say

**The iPhone LiDAR sensor**
- Depth map is only **256 × 192** pixels. Usable range is about **5 m**. (Apple developer forum)

**Luetzenburg et al. 2021, *Scientific Reports*, iPhone 12 Pro**
- About **±1 cm** accuracy on objects larger than 10 cm. Objects smaller than about 5 cm are not reliably captured.
- Point density drops fast with distance: about 7,200 pts/m² at 25 cm, but only about 150 pts/m² at 2.5 m.
- Their conclusion: LiDAR is faster than photogrammetry, but not as accurate.

**Sensors 2022: iPad LiDAR vs photogrammetry on small, complex objects**
- Up to **50 % of LiDAR points** did not match the reference, mainly because LiDAR is too sparse.
- Photogrammetry did clearly better on these objects.

**ISPRS 2026: iPhone 12 Pro apps vs a professional laser scanner (Trimble X9)**
- Error (3D RMSE) was about **5–22 cm** depending on the app. Scaniverse wasn't in this test.

**Scale and drift**
- LiDAR scans are in real metres.
- Plain photogrammetry has no real scale unless the app takes it from ARKit/LiDAR. Scaniverse gives metres in both modes, *to verify with our own scans*.
- On large objects the phone's motion tracking drifts, so **walk slowly**.

**Known failure modes**

| Situation | LiDAR | Photogrammetry |
|---|---|---|
| Thin parts (spokes, rods, ropes) | often missing (Polycam's help: loses "railings, narrow pipes, chair legs") | patchy, but better than LiDAR |
| Shiny / chrome metal | distorted or missing | also fails (no stable texture) |
| Plain surfaces with no texture | fine | struggles (needs visual features) |
| Dark surfaces, direct sun | reported to degrade, **but only blogs say so, no controlled study found** | needs even light |

Gap: we found no peer-reviewed study of Scaniverse specifically, or of iPhone photo mode vs LiDAR on vehicle-sized objects. **That's a gap our project can fill.**

### How we compare two scans (implemented in notebook 02)

1. Remove the floor, subsample both scans to the same number of points, then normalise to the unit sphere. This is the same normalisation ModelNet40/PointLLM use.
2. Align the two scans:
   - Try 12 starting angles around the up axis.
   - Refine each one with **ICP** (iterative closest point).
   - Keep the best fit.
3. Measure:
   - **Chamfer distance**: average gap between the two clouds.
   - **Hausdorff distance**: the single worst gap. We also report the 95 % version, which ignores the worst 5 % of points.
   - **F-score @ 1 %, 2 %, 5 %** of the object size: the share of points that have a close match in the other cloud, in both directions.
   - A heatmap of **where** the scans differ.
4. **Baseline first:** also compare two takes made with the *same* method. LiDAR vs photo only "differs" if the gap is bigger than the take-to-take gap.

Tested 2026-10-17 on a synthetic chair (B rotated 70°, one leg removed):
- Alignment was recovered.
- F-score@0.02 = 0.96.
- The missing leg showed up clearly in the heatmap and in the Hausdorff distance (≈ 40 cm).

### Sources

- Luetzenburg et al. 2021: https://pmc.ncbi.nlm.nih.gov/articles/8593014
- Sensors 2022, iPad LiDAR vs photogrammetry: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9657006/
- ISPRS 2026, iPhone apps vs Trimble X9: https://isprs-archives.copernicus.org/articles/XLIX-M-1-2026/1/2026/isprs-archives-XLIX-M-1-2026-1-2026.pdf
- Smartphone LiDAR for surveying (2022): https://iris.poliba.it/retrieve/7da9a53f-82ec-4471-9402-88aec7532fe7/2022_Smartphone_LiDAR_Technologies_for_Surveying_pdfeditoriale.pdf
- Apple forum, LiDAR depth resolution: https://developer.apple.com/forums/thread/695771
- Apple WWDC23, Object Capture with LiDAR: https://developer.apple.com/videos/play/wwdc2023/10191
- Polycam, scan accuracy and thin objects: https://learn.poly.cam/hc/en-us/articles/50434745985556-How-Accurate-Are-Polycam-Scans
- Scaniverse export formats: https://community.scaniverse.com/t/which-formats-can-scaniverse-export-to/50
- Open3D point cloud tutorial (alternative tool): https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

---

## 2026-10-18 · Scaniverse-specific follow-up (web search)

**Which mode is which.** Under *New Scan → Mesh*, Scaniverse has three processing modes:

| Mode | Method | Made for | Stated detail |
|---|---|---|---|
| Speed (was "Standard") | LiDAR | quick results | ~10 mm |
| Area (was "Ultra") | LiDAR | rooms, buildings, outdoors | ~5 mm |
| Detail | photogrammetry | objects, people, small textured scenes | not stated |

- Scaniverse's own advice: Detail mode is for textured objects. The LiDAR modes work better on **plain or shiny surfaces**. This matches the failure-mode table above.
- On phones without LiDAR, Scaniverse 2.0+ estimates depth from the camera images with a neural network (Niantic's ManyDepth). That counts as a third kind of input, not "LiDAR".
- All processing happens on the phone.
- **The exported OBJ header records the mode** (we saw `Mode: Detail` in one of our files). So we can tell LiDAR from photo afterwards even when the file name doesn't say.

**Accuracy evidence for Scaniverse itself is thin.**
- One study of an iPhone 13 Pro on building documentation, measured against a terrestrial laser scanner: Scaniverse did worst of the apps tested, with RMSE ≈ 55 cm against ≤ 10 cm for the others. It also had very few points, and drift split the surfaces.
- That test was **room scale, with an older app version**, so it says little about a chair or a rickshaw.
- We found **no study of Detail mode vs Area mode on the same object.** Our notebook 02 measurements would be new.

**Answer for now:** the modes should be interchangeable for the **overall shape** (big parts within a few cm). They are **not** interchangeable for thin, shiny or plain parts, and which mode wins depends on the surface. This is still unmeasured on our own scans.

Sources:
- Scaniverse Detail mode announcement: https://scaniverse.com/news/scaniverse-detail-mode-photogrammetry (now redirects to Niantic Spatial)
- Scaniverse 2.0, non-LiDAR devices: https://scaniverse.com/news/scaniverse-2-0-adds-support-for-non-lidar-devices
- Mode names and resolutions (secondary): https://podfeet.com/blog/?p=27025
- iPhone 13 Pro scanning-app comparison: https://www.mdpi.com/2673-7418/3/4/30

---

## Capture log

| Date | Who | Object | Phone | Methods / takes | Conditions | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

## Our measurements

*(paste the summary rows from `results/lidar_vs_photo.csv` here with a sentence on what they mean)*
