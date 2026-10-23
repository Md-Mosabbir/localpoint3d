# 02 · PointLLM

## What it is

- A multimodal LLM that takes a **coloured point cloud + a text question** and answers in text.
  - Paper: *PointLLM: Empowering Large Language Models to Understand Point Clouds* (ECCV 2024): https://arxiv.org/abs/2308.16911
  - Code: https://github.com/InternRobotics/PointLLM
- Architecture:
  - A point encoder (Point-BERT) turns the cloud into tokens.
  - A projector maps those tokens into the LLM's embedding space.
  - The LLM is Vicuna/LLaMA-7B.
- Trained on **Objaverse** objects with Cap3D-style captions plus instruction data. It is *not* trained on ModelNet40/ShapeNet, so it's a good "open-vocabulary" check before we try classic classifiers.

## Which version we use

Checked 2026-10-17 on the GitHub README:

| Version | Status |
|---|---|
| `RunsenXu/PointLLM_7B_v1.2` | **Latest released checkpoint (Dec 2023). This is what we use.** |
| `PointLLM_13B_v1.1_init` | only initialisation weights, no trained 13B v1.2 listed |
| PointLLM-V2 | accepted to TPAMI (Jul 2025). Only the *training annotations* were released (Mar 2026). Model weights: "coming". |

→ If PointLLM-V2 weights get released, switch `MODEL_NAME` in `src/pointllm_helpers.py` and re-run our batch.

## Requirements and why our notebook looks the way it does

- Tested by the authors on Python 3.10, PyTorch 2.0.1, Transformers 4.28 dev, CUDA 11.7.
- Colab's Python is newer, so we build a separate **Python 3.10 venv** with `uv` and run the model in a background process (`src/pointllm_worker.py`).
- 7B in float16 needs about 14 GB of GPU memory, which fits a free **T4** (15 GB). T4 has no bfloat16, so we use float16.

## Input format (important for our scans)

- Exactly **8192 points**, each `[x, y, z, r, g, b]`.
- xyz is centred and scaled into the unit sphere. rgb is in [0, 1].
- We pick the 8192 points by farthest-point sampling (`preprocess()` in `src/pointllm_helpers.py`).
- No colour → we fill grey (0.5). The model then tends to describe the object as "white/grey/monochrome".
- **Orientation:** iPhone scans are Y-up. If answers are weird, try the `"+90 deg about X"` rotation. *Open question:* which up-axis does PointLLM's training data use? We should test this systematically: same scan, all 4 rotations.

## Prompts from the paper

- `What is this?` (classification, "I" prompt)
- `This is an object of ` (classification, "C" prompt, the model completes the sentence)
- `Caption this 3D model in detail.`

---

## 2026-10-16 · team · first test (old notebook, `archive/pointllm_colab_v0.ipynb`)

**Setup:** T4, PointLLM-7B v1.2, two chair files (`poly.ply` and `processed_Chair_1_8192pts_model.ply`, no colour).

**What happened**
- `What is this?` → "This is a 3D model of a chair." ✅ on both files. **The pipeline works.**
- It **hallucinates context**: "sitting on a white mat", "sitting on a wooden branch".
- Counting is unreliable: "It has four legs" when we saw 6. When we pushed back, it repeated "four legs".
- Without colour it calls the chair "white / monochrome".

**Takeaways**
- Recognising a *common* object works.
- Don't trust detail answers (counts, materials, surroundings) without checking.
- Always export **with colour** from Scaniverse.

**Next**
1. Scan the same chair in Scaniverse LiDAR + photo mode, with colour. Run notebook 01 → CSV.
2. Then local objects: rickshaw, thelagari, CNG, …
3. Track per object: does any of the 3 prompts name it correctly? If not, what does it think it is (bicycle? cart? car?)

---

## 2026-10-18 · Md · baseline: 17 everyday objects (notebook 01)

Full write-up: [`reports/2026-10-18_pointllm_baseline.md`](../reports/2026-10-18_pointllm_baseline.md)

**Setup**
- PointLLM-7B v1.2 on a Colab T4.
- 17 Scaniverse PLY scans from `Drive/Models`: 8 chairs, 4 cups, 3 keyboards, 2 tables. All are ModelNet40 classes, so this is the *easy* baseline.
- Rotation `None`, the 3 paper prompts.
- Raw output: [`results/pointllm_2026-10-18_baseline.csv`](../results/pointllm_2026-10-18_baseline.csv)

**Scoring:** an answer counts as correct if it names the right kind of object. Judged by reading the answers (Claude, to be double-checked by the team).
- I = "What is this?"
- C = "This is an object of "
- Cap = "Caption this 3D model in detail."

| Scan | I | C | Cap | What it thought instead |
|---|---|---|---|---|
| Chair_1 | ✅ | ✅ | ✅ | |
| Chair_2 | ❌ | ❌ | ❌ | treadmill (all 3) |
| Chair_3 | ❌ | ❌ | ~ | "a location", "cleared area in a forest"; Cap: "location featuring a chair" |
| Chair_4 | ❌ | ❌ | ❌ | bird, toy airplane |
| Chair_5 | ✅ | ✅ | ✅ | |
| Chair_6 | ✅ | ✅ | ✅ | (Cap: "situated on a textured tiled floor") |
| Chair_7 | ✅ | ✅ | ✅ | |
| Chair_8 | ✅ | ❌ | ✅ | C: "ruined building" |
| Cup_1 | ✅ | ✅ | ✅ | coffee cup |
| Cup_2 | ✅ | ✅ | ✅ | cup / mug |
| Cup_3 | ✅ | ✅ | ✅ | mug |
| Cup_4 | ❌ | ❌ | ❌ | "cartoon fish figure" (all 3) |
| Keyboard_1 | ❌ | ❌ | ❌ | old painting, building |
| Keyboard_2 | ✅ | ❌ | ✅ | C: carpet |
| Keyboard_3 | ❌ | ❌ | ❌ | house wall + yard, flat board, fish |
| Table_1 | ❌ | ❌ | ❌ | ranch, motorbike |
| Table_2 | ❌ | ❌ | ❌ | bar stool "tipped onto its side", chair |

**Totals**

| Class | Scans right (majority of 3 prompts) | Answers right |
|---|---|---|
| Chair | 5 / 8 | 15 / 24 (Chair_3's "~" counted as right) |
| Cup | 3 / 4 | 9 / 12 |
| Keyboard | 1 / 3 | 2 / 9 |
| Table | 0 / 2 | 0 / 6 |
| **All** | **9 / 17 (53 %)** | **26 / 51 (51 %)** |

**Patterns (these are hypotheses, not proven yet)**
1. **Background or floor is probably in the scans.**
   - Several answers describe a *scene*, not an object: "a location", "forest", "ranch", "house wall and yard", "ruined building".
   - One caption even mentions the "tiled floor".
   - If the floor is a big part of the 8192 points, the object is only a small part of what the model sees. Flat objects (keyboard, table top) would suffer most, and they did.
2. **Orientation might be wrong for some scans.**
   - "Bar stool *tipped onto its side*" (Table_2) and "toy airplane" (Chair_4) sound like a sideways object.
   - Scaniverse is Y-up. We only tried rotation `None`.
3. **The prompt changes the answer.** Same scan, different prompt, different object (Chair_8, Keyboard_2, Table_1). One prompt isn't enough, so keep all 3.
4. **"Cartoon / whimsical" in almost every answer** is the model's style. It comes from its Objaverse training captions and means nothing about our scans.
5. Shape-wise, **cups and chairs work best** and **tables and keyboards worst**. Tables may be confused with chairs/stools because they're from the same furniture family.

**Next**
1. Look at the previews of the failed scans (Chair_2, 3, 4, Cup_4, Keyboard_1, 3, Table_1, 2). Is the floor or background in there? Is the object upright?
2. Re-run the failures with `ROTATIONS_TO_TRY = ["None", "+90 deg about X", "-90 deg about X"]`.
3. Remove the floor (crop in Scaniverse, or remove it in code the way notebook 02 does), then re-run all 17. If accuracy jumps, pattern 1 was right.
4. Only then local objects. This baseline tells us how much of a failure is "PointLLM doesn't know rickshaws" and how much is "our scans are messy".

## 2026-10-18 · Claude · looked at the 17 raw scans (next step 1 done)

Drew every scan from the raw PLY on the laptop and measured each one. Figures and the full table are in the [progress report PDF](../reports/2026-10-18_progress_report_1.pdf), sections 3 and 6. Scripts are in `reports/build/`.

**Measured**
- **The scans are Z-up, not Y-up.** In 15 of 17 scans the biggest flat surface (floor or desk) is perpendicular to Z, and all 17 stand upright when drawn with Z up. The "Scaniverse is Y-up" note above was wrong for these PLY exports. PointLLM got them unrotated. `"-90 deg about X"` turns Z-up into Y-up.
- **Every scan contains the floor or desk:** 22–78 % of all points lie within 1 cm of one plane (RANSAC).
- **Too much scene is the strongest pattern.** Scan size ÷ rough real object size (chair 1 m, cup 12 cm, keyboard 45 cm, table 1 m):
  - below 4× → **9 / 12** named correctly
  - 4× or more → **0 / 5** (Keyboard_1 4×, Chair_3 4.7×, Cup_4 8.7×, Chair_4 11×, Table_1 13×)
- The floor share alone doesn't separate right from wrong (Cup_1 is 65 % floor and was still named correctly).

**What it means:** pattern 1 is confirmed, in a sharper form: it's about how much *scene* the scan contains, not only the floor. Pattern 2 (orientation) is still open. Table_2 is a clean, upright scan and was still called "a bar stool tipped onto its side", which fits an axis mismatch.

**Next:** crop / remove the floor and re-run, and run the failures with `"-90 deg about X"` (Z-up → Y-up).

## 2026-10-21 · team + Claude · colour test set up (does PointLLM need colour?)

**Question:** if we grey the scans out, do PointLLM's answers change? If not, it recognises by shape alone. If yes, colour is part of how it decides.

**How:** notebook 01 can now ask every question twice: `COLOURS_TO_TRY = ["original", "grey"]`. `preprocess(..., colour="grey")` replaces every colour with 0.5 grey *in memory*. The same seed picks exactly the same 8192 points both times (checked on Cup_3: identical xyz), so **only the colour differs**. The CSV gains `colour` and `file_has_colour` columns. A new cell puts the original and grey answers side by side.

**Why not grey copies of the files** (Fahim's first version, `Copy of 01_…`)? It worked, and the previews were fully grey. But:
- `Muura_1.obj` and `Muura_1.ply` both became `Muura_1_grey.ply`, so one overwrote the other.
- The `_grey` suffix broke the object/method/take columns.
- Point clouds came out lighter grey than meshes.
- It wrote 40 files to Drive.

**Noticed in that run (2026-10-21, 40 scans on Drive, 23 of them new local objects):**
- **OBJ exports are Y-up** (floor in the X-Z plane). The PLYs are Z-up.
- Several local-object scans include a ~2 m floor disc around a small object (Kula_1, Matir_Cup_1, Bodna_1). By our baseline rule (scan ≥ 4× the object → never recognised), crop these first.
- 4 mesh PLYs (Chaari_1, Muura_1, Piri_1, Tubewell_1) point to a texture `3DModel.jpg` that isn't on Drive. Their colour is lost even in the "original" run.
- Thela_Gari_2 / _4 and Wooden_Bench_2 / _3 have identical bounding boxes. Check whether they are the same file uploaded twice.
- `parse_scan_name` split two-word names wrongly (`Matir_Cup_1` → object "matir", method "cup"). Fixed: the method only counts if it is lidar/photo/detail/area/speed.

## 2026-10-21 · Claude · floor removal option (off by default)

`preprocess(..., floor="removed")` deletes the biggest horizontal plane in the bottom 30 % of the scan (RANSAC, `src/cleanup.py`). Notebook 01: `FLOOR` in the preview cell, `FLOORS_TO_TRY = ["kept", "removed"]` in the batch.

**Checked on 8 local scans:**
- **Threshold 1 cm.** At 2 cm, 86–91 % of each keyboard was deleted along with the desk, because a keyboard is only ~2 cm tall. At 1 cm the keyboards survive, and chairs, cups and Table_2 come out clean at both thresholds.
- **It removes floors, not the whole room.** Chair_4 still has other objects around it after removal, so cropping in Scaniverse is still best.
- **The up-axis is guessed from the file type:** `.ply` Z-up, `.obj` Y-up.
- **Cost:** about 9 s extra per scan, once (cached).

**Plan:** run the colour test first, then the floor test, one change at a time.

## 2026-10-21 · Md (runs) + Claude (analysis) · colour and floor test

Full write-up: [reports/2026-10-21_colour_and_floor_test.md](../reports/2026-10-21_colour_and_floor_test.md). CSVs: `results/original_grey_floor_kept.csv` and `results/original_grey_floor_removed.csv`, scored.

**Measured (answers correct of 51):**
- colour, floor kept: 24
- grey, floor kept: 13
- colour, floor removed: 24
- grey, floor removed: 19
- 18 Oct baseline (colour, floor kept): 26

**Findings:**
- **Colour matters.** Removing it cost 18 answers and gained 2. Chair_5, 6 and 8 (red, blue, brown) lose recognition without colour.
- **Floor removal helps only grey scans** (+6, 0 lost).
- Cups 1–3 were right every time. Eight cluttered or flat scans were never right.
- 6 empty answers ("This is an object of"), all grey and all on prompt C.

**Surprise: PointLLM isn't repeatable.** The same input on 18 Oct and 21 Oct gave the same text for only 3 of 51 answers, and correctness flipped for 6 of 51. Cause: PointBERT's fps starts at `torch.randint` with no seed. Fixed: `ask(..., seed)` + `SEEDS_TO_TRY` in notebook 01 (not yet tested on GPU). The 18 Oct report's "greedy = repeatable" sentence now has a correction.

**Next:** re-run colour vs grey with seeds [0, 1, 2]; clean the 8 never-right scans by hand into `Models_clean/`.

## 2026-10-21 · Claude · is PointLLM-V2 usable yet? No

- We use **V1**: `RunsenXu/PointLLM_7B_v1.2`.
- **V2 (TPAMI 2025) weights are still not released.** The README TODO "Release PointLLM-V2 with a better model and data" is unchecked. Hugging Face has only v1.1 and v1.2 checkpoints under RunsenXu, and nothing V2 under InternRobotics or OpenRobotLab.
- The only V2 release is the training annotations (2026-03-17). Training a 7B model on them isn't possible on a free T4.
- `QileXu/PointLLM-R-7B` exists on Hugging Face. It's a third-party model, not the official V2, and we haven't checked it.
- Sources: https://github.com/InternRobotics/PointLLM (News + TODO), https://huggingface.co/RunsenXu

## 2026-10-23 · Md (cleaning) + Claude (check) · hand-cleaned scans

Md cleaned all 17 scans by hand and saved them as `data/Models_Cleaned/<Name>_<take>_cleaned.ply` (point clouds with colour, x y z + rgb + alpha).

**Check, by drawing all 17 and comparing their size with the raw scans:**
- **Clean, object only:** Chair_01, 02, 05, 06, 07, 08, Cup_1, 2, 3, Table_2, Keyboard_3.
  - Chair_02 shrank from 1.98 m to 1.24 m. The wall and floor are gone.
- **Stray specks still far from the object:**
  - Chair_04: 7.4 m across
  - Table_1: 12.4 m
  - Cup_4: 1.0 m for a ~12 cm cup
  - Keyboard_1: 1.8 m
  - Keyboard_2: 1.3 m
  - Chair_03 is borderline at 1.4 m.
- Normalisation scales by the farthest point, so in these the object still ends up tiny. Fix: CloudCompare *Tools → Clean → SOR filter*, or delete the specks by hand.

**Notebook 01 now has one ⚙️ Settings cell at the top.** It holds `SCANS_DIR`, `RUN_NAME` and `ONLY_SCANS`, plus the preview settings and the batch lists. The CSV gains `scans_folder` and `version` columns. `parse_scan_name` understands the `_cleaned` suffix.

## 2026-10-23 · Claude · stray specks shrink the object → speck removal

In the Colab previews of the cleaned Chair_04, Table_1 and Keyboard_1, the object was a tiny blob. Cause: scaling to the unit sphere divides by the **farthest** point, and a few specks metres away set that distance.

**Fix:** `cleanup.remove_specks`.
- How it works: cut space into 10 cm cubes, join touching cubes into connected pieces, and drop pieces with less than 1 % of the points.
- Notebook 01 setting: `SPECKS = "removed"` (the default), recorded in the CSV.

**Checked on all 17 cleaned scans** (40k-point samples, distance from the centre to the farthest point):

| Scan | Before | After | Points removed |
|---|---|---|---|
| Chair_04 | 6.30 m | 0.64 m | 0.0 % |
| Table_1 | 9.62 m | 1.25 m | 0.1 % |
| Keyboard_1 | 2.04 m | 0.30 m | 0.0 % |
| Keyboard_2 | 0.70 m | 0.27 m | 0.0 % |
| Cup_4 | 0.36 m | 0.14 m | 0.0 % |
| Chair_03 | 1.05 m | 0.74 m | 0.1 % |
| all others | | | ≤ 0.1 % |

**First try was 5 cm / 2 %.** That deleted 1.3 % of Chair_05: a real leg piece captured separately with a gap above 5 cm. 10 cm / 1 % keeps it.
