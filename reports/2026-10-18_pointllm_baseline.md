# Report: PointLLM baseline on 17 everyday-object scans

| | |
|---|---|
| **Date** | 2026-10-18 |
| **Project** | localpoint3d: can 3D models trained on Western datasets recognise Bangladeshi objects? |
| **Team** | Shahedul Islam Shikder · Fahim Muntasir · Md Mosabbir |
| **Experiment run by** | Md Mosabbir |
| **Notebook** | [`notebooks/01_pointllm_recognition_test.ipynb`](../notebooks/01_pointllm_recognition_test.ipynb) (repo at commit `fc3d3f7`) |
| **Raw results** | [`results/pointllm_2026-10-18_baseline.csv`](../results/pointllm_2026-10-18_baseline.csv) (51 rows) |

---

## 1. Summary

We gave 17 of our own iPhone scans of everyday objects (chairs, cups, keyboards, tables) to **PointLLM-7B** and asked it what each object is.

- **9 of 17 scans (53 %)** were named correctly by most of the three prompts.
- **26 of 51 individual answers (51 %)** were correct.
- Results differ a lot by object:

| Object | Scans correct |
|---|---|
| Cups | 3 / 4 |
| Chairs | 5 / 8 |
| Keyboards | 1 / 3 |
| Tables | 0 / 2 |

Many wrong answers describe a **scene** ("a location", "a ranch", "a cleared area in a forest", "a house wall and yard") rather than an object. This suggests our scans contain a lot of **floor or background**. A few answers hint that some objects are **lying on their side** ("a bar stool tipped onto its side"). Both are problems with our input, not necessarily with the model, and we can fix and re-test both.

**Why this matters for the project:** all four objects are classes in ModelNet40 and are common in 3D datasets. This baseline measures how well the pipeline works on *easy, familiar* objects. When we later test rickshaws and thelagaris, we need it to separate two causes of failure:
- "the model doesn't know this object"
- "our scans are messy"

---

## 2. Goal

1. Check that the full pipeline works end to end on many real scans: iPhone scan → Google Drive → Colab → PointLLM → CSV.
2. Measure how often PointLLM correctly names everyday objects scanned by us. This is the reference point for later tests on local objects.
3. Spot problems in our scanning or preprocessing before we scan local objects.

---

## 3. Setup

### 3.1 Hardware and software

| Item | Value |
|---|---|
| Compute | Google Colab, free tier, **NVIDIA T4** GPU (15 GB) |
| Model | `RunsenXu/PointLLM_7B_v1.2`: the latest released PointLLM checkpoint (PointLLM-V2 weights are not public yet) |
| Precision | float16 (the T4 doesn't support bfloat16) |
| Model environment | separate Python 3.10 venv (built with `uv`), PyTorch 2.0.1, Transformers 4.28 dev, as required by PointLLM |
| How the model runs | as a background worker process (`src/pointllm_worker.py`); the notebook sends it questions |
| Decoding | greedy (`sample=False`), so the same input always gives the same answer |

### 3.2 Data

17 PLY files captured with an iPhone using the Scaniverse app, stored in the shared Google Drive folder `My Drive/Models`.

| Object | Files | File sizes |
|---|---|---|
| Chair | `Chair_1` … `Chair_8` | 1.7 – 9.3 MB |
| Cup | `Cup_1` … `Cup_4` | 1.9 – 5.2 MB |
| Keyboard | `Keyboard_1` … `Keyboard_3` | 2.8 – 4.8 MB |
| Table | `Table_1`, `Table_2` | 3.5 – 9.0 MB |

The file names don't record the capture method (LiDAR or photo mode), so this run doesn't compare methods.

### 3.3 Preprocessing: what the model actually receives

PointLLM always expects exactly **8192 points**, each with position and colour `[x, y, z, r, g, b]`. Our `preprocess()` in `src/pointllm_helpers.py` produces this:

1. **Load** the file with `src/pc_io.py`. A mesh gets 100,000 points sampled on its surface; a point cloud is used as it is. Colour is read from the file; if there is none, every point is grey.
2. **Reduce** to at most 40,000 points at random (for speed).
3. **Farthest point sampling** down to 8192 points, so they are spread evenly over the object.
4. **Rotation:** none in this run.
5. **Normalise:** move the centre to (0, 0, 0) and scale so the farthest point is at distance 1. This is the same convention as PointLLM's training data and ModelNet40.

Important: **nothing removes the floor or background.** Whatever was in the Scaniverse export goes to the model.

### 3.4 Prompts

Each scan got the three prompts used in the PointLLM paper:

| Short name | Prompt | Type |
|---|---|---|
| I | `What is this?` | open question |
| C | `This is an object of ` | sentence completion |
| Cap | `Caption this 3D model in detail.` | detailed description |

17 scans × 3 prompts = **51 answers**.

### 3.5 Scoring

An answer is **correct** if it names the right kind of object (e.g. "mug" counts for a cup). A scan is **correct** if at least 2 of its 3 answers are correct.

⚠️ The answers were scored by **one rater (Claude) reading them**, with no second check yet. The team should review the table in section 4.1.

---

## 4. Results

### 4.1 Every scan

✅ correct · ❌ wrong · ~ partly right (object mentioned, but as part of a scene)

| Scan | I | C | Cap | What PointLLM said |
|---|---|---|---|---|
| Chair_1 | ✅ | ✅ | ✅ | "a black chair … cartoon style" |
| Chair_2 | ❌ | ❌ | ❌ | **treadmill** in all three ("motor, deck, and control panel") |
| Chair_3 | ❌ | ❌ | ~ | "a location … buildings, roads"; "a cleared area in a forest"; Cap: "a location featuring a white cartoon-style chair" |
| Chair_4 | ❌ | ❌ | ❌ | "resembles a rare species of a bird"; "a toy airplane" |
| Chair_5 | ✅ | ✅ | ✅ | "a chair … vibrant red seat and backrest" |
| Chair_6 | ✅ | ✅ | ✅ | "a chair … painted in a vibrant shade of blue"; Cap adds "situated on a textured tiled floor" |
| Chair_7 | ✅ | ✅ | ✅ | "a chair … designed in a cartoon style" |
| Chair_8 | ✅ | ❌ | ✅ | chair; C: "a dilapidated, ruined building" |
| Cup_1 | ✅ | ✅ | ✅ | "a coffee cup" |
| Cup_2 | ✅ | ✅ | ✅ | cup / "a traditional mug" |
| Cup_3 | ✅ | ✅ | ✅ | "resembles a mug"; "a ceramic mug" |
| Cup_4 | ❌ | ❌ | ❌ | **"a whimsical, rare cartoon fish figure"** in all three |
| Keyboard_1 | ❌ | ❌ | ❌ | "a rare, old painting"; "a building" |
| Keyboard_2 | ✅ | ❌ | ✅ | "a rare type of keyboard … QWERTY"; C: "a cartoon-like carpet" |
| Keyboard_3 | ❌ | ❌ | ❌ | "a residential house's exterior … wall and yard"; "a rectangular flat board"; "cartoon fish" |
| Table_1 | ❌ | ❌ | ❌ | "a ranch … main building and an outhouse"; "a modern white motorbike" |
| Table_2 | ❌ | ❌ | ❌ | "a bar stool that is **tipped onto its side**"; "a chair" |

### 4.2 Totals

| Object | Scans correct | Answers correct |
|---|---|---|
| Chair | 5 / 8 (63 %) | 15 / 24 (Chair_3's "~" counted as correct) |
| Cup | 3 / 4 (75 %) | 9 / 12 |
| Keyboard | 1 / 3 (33 %) | 2 / 9 |
| Table | 0 / 2 (0 %) | 0 / 6 |
| **All** | **9 / 17 (53 %)** | **26 / 51 (51 %)** |

### 4.3 By prompt

| Prompt | Correct |
|---|---|
| I: "What is this?" | 9 / 17 |
| C: "This is an object of" | 7 / 17 |
| Cap: "Caption … in detail" | 10 / 17 (counting Chair_3's "~") |

The sentence-completion prompt (C) did worst. It produced most of the one-off wrong answers on scans the other two prompts got right (Chair_8, Keyboard_2).

---

## 5. Analysis

These are **hypotheses** drawn from the answers. None of them has been confirmed yet by looking at the scans themselves.

### 5.1 Floor or background in the scans (most likely the biggest problem)

**Evidence**
- 5 scans got at least one *scene* description instead of an object:
  - Chair_3: "location", "forest"
  - Chair_8: "ruined building"
  - Keyboard_1: "building"
  - Keyboard_3: "house exterior … wall and yard"
  - Table_1: "ranch … outhouse"
- One caption explicitly mentions "a textured tiled floor" (Chair_6).

**Why it matters**
- Normalisation scales the *whole* scan into the unit sphere. If a 2 m × 2 m patch of floor is included, the object becomes a small bump on a big flat surface.
- Farthest-point sampling then spends many of the 8192 points on the floor.
- The flat objects did worst: keyboards 1/3, tables 0/2. They are the easiest to lose against a flat floor.

**How to test:** remove the floor and re-run (section 7).

### 5.2 Orientation

**Evidence**
- "A bar stool that is *tipped onto its side*" (Table_2).
- A chair seen as a "toy airplane" or a "bird" (Chair_4).

**Why it matters:** Scaniverse exports are Y-up. We don't know which axis PointLLM's training data treats as up. This run only used rotation `None`.

**How to test:** run the failed scans with `"+90 deg about X"` and `"-90 deg about X"`. The notebook already supports this through `ROTATIONS_TO_TRY`.

### 5.3 Keyboards: unknown object, or too flat to tell apart?

**Team note (Md):** maybe PointLLM simply hasn't learned "keyboard".

**Counter-argument**
- PointLLM was trained on Objaverse (~800k objects), and keyboard is a ModelNet40 class, so keyboards are very likely in its training data. Not verified.
- The wrong answers for keyboards are all **flat rectangles**: "painting", "carpet", "flat board", "house wall". So the model got the *shape* right but had nothing to distinguish it from other flat rectangles.
- The keys are a few mm tall. That is below what iPhone scanning reliably captures (about 5 cm limit of detection for LiDAR, see journal 01), and 8192 points leave very few points per key.

**Current best guess:** our keyboard scans look like a plank, so the cause is scan detail rather than a missing concept.

**Why this matters for the project:** rickshaw spokes and hood frames are also thin details, and could be lost in the same way.

### 5.4 The prompt changes the answer

- Even with greedy decoding, different prompts give different objects for the same scan (Chair_8, Keyboard_2, Table_1).
- So the model's "belief" about the object is weak for these scans.
- We should keep all three prompts and report agreement between them as a confidence signal.

### 5.5 Things that are NOT findings

- **"Cartoon", "whimsical", "rare"** appear in almost every answer. This is the style of PointLLM's training captions (Objaverse) and tells us nothing about our scans.
- **Colours** (e.g. "vibrant red seat", "blue chair") show that colour from the scans reached the model. We haven't checked whether the colours are actually correct.

### 5.6 Comparison with the first test (2026-10-16)

The earlier test, with two downloaded chair files and no colour, also identified "a chair". It also showed hallucinated surroundings ("sitting on a white mat") and wrong counts ("four legs"). Today's run confirms both on a larger set:
- recognition works on some clean scans
- surroundings and details in the answers are unreliable

---

## 6. Limitations of this run

- **Small sample:** 2–8 scans per class. One scan more or less changes a class's percentage a lot.
- **One rater, by reading.** The scoring hasn't been reviewed by the team.
- **Scan quality is uneven.** The person who captured the scans confirms that **many of them are bad**. So this run mixes good and bad scans, and the 53 % reflects scan quality as much as the model. Which scans are bad hasn't been recorded yet.
- **Scans not inspected.** We haven't looked at the previews of what the model saw, so sections 5.1 and 5.2 are inferred from the answers only.
- **Single setting:** one rotation (`None`), greedy decoding, one model. No repeated runs. Greedy decoding is deterministic, so repeating wouldn't change anything; a different random sample of the 8192 points could.
- **Capture method unknown:** LiDAR or photo mode isn't recorded in the file names.

---

## 7. Next steps

| # | Step | Purpose | Effort |
|---|---|---|---|
| 1 | Look at the preview images of the 8 failed scans (CPU, laptop) | confirm or reject the floor and rotation hypotheses | ~minutes |
| 2 | Re-run the failures with `ROTATIONS_TO_TRY = ["None", "+90 deg about X", "-90 deg about X"]` | test the rotation hypothesis | 1 Colab session |
| 3 | Remove the floor (crop in Scaniverse before exporting, and/or add automatic floor removal to `preprocess()` like notebook 02 does), then re-run all 17 | test the floor hypothesis. If accuracy rises a lot, it was the main cause. | code change + 1 Colab session |
| 4 | Have a teammate re-score section 4.1 | remove single-rater bias | ~15 min |
| 5 | Start recording the capture method in file names (`Chair_lidar_1.ply`) | needed for the LiDAR-vs-photo study | from the next scan on |
| 6 | Once the baseline is clean, scan local objects (rickshaw, thelagari, CNG) and run the same notebook | the actual research question | — |

---

## 8. How to reproduce

1. Open the notebook in Colab: https://colab.research.google.com/github/Md-Mosabbir/localpoint3d/blob/main/notebooks/01_pointllm_recognition_test.ipynb
2. *Runtime → Change runtime type → T4 GPU*.
3. Make sure `My Drive/Models` contains the 17 PLY files. Teammates: add a shortcut to the shared folder in *My Drive*.
4. Run all cells from top to bottom. Keep `ROTATION = "None"` and `ROTATIONS_TO_TRY = [ROTATION]`.
5. The batch cell writes `pointllm_<date>_<time>.csv` to `My Drive/localpoint3d/results/`.

Total time: about 35–40 minutes on a fresh runtime (≈5 min install, ≈10 min model download/load, ≈15–20 min for the 51 answers).
