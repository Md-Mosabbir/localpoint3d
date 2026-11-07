# CLAUDE.md

## Project

Junior design project (team of 3: Shahedul Islam Shikder, Fahim Muntasir, Md Mosabbir).
Question: can 3D models trained on ModelNet40 / ShapeNet / Objaverse recognise **Bangladeshi objects** (rickshaw, thelagari, CNG, …)?
Pipeline: iPhone scan (**Scaniverse**, Mesh export as PLY/OBJ, *not* Gaussian splats) → point cloud → model (PointLLM now; ModelNet40/ShapeNet classifiers later).

Current tasks, in order:
1. Scaniverse LiDAR vs photogrammetry: are the point clouds similar? (notebook 02, journal 01)
2. Does PointLLM recognise our scans? Chair first, then local objects (notebook 01, journal 02)
3. Local objects (bodna, piri, …) as **near-OOD**: grey multiple choice + blind baselines + OOD check (notebook 03, journal 05).
4. **Far-OOD** (tubewell, sheel pata, rickshaw) vs near-OOD vs everyday: fingerprint distance, multiple choice, all-labels confusion matrix, confidence, and an auto-written report.md + report.pdf per run (notebook 04, journal 05). CLIP with the same options file comes later as notebook 05.

## The user is learning

The user wants to **learn Python and the ML/3D concepts as they go**. So:
- Explain *why*, not just *what*, briefly. Introduce new concepts in one or two sentences, with no lectures.
- In notebooks, add a short `> 🧠` note when a cell uses a Python/tool concept a beginner may not know.
- Prefer readable code over clever code. Small named functions, plain numpy where reasonable.

## Notebook conventions

- Notebooks run on **Google Colab**. Every notebook has an *Open in Colab* badge and works top to bottom on a fresh runtime.
- **One step per cell.** Each code cell has a markdown cell before it saying what it does and why. Descriptive but not wordy.
- Settings the user is expected to change go in **one settings cell** near the top. Notebook 01: the `⚙️ Settings` cell right after the intro holds SCANS_DIR, RUN_NAME, ONLY_SCANS, ROTATION/COLOUR/FLOOR/SEED and the batch `..._TO_TRY` lists. Don't define settings anywhere else.
- **Don't hide output**: no `pip -q` / quiet flags. The user wants to see what installs do.
- Setup cells must be safe to run twice in one Colab session (e.g. skip creating `/content/venv` if it exists).
- Reusable code goes in `src/`, and notebooks import it after cloning the repo. Exception: maths we want to *learn from* (e.g. ICP, metrics in notebook 02) stays visible in notebook cells.
- PointLLM needs Python 3.10 + torch 2.0.1, so it runs in a `uv` venv at `/content/venv` as a background worker (`src/pointllm_worker.py`, JSON lines over stdin/stdout). Don't import torch/pointllm in Colab's own kernel.
- Keep `pointllm_helpers.preprocess`/`preview` free of torch so they work on CPU.
- Greying out (colour test) happens in memory: `preprocess(..., colour="grey")`. Don't write grey copies of scans to Drive.
- Floor removal: `preprocess(..., floor="removed")` → `src/cleanup.py` (RANSAC, horizontal planes only, 1 cm threshold, up-axis guessed from file type). Off by default.
- Notebook 03 settings cell holds LOCAL_DIR, EVERYDAY_DIR, COLOUR, OPTIONS_FILE, SEEDS, BLIND_INPUTS, SCORE_BY, CIRCULAR_EVAL, CAPTION_WORDS. MCQ answers are scored by log-likelihood (`pointllm_helpers.score_answers`, worker task "score"), not generated. Compare by plain `log_prob` (per-char normalisation favoured the longest option, 2026-10-31); analysis lives in `src/mcq_analysis.py`; keep options 1–2 words. OOD fingerprints = mean of Point-BERT's 513 tokens (`embed`, task "embed"). Blind inputs: worker path `"fake:sphere"` / `"fake:noise"`.
- Notebook 04 settings: EVERYDAY_DIR / NEAR_DIR / FAR_DIR, GROUP_OVERRIDES (object -> group, wins over the folder; a file name found in two folders is used once, far first), SEEDS, CONFUSION_SEEDS. Each run writes one folder `RESULTS_DIR/<RUN_NAME>_<date>/` with report.md, report.pdf, figures/, tables/.
- Speck removal: `preprocess(..., specks="removed")` → `cleanup.remove_specks` (10 cm voxels, connected pieces < 1 % of points dropped). Needed because unit-sphere scaling uses the farthest point: one speck metres away shrinks the object to a dot. Notebook 01 setting `SPECKS` (default "removed").

## Repo layout

- `notebooks/`: Colab notebooks (`NN_name.ipynb`)
- `src/pc_io.py`: load PLY/OBJ/GLB/OFF → numpy xyz + rgb; scan-name parsing
- `src/pointllm_helpers.py`, `src/pointllm_worker.py`: PointLLM preprocessing + inference
- `src/charts.py`: presentation charts (fixed colour per answer type, colour-blind checked); notebook 03 saves PNGs to `RESULTS_DIR/figures_<run>/` and can export tables to Google Sheets
- `data/mcq_options.csv`: answer options per object, shared by the PointLLM and CLIP MCQ notebooks
- `src/ood_analysis.py`: notebook 04 maths (kNN distance per group, all-labels predictions, confusion matrix, softmax confidence); `src/report_writer.py`: one content list -> report.md + report.pdf (reportlab)
- `journal/`: research log, one Markdown file per topic, dated entries. **Update it whenever we research or measure something.** Keep sources (links) separate from our own measurements.
- `data/README.md`: capture protocol + file naming `<object>_<method>_<take>.<ext>` (method = `lidar` | `photo`)
- `results/`: tables worth keeping, one folder per kind of test (`pointllm_open_questions/`, `pointllm_multiple_choice/`, `clip/`), indexed in `results/README.md`. No duplicates: raw Colab exports stay on Drive. Scans themselves are on Google Drive (`MyDrive/Models`, flat list like `Chair_1.ply`; local copy in `data/Models/`), never in git. Hand-cleaned copies: `MyDrive/Models_Cleaned` / `data/Models_Cleaned/`, named `Chair_01_cleaned.ply` (parse_scan_name → version=cleaned, take 1).
- `archive/`: old notebooks, read-only reference

## Facts worth remembering

- **No local GPU.** The laptop is CPU-only. Anything needing a GPU (PointLLM, training classifiers) runs on Colab. Local work = loading/previewing scans, comparison maths, tests.
- PointLLM latest weights: `RunsenXu/PointLLM_7B_v1.2`. PointLLM-V2 (TPAMI 2025) weights **not released** (re-checked 2026-10-21: README TODO still open, no V2 model on Hugging Face; only V2 training annotations are out). Re-check before big runs.
- PointLLM input: 8192 points, xyz in unit sphere, rgb in [0, 1]. Free T4 works in float16.
- **PointLLM is random even with greedy decoding**: its PointBERT encoder starts farthest-point sampling at `torch.randint` (pointllm/model/pointbert/misc.py). `ask()` sets `torch.manual_seed(seed)`; use `SEEDS_TO_TRY = [0, 1, 2]` to measure stability. Runs before 2026-10-21 had no seed.
- Scaniverse **PLY point-cloud** exports are **Z-up** and in metres (checked on all 17 scans, 2026-10-18; floor lies in the x-y plane). We first assumed Y-up (ARKit convention), which was wrong for these files. Scaniverse **OBJ** exports are **Y-up** (floor in the x-z plane, seen in Colab previews 2026-10-21). So PLY and OBJ reach PointLLM in different orientations unless rotated. `-90 deg about X` turns Z-up into Y-up.
- Google Colab CLI (`uv tool install google-colab-cli`, `colab run --gpu T4 …`) exists but isn't set up. It needs the user to log in once.

## Style

- README stays short and simple. Detail goes in `journal/`.
- Don't commit scans, model weights or large outputs.
