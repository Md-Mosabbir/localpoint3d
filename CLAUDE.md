# CLAUDE.md

## Project

Junior design project (team of 3: Shahedul Islam Shikder, Fahim Muntasir, Md Mosabbir).
Question: can 3D models trained on ModelNet40 / ShapeNet / Objaverse recognise **Bangladeshi objects** (rickshaw, thelagari, CNG, …)?
Pipeline: iPhone scan (**Scaniverse**, Mesh export as PLY/OBJ, *not* Gaussian splats) → point cloud → model (PointLLM now; ModelNet40/ShapeNet classifiers later).

Current tasks, in order:
1. Scaniverse LiDAR vs photogrammetry: are the point clouds similar? (notebook 02, journal 01)
2. Does PointLLM recognise our scans? Chair first, then local objects (notebook 01, journal 02)

## The user is learning

The user wants to **learn Python and the ML/3D concepts as they go**. So:
- Explain *why*, not just *what*, briefly. Introduce new concepts in one or two sentences, with no lectures.
- In notebooks, add a short `> 🧠` note when a cell uses a Python/tool concept a beginner may not know.
- Prefer readable code over clever code. Small named functions, plain numpy where reasonable.

## Notebook conventions

- Notebooks run on **Google Colab**. Every notebook has an *Open in Colab* badge and works top to bottom on a fresh runtime.
- **One step per cell.** Each code cell has a markdown cell before it saying what it does and why. Descriptive but not wordy.
- Settings the user is expected to change go in **one settings cell** near the top.
- Reusable code goes in `src/`, and notebooks import it after cloning the repo. Exception: maths we want to *learn from* (e.g. ICP, metrics in notebook 02) stays visible in notebook cells.
- PointLLM needs Python 3.10 + torch 2.0.1, so it runs in a `uv` venv at `/content/venv` as a background worker (`src/pointllm_worker.py`, JSON lines over stdin/stdout). Don't import torch/pointllm in Colab's own kernel.
- Keep `pointllm_helpers.preprocess`/`preview` free of torch so they work on CPU.

## Repo layout

- `notebooks/`: Colab notebooks (`NN_name.ipynb`)
- `src/pc_io.py`: load PLY/OBJ/GLB/OFF → numpy xyz + rgb; scan-name parsing
- `src/pointllm_helpers.py`, `src/pointllm_worker.py`: PointLLM preprocessing + inference
- `journal/`: research log, one Markdown file per topic, dated entries. **Update it whenever we research or measure something.** Keep sources (links) separate from our own measurements.
- `data/README.md`: capture protocol + file naming `<object>_<method>_<take>.<ext>` (method = `lidar` | `photo`)
- `results/`: CSVs worth keeping. Scans themselves are on Google Drive (`MyDrive/Models`, flat list like `Chair_1.ply`; local copy in `data/Models/`), never in git.
- `archive/`: old notebooks, read-only reference

## Facts worth remembering

- **No local GPU.** The laptop is CPU-only. Anything needing a GPU (PointLLM, training classifiers) runs on Colab. Local work = loading/previewing scans, comparison maths, tests.
- PointLLM latest weights: `RunsenXu/PointLLM_7B_v1.2`. PointLLM-V2 (TPAMI 2025) weights not released as of 2026-10. Re-check before big runs.
- PointLLM input: 8192 points, xyz in unit sphere, rgb in [0, 1]. Free T4 works in float16.
- iPhone/Scaniverse scans are Y-up and in metres.
- Google Colab CLI (`uv tool install google-colab-cli`, `colab run --gpu T4 …`) exists but isn't set up. It needs the user to log in once.

## Style

- README stays short and simple. Detail goes in `journal/`.
- Don't commit scans, model weights or large outputs.
