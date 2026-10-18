# localpoint3d

**Can 3D models trained on Western datasets (ModelNet40, ShapeNet, Objaverse) recognise Bangladeshi objects like a rickshaw or a thelagari?**

Junior design project. We scan local objects with an iPhone (Scaniverse), turn them into point clouds, and test existing 3D models on them. Everything runs on Google Colab.

**Team:** Shahedul Islam Shikder · Fahim Muntasir · Md Mosabbir

## Notebooks

| Notebook | What it does | Runtime | |
|---|---|---|---|
| [01 · PointLLM recognition test](notebooks/01_pointllm_recognition_test.ipynb) | Ask PointLLM-7B "what is this?" about our scans → CSV | T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Md-Mosabbir/localpoint3d/blob/main/notebooks/01_pointllm_recognition_test.ipynb) |
| [02 · LiDAR vs photogrammetry](notebooks/02_lidar_vs_photogrammetry.ipynb) | How different are two scans of the same object? Align + Chamfer/F-score | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Md-Mosabbir/localpoint3d/blob/main/notebooks/02_lidar_vs_photogrammetry.ipynb) |

Click a badge, then run the cells top to bottom. To save your edits back here: in Colab, *File → Save a copy in GitHub*.

## Folders

```
notebooks/   the Colab notebooks (one step per cell, explained)
src/         helper Python code the notebooks import
journal/     our research notes: what we read, tried, found
reports/     write-ups of finished experiments
data/        how to capture + name scans (the scans themselves live on Google Drive)
results/     CSV outputs we want to keep
archive/     old versions, for reference
```

## Results so far

| Date | Experiment | Result | Data |
|---|---|---|---|
| 2026-10-18 | PointLLM on 17 everyday scans | 9 / 17 scans named correctly | [report](reports/2026-10-18_pointllm_baseline.md) · [table (CSV)](results/pointllm_2026-10-18_baseline.csv) |
| 2026-10-18 | **Progress report 1** (scans, PointLLM, CLIP test, LiDAR vs photo reading) | failures come mostly from cluttered scans | [PDF](reports/2026-10-18_progress_report_1.pdf) |

## Roadmap

- [x] PointLLM runs on Colab and recognises a chair
- [ ] Scaniverse LiDAR vs photo mode: same object, compare (notebook 02)
- [ ] PointLLM on local objects (rickshaw, thelagari, CNG, …)
- [ ] Classic classifiers trained on ModelNet40 / ShapeNet (e.g. PointNet++, DGCNN) on the same scans
