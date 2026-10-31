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
| 2026-10-21 | Colour vs grey, floor kept vs removed (204 answers) | colour matters (24 → 13 of 51 without it); removing the floor only helps grey scans; PointLLM is random without a seed | [report](reports/2026-10-21_colour_and_floor_test.md) · [CSV kept](results/original_grey_floor_kept.csv) · [CSV removed](results/original_grey_floor_removed.csv) |
| 2026-10-23 | **Hand-cleaned scans**, colour vs grey, 3 seeds (306 answers) | cleaning: 24 → 33 (colour) and 13 → 32 (grey) of 51; 11 of 17 scans recognised; keyboards still fail | [report](reports/2026-10-23_cleaned_scans.md) · [CSV](results/pointllm_cleaned_2026-10-23.csv) |
| 2026-10-25 | **Bangladeshi objects** (10 cleaned scans, 7 objects; 180 answers) | 13 % (colour) / 20 % (grey) of answers correct vs 65 % / 58 % for everyday objects; only the matir cup is recognised; rehal, kula, piri get the right shape but the wrong name | [report](reports/2026-10-25_local_objects.md) · [CSV](results/pointllm_local_objects_2026-10-25.csv) |
| 2026-10-31 | **Near-OOD multiple choice**, grey, 8 local objects + 17 everyday (1,340 scores) | everyday 15 / 17 correct (ball/noise 1 / 8); local **0 / 14**: picks the nearest familiar object (rehal → bench, patil → bowl, bodna → vase); the local names never appear in training | [report](reports/2026-11-01_near_ood_mcq.md) · [CSV](results/pointllm_mcq_near_ood_2026-10-31.csv) |
| 2026-11-02 | **Mid-project report** (problem statement update, everything so far, week-by-week plan) | PDF for class | [PDF](reports/2026-11-02_mid_project_report.pdf) |

## Roadmap

- [x] PointLLM runs on Colab and recognises a chair
- [ ] Scaniverse LiDAR vs photo mode: same object, compare (notebook 02)
- [x] PointLLM on small local objects (bodna, kula, piri, …): mostly not recognised
- [ ] PointLLM on local vehicles (rickshaw, thelagari, CNG, …)
- [ ] Classic classifiers trained on ModelNet40 / ShapeNet (e.g. PointNet++, DGCNN) on the same scans
