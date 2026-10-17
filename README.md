# bd-pointcloud

**Can 3D models trained on Western datasets (ModelNet40, ShapeNet, Objaverse) recognise Bangladeshi objects like a rickshaw or a thelagari?**

Junior design project. We scan local objects with an iPhone (Scaniverse), turn them into point clouds, and test existing 3D models on them. Everything runs on Google Colab.

**Team:** Shahedul Islam Shikder · Fahim Muntasir · Md Mosabbir

## Notebooks

| Notebook | What it does | Runtime |
|---|---|---|
| [01 · PointLLM recognition test](notebooks/01_pointllm_recognition_test.ipynb) | Ask PointLLM-7B "what is this?" about our scans → CSV | T4 GPU |
| [02 · LiDAR vs photogrammetry](notebooks/02_lidar_vs_photogrammetry.ipynb) | How different are two scans of the same object? Align + Chamfer/F-score | CPU |

Open a notebook on GitHub and click the **Open in Colab** badge at the top. Then run the cells top to bottom.

## Folders

```
notebooks/   the Colab notebooks (one step per cell, explained)
src/         helper Python code the notebooks import
journal/     our research notes: what we read, tried, found
data/        how to capture + name scans (the scans themselves live on Google Drive)
results/     CSV outputs we want to keep
archive/     old versions, for reference
```

## Roadmap

- [x] PointLLM runs on Colab and recognises a chair
- [ ] Scaniverse LiDAR vs photo mode: same object, compare (notebook 02)
- [ ] PointLLM on local objects (rickshaw, thelagari, CNG, …)
- [ ] Classic classifiers trained on ModelNet40 / ShapeNet (e.g. PointNet++, DGCNN) on the same scans
