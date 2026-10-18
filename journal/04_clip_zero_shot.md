# 04 · CLIP-style zero-shot classification

**Idea:** a CLIP-style model scores how well a 3D input matches each text label in a list. "Zero-shot" means it was never trained on our label list, so we can give it any labels, including "rickshaw".

---

## 2026-10-18 · team · first single test (one cup scan)

**Not recorded yet, please fill in:** model name / checkpoint, which cup scan, whether the scan went in as a point cloud or as rendered images, and the notebook used.

Output, exactly as printed:

```
--- Zero-Shot Classification Results ---
a 3D scan of a chair: 47.99%
a three-dimensional photo of a coffee Cup: 37.09%
a 3D scan of a bottle: 9.26%
a photo of a coffee Cup: 3.01%
a 3D scan of a building: 2.65%
```

**What it shows**
- The top answer is wrong: chair (48 %) for a cup.
- The cup is in the list twice, with different wording: "three-dimensional photo" 37 % vs "photo" 3 %. Same object, a 12× difference from wording alone.
- If a label mentions 2D ("photo"), it loses confidence.

**Explanation (from our supervisor):** during training, every 3D object's text began with "a 3D scan of …". The model learned to favour labels that use that phrase, whatever the object.

So this list compared wordings as much as objects. It can't be read as "the model thinks the cup is a chair". Clutter in the scan may also contribute (see journal 02, 2026-10-18).

**How to run it fairly next time**
1. Use one template for every label: `a 3D scan of a {chair, cup, bottle, keyboard, table, building, …}`.
2. Better still, average over several templates. This is prompt ensembling, as in the CLIP paper (Radford et al. 2021, https://arxiv.org/abs/2103.00020).
3. Run all 17 scans and compare with PointLLM scan by scan (`results/pointllm_2026-10-18_baseline.csv`).
