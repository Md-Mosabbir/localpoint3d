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
3. Run all 17 scans and compare with PointLLM scan by scan (`results/pointllm_open_questions/pointllm_2026-10-18_baseline.csv`).

---

## 2026-11-01 · Shahedul (run) + Claude (reading) · "MetaCLIP" on 11 local scans

**Files** (uploaded to `results/`, commit 17f26d3):
- `results/clip/metaclip_4_labels_3_runs.xlsx`: 4 English labels per object, the same test 3 times with the label order shuffled.
- `results/clip/metaclip_5_labels_with_description.xlsx`: 5 English labels, including a descriptive one ("folding book rest", "floor stool", "winnowing basket", …).
- `results/clip/metaclip_plus_local_name_latin.xlsx`: the same + the local name in Latin letters ("bodna", "rehal").
- `results/clip/metaclip_plus_local_name_bangla_script.xlsx`: the same + the local name in Bangla script (বদনা, রেহাল).

**Setup:** 11 scans (Bodna 1–2, Kula 1–2, Muura, Patil 1–2, Piri 3–4, Rehal, Wooden_Bench), 8192 points, `has_color` False.
**Model:** MetaCLIP, a **2D** CLIP model, on **depth-shaded pictures rendered** from the 8192-point scans. We know this from the title of Shahedul's chart, `results/clip/metaclip_4_labels_3_runs_chart.jpg`. It is not a 3D point encoder. **Still not recorded:** the text template and how many views were rendered.

**Top-1 pick, 4 English labels:**

| Object | Labels offered | Top-1 pick | Probability |
|---|---|---|---|
| bodna | kettle / teapot / cup / bottle | kettle (B1) / teapot (B2) | 0.43 / 0.55 |
| kula | basket / bowl / tray / plate | plate | 0.36–0.39 |
| muura | basket / flower pot / bucket / stool | **stool** | 0.99 |
| patil | pot / bowl / saucer / basin | bowl (P1) / saucer (P2) | 0.66 / 0.41 |
| piri | bench / chair / stool / table | **stool** | 0.45–0.80 |
| rehal | book stand / chair / small table / bench | small table; book stand 3rd | 0.42; 0.14 |
| wooden bench | bench / table / stool / chair | **bench** | 0.83 |

**Label order:** shuffling barely matters (runs 1–3 identical except Piri_3, Rehal and Wooden_Bench, ±0.03). That is expected for CLIP, since every label is scored on its own.

**With a descriptive label added:**
- "folding book rest" **wins for rehal** (0.39–0.47).
- "floor stool" **wins for piri** (0.45–0.59).
- "winnowing basket" 0.12–0.19 (not top), "earthenware pot" 0.07–0.12, "Toilet Wash Jug" 0.02–0.03.

**Local names:**

| Script | Result |
|---|---|
| Latin ("bodna", "rehal") | always low (0.01–0.13); "Kather Bench" wins 0.41, but it contains "Bench" |
| Bangla script | mostly low; কুলা is top for Kula_1 (0.29) but 3rd for Kula_2 |

The text encoder is mostly English, so treat Bangla-script results as unreliable.

**Compared with PointLLM's grey MCQ (journal 05):**

| | PointLLM (journal 05) | MetaCLIP |
|---|---|---|
| Both | patil → bowl, bodna → teapot/vase | patil → bowl, bodna → teapot/kettle |
| PointLLM worse | muura → chair, piri → bench/box | muura and piri → stool |
| Rehal | bench | small table (4 labels); folding book rest (when offered) |

CLIP seems better at "stool-like" shapes and can match a good English *description*. Neither recognises a local *name*.

**Caveat:** the option sets differ from `data/mcq_options.csv`, so this is not a like-for-like comparison yet.
