# 05 · Local objects: are they out-of-distribution, and does the model read the shape?

**Plan (team, 2026-11-01):** test local objects (bodna, piri, …) with **multiple-choice questions** on PointLLM *and* a CLIP-style model, using **grey** point clouds so only the shape counts. Before that, check whether our scans really are **near-OOD** (a new kind of object that is close to things the model has seen).

---

## 2026-11-01 · Claude (2 research agents) · how to run the test

### What sources say

**Near-OOD vs far-OOD** ([OpenOOD, arXiv 2210.07242](https://arxiv.org/abs/2210.07242))
- Near-OOD: a new class from the same kind of data (*semantic shift*).
- Far-OOD: the class *and* the kind of data change.
- For us, phone scans vs Objaverse CAD models are a change of data (*covariate shift*), and bodna vs teapot is a change of class.

**MCQ pitfalls**
- LLMs prefer certain answer letters (A/B/C/D) whatever the options say: [PriDe, arXiv 2309.03882](https://arxiv.org/abs/2309.03882).
- MMBench's *CircularEval* asks once per rotation of the options and counts a question correct only if every rotation is right: [arXiv 2307.06281](https://arxiv.org/abs/2307.06281).
- A way around letter bias is to score each option's likelihood directly instead of letting the model write an answer.

**3D models that may ignore the shape**
- Blind text-only models matched full 3D-LLMs on a 3D question-answering benchmark: [arXiv 2603.23523](https://arxiv.org/abs/2603.23523).
- [arXiv 2502.15888](https://arxiv.org/abs/2502.15888) counts it as hallucination when a random other point cloud gets the same answer.
- Yes/no object probes in the style of 3D-POPE: [3D-GRAND, arXiv 2406.05132](https://arxiv.org/abs/2406.05132).

**PointLLM's own evaluation** ([arXiv 2308.16911](https://arxiv.org/abs/2308.16911))
- Prompts: "What is this?" and "This is an object of ".
- Accuracy for the 7B model: about 53 % on ModelNet40 and 55 % on Objaverse.
- ModelNet40 objects (which have no colour) were fed as **black** (rgb = 0), not grey.

**CLIP-style 3D models**
- **OpenShape PointBERT G14** ([HF](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb), [code](https://github.com/Colin97/OpenShape_code))
  - Reported zero-shot: Objaverse-LVIS 46.8 % and ModelNet40 84.4 %.
  - Its text side is OpenCLIP ViT-bigG-14.
  - Without colour it was trained with rgb = **0.4** (ULIP and Uni3D use the same value).
  - Precomputed Objaverse embeddings are available for nearest-neighbour search: [openshape-objaverse-embeddings](https://huggingface.co/datasets/OpenShape/openshape-objaverse-embeddings), about 1.5 GB.
- **ULIP-2, Uni3D:** harder to install on Colab (CUDA builds, huge text model).
- **PointLLM's encoder can't score text:** its Point-BERT was pretrained with ULIP-2, but no matching text head is released.

**OOD scores**
- MCM, the max softmax over class-name similarities: [arXiv 2211.13445](https://arxiv.org/abs/2211.13445).
- kNN distance in embedding space: [Sun et al. 2022, arXiv 2204.06507](https://arxiv.org/abs/2204.06507).

### What we measured: are our objects in the training text?

**PointLLM's training captions:**
- File: `PointLLM_brief_description_660K_filtered.json`, [HF RunsenXu/PointLLM](https://huggingface.co/datasets/RunsenXu/PointLLM), 658,577 captions.
- Counted: captions containing the whole word (case ignored).
- Re-counted by Claude on 2026-11-01.

| Word | Captions | | Word | Captions |
|---|---|---|---|---|
| bodna / lota / ewer | **0 / 0 / 0** | | piri | **0** |
| water pot | 0 | | stool | 2,101 |
| teapot | 2,980 | | footstool | 20 |
| kettle | 113 | | bench | 8,141 |
| jug | 1,138 | | rickshaw | 23 |
| pitcher | 21 | | cart | 869 |
| vase | 7,335 | | hand fan | 0 |
| clay pot | 619 | | chair | 17,243 |
| spout | 612 | | table | 35,994 |

**Objaverse-LVIS categories** (1,156 categories; objects per category):
- teapot 75, kettle 75, pitcher 73, water_jug 37, vase 106
- stool 56, footstool 29, handcart 48, tricycle 28, chair 453
- **no** ewer, lota or rickshaw

**Reading**
- Bodna and piri are **near-OOD**: the model never saw their names, but close relatives are common (teapot, jug and vase for bodna; stool and bench for piri).
- Chairs and tables are everywhere in training. That fits chairs being our best class so far (journal 02).

### Decided plan (to build as notebooks 03 and 04)

1. **OOD check first.**
   - Encode every scan.
   - Measure the distance from each local scan to our 17 everyday scans, which serve as the "in-distribution" reference from the same phone and cleaning.
   - Find the nearest Objaverse objects with OpenShape.
   - The nearest classes become the *shape-similar* wrong options.
2. **Grey MCQ, 4 options** (chance = 25 %):
   - the correct answer, described in plain English (the model can't know "bodna")
   - a shape look-alike (e.g. teapot)
   - a same-family object (e.g. vase)
   - an unrelated object
3. **PointLLM, scored by likelihood:**
   - Rank the options by how likely PointLLM finds each one. The model doesn't write an answer, so there are no empty answers and no letter bias.
   - Use 5 seeds.
   - Optionally, check with generated letters using CircularEval.
4. **Blind baseline:** the same options for a grey sphere and for random points. If the answer doesn't change, it comes from the language prior, not the shape.
5. **CLIP (OpenShape):** the same options, one shared template set for every label (fixes the "a 3D scan of…" bias from journal 04), rgb = 0.4.
6. **Statistics:** with 10 objects, at least 6 of 10 correct is better than chance (one-sided binomial test, p = 0.02).

**Built (2026-11-01):** `notebooks/03_pointllm_mcq_near_ood.ipynb`, options in `data/mcq_options.csv`.
- New code in `src/pointllm_helpers.py`: `score_answers`, `embed`, `fake_cloud`, and `colour="black"`.
- Tested locally with a fake model: every cell runs.
- Checked with PointLLM's real tokenizer: the scored answer tokens are exactly what the model would generate after "ASSISTANT:".
- **Not yet run on the GPU.**

**Open question:** which grey works best for PointLLM: 0.5 (what we use), 0 (black, like the paper) or 0.4? Run it once on the 17 everyday scans.
