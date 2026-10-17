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
