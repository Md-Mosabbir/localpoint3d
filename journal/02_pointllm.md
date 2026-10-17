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
