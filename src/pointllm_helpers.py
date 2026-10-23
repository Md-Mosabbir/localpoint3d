"""PointLLM helpers: turn a scan into the 8192-point input PointLLM expects, and ask it questions.

`preprocess` and `preview` only need numpy/matplotlib, so the notebook can show
"what the model will see" before the big model is loaded.
`load_model` and `ask` need torch + the PointLLM repo, so they import those lazily.
"""
import io

import numpy as np

from cleanup import guess_up_axis, remove_floor, remove_specks
from pc_io import load_points

MODEL_NAME = "RunsenXu/PointLLM_7B_v1.2"
N_POINTS = 8192  # PointLLM was trained on 8192 points per object
PAPER_PROMPTS = ["What is this?", "This is an object of ", "Caption this 3D model in detail."]
COLOURS = ["original", "grey"]  # "grey" = throw the scan's colours away (to test how much PointLLM relies on colour)
GREY = 0.5  # neutral grey, used for "grey" and for files that have no colour at all
FLOORS = ["kept", "removed"]  # "removed" = delete the floor/desk plane under the object (src/cleanup.py)
SPECKS = ["kept", "removed"]  # "removed" = delete small floating bits far from the object (src/cleanup.py)


# ---------------------------------------------------------------- preprocessing

def _rot_x(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


ROTATIONS = {
    "None": np.eye(3, dtype=np.float32),
    "+90 deg about X": _rot_x(90),   # Y-up scan -> Z-up
    # "-90 deg about X" turns a Z-up scan (our Scaniverse PLYs) into Y-up
    "-90 deg about X": _rot_x(-90),
    "180 deg about X": _rot_x(180),
}


def farthest_point_sample(xyz, k, seed=0):
    """Pick k points that are spread out as evenly as possible (greedy farthest-point sampling)."""
    rng = np.random.default_rng(seed)
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = rng.integers(len(xyz))
    dist = np.full(len(xyz), np.inf)
    for i in range(1, k):
        dist = np.minimum(dist, np.sum((xyz - xyz[chosen[i - 1]]) ** 2, axis=1))
        chosen[i] = np.argmax(dist)  # the point farthest from everything chosen so far
    return chosen


def grey_out(xyz):
    """One grey colour per point. Positions are not touched, so only the colour changes."""
    return np.full_like(xyz, GREY)


def preprocess(path, n_points=N_POINTS, rotation="None", seed=0, colour="original", floor="kept", specks="kept"):
    """Returns ((n_points, 6) float32 array [x y z r g b], has_color).

    xyz is centred and scaled into the unit sphere, rgb is in [0, 1].
    colour="grey" replaces every colour with the same grey. The same seed picks the same
    points either way, so "original" and "grey" differ ONLY in colour.
    has_color says whether the *file* had colour (False means "original" is grey anyway).
    floor="removed" deletes the floor plane first, using the file type to know which way is up.
    specks="removed" deletes stray specks first. Scaling uses the farthest point, so one speck
    metres away would otherwise shrink the object to a dot.
    """
    if colour not in COLOURS:
        raise ValueError(f"colour must be one of {COLOURS}, got {colour!r}")
    if floor not in FLOORS:
        raise ValueError(f"floor must be one of {FLOORS}, got {floor!r}")
    if specks not in SPECKS:
        raise ValueError(f"specks must be one of {SPECKS}, got {specks!r}")
    rng = np.random.default_rng(seed)
    xyz, rgb, _ = load_points(path)
    if specks == "removed":
        xyz, rgb, _ = remove_specks(xyz, rgb)
    if floor == "removed":
        xyz, rgb, _ = remove_floor(xyz, rgb, up_axis=guess_up_axis(path), seed=seed)
    if len(xyz) == 0:
        raise ValueError("No points found in file")
    has_color = rgb is not None
    if colour == "grey" or not has_color:
        rgb = grey_out(xyz)

    if len(xyz) > 40000:  # cheap random pre-reduction so farthest point sampling stays fast
        idx = rng.choice(len(xyz), 40000, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    if len(xyz) >= n_points:
        idx = farthest_point_sample(xyz, n_points, seed)
    else:  # too few points: repeat some at random
        idx = rng.choice(len(xyz), n_points, replace=True)
    xyz, rgb = xyz[idx], rgb[idx]

    xyz = xyz @ ROTATIONS[rotation].T
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.max(np.linalg.norm(xyz, axis=1))
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32), has_color


def preview(points, title=None):
    """Three side views (X-Y, X-Z, Y-Z) of a preprocessed cloud, returned as a PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    xyz, rgb = points[:, :3], points[:, 3:]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (a, b, name) in zip(axes, [(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]):
        ax.scatter(xyz[:, a], xyz[:, b], c=rgb, s=1)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------- model (GPU, Python 3.10 env only)

def load_model(model_name=MODEL_NAME):
    import torch
    from transformers import AutoTokenizer
    from pointllm.utils import disable_torch_init
    from pointllm.model import PointLLMLlamaForCausalLM

    dtype = torch.float16  # T4 has no bfloat16 support
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        # stream weights straight to the GPU to avoid needing ~14 GB of system RAM
        model = PointLLMLlamaForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, use_cache=True,
            torch_dtype=dtype, device_map={"": 0})
    except Exception as e:
        print("device_map loading failed, falling back to .cuda():", repr(e))
        torch.cuda.empty_cache()
        model = PointLLMLlamaForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=False, use_cache=True,
            torch_dtype=dtype).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, points, prompt, sample=False, seed=0):
    """One answer from PointLLM.

    PointLLM's point encoder picks a *random* start point when it groups the cloud
    (torch.randint in pointllm/model/pointbert/misc.py fps). Without a fixed seed the same
    input can give a different answer every call (seen 2026-10-21: 48 of 51 answers changed).
    Setting the seed makes each call repeatable; different seeds show how stable an answer is.
    """
    import torch
    from pointllm.conversation import conv_templates, SeparatorStyle
    from pointllm.model.utils import KeywordsStoppingCriteria

    cfg = model.get_model().point_backbone_config
    use_start_end = getattr(model.config, "mm_use_point_start_end", False)
    conv = conv_templates["vicuna_v1_1"].copy()

    # the point cloud is represented in the prompt by a run of special placeholder tokens
    patch = cfg["default_point_patch_token"] * cfg["point_token_len"]
    if use_start_end:
        qs = cfg["default_point_start_token"] + patch + cfg["default_point_end_token"] + "\n" + prompt
    else:
        qs = patch + "\n" + prompt
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    input_ids = torch.as_tensor(tokenizer([conv.get_prompt()]).input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    pc = torch.from_numpy(points).unsqueeze(0).to(model.dtype).cuda()

    kw = dict(do_sample=True, temperature=1.0, top_k=50, top_p=0.95) if sample else dict(do_sample=False)
    torch.manual_seed(seed)  # fixes PointLLM's random grouping (and sampling, if on)
    with torch.inference_mode():
        out = model.generate(input_ids, point_clouds=pc, max_length=2048,
                             stopping_criteria=[stopping], **kw)
    ans = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    if ans.endswith(stop_str):
        ans = ans[:-len(stop_str)].strip()
    return ans
