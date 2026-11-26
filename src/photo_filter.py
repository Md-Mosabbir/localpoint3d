"""Automatic first pass over collected web photos, before anyone looks at them.

    python src/photo_filter.py data/web_images_far_ood --only rickshaw

Two checks per photo (CPU is fine, about 1-2 s per photo):

1. CLIP (an image-text model): does the photo match "a photo of <class>" better than every decoy?
   Decoys = things web searches often return instead (a person, a crowd, a poster, a drawing, a map,
   a room) plus the OTHER far-OOD classes. If a decoy wins, the photo is dropped.
2. rembg (background removal): the cut-out tells us where the main object is. We keep the photo only if
   - there is ONE main object (no second big piece),
   - it fills about 10-80 % of the picture (not a speck, not a close-up),
   - it is not cut off at the picture edge.

Dropped photos are moved to <out_dir>/_rejected/<class>/ (never deleted) with the reason in reasons.csv.
Every checked photo gets a row in <out_dir>/<class>/filter_log.csv, so running again only checks new photos.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_images import log_reason, numbered_pictures  # noqa: E402

# Smaller, faster models by default (this is a coarse first pass; a person checks the sheets afterwards).
# Bigger, slower, a bit more accurate: --clip openai/clip-vit-large-patch14 --rembg u2net
CLIP_MODEL = "openai/clip-vit-base-patch16"
REMBG_MODEL = "u2netp"

# What "a photo of a <class>" means for each class: plain words CLIP knows (it doesn't know "leguna").
CLASS_PROMPTS = {
    "rickshaw": "a cycle rickshaw",
    "van": "a pedal rickshaw van with a flat wooden cargo bed",
    "cng": "a green CNG auto rickshaw",
    "leguna": "a small covered pickup truck used as a shared passenger van",
    "tubewell": "a tubewell hand pump",
    "sheel_pata": "a flat grinding stone slab with a stone roller",
    "thela_gari": "a wooden hand-pushed cart with two wheels",
}
DECOY_PROMPTS = ["a photo of a person", "a photo of a crowd of people", "a photo of text or a poster",
                 "a drawing or cartoon", "a photo of a map", "a photo of a room interior"]

MASK_THRESHOLD = 128        # rembg's mask is 0-255; above this = object
MIN_FILL, MAX_FILL = 0.10, 0.80
SECOND_OBJECT = 0.30        # a second piece bigger than 30 % of the main one = "two objects"
SPECK = 0.005               # pieces smaller than 0.5 % of the picture are ignored
EDGE_TOUCH = 0.05           # main object covers > 5 % of an edge's length = cut off there
LOG_COLUMNS = ["file", "clip_best", "p_class", "p_best", "objects", "fill", "cut_edges", "decision", "reason"]


# ---------------------------------------------------------------- CLIP

class ClipJudge:
    """Scores one photo against the class prompt and all decoy prompts."""

    def __init__(self, name, model_name=CLIP_MODEL):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        others = [f"a photo of {p}" for c, p in CLASS_PROMPTS.items() if c != name]
        self.labels = [f"a photo of {CLASS_PROMPTS[name]}"] + DECOY_PROMPTS + others
        with torch.no_grad():
            text = self.processor(text=self.labels, return_tensors="pt", padding=True)
            t = self.model.get_text_features(**text)
            self.text = t / t.norm(dim=-1, keepdim=True)      # unit length, so a dot product = similarity

    def __call__(self, image):
        """(best label, probability of the class label, probability of the best label)."""
        with self.torch.no_grad():
            v = self.model.get_image_features(**self.processor(images=image, return_tensors="pt"))
            v = v / v.norm(dim=-1, keepdim=True)
            probs = (self.model.logit_scale.exp() * v @ self.text.T).softmax(dim=-1)[0].numpy()
        best = int(probs.argmax())
        return self.labels[best], float(probs[0]), float(probs[best])


# ---------------------------------------------------------------- rembg

def object_checks(image, session):
    """From the background-removal mask: number of big pieces, fill of the main one, edges it touches."""
    from rembg import remove
    mask = np.asarray(remove(image, session=session, only_mask=True)) > MASK_THRESHOLD
    pieces, n = ndimage.label(mask)                      # number every connected blob of object pixels
    if n == 0:
        return 0, 0.0, []
    sizes = np.bincount(pieces.ravel())[1:]              # pixels per blob (blob 0 = background)
    sizes_sorted = np.sort(sizes)[::-1]
    big = sizes_sorted[sizes_sorted > SPECK * mask.size]
    main = pieces == (np.argmax(sizes) + 1)
    objects = 1 + int(np.sum(big[1:] > SECOND_OBJECT * big[0])) if len(big) else 0
    edges = {"top": main[0, :], "bottom": main[-1, :], "left": main[:, 0], "right": main[:, -1]}
    cut = [side for side, line in edges.items() if line.mean() > EDGE_TOUCH]
    return objects, float(main.mean()), cut


# ---------------------------------------------------------------- one class

def filter_class(out_dir, name, clip_model=CLIP_MODEL, rembg_model=REMBG_MODEL):
    folder = Path(out_dir) / name
    log_path = folder / "filter_log.csv"
    done = set()
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, newline="", encoding="utf-8") as fh:
            done = {row["file"] for row in csv.DictReader(fh)}
    todo = [path for _, path in numbered_pictures(folder) if path.name not in done]
    print(f"{name}: {len(todo)} photos to check ({len(done)} checked before)", flush=True)
    if not todo:
        return

    from rembg import new_session
    judge, session = ClipJudge(name, clip_model), new_session(rembg_model)
    rejected_dir = Path(out_dir) / "_rejected" / name
    rejected_dir.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    counts = {}
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if new_log:
            writer.writeheader()
        for i, path in enumerate(todo, 1):
            image = Image.open(path).convert("RGB")
            best, p_class, p_best = judge(image)
            objects, fill, cut = object_checks(image, session)
            if best != judge.labels[0]:
                reason = f"clip prefers '{best.replace('a photo of ', '')}'"
            elif objects != 1:
                reason = "no clear object" if objects == 0 else f"{objects} objects"
            elif not MIN_FILL <= fill <= MAX_FILL:
                reason = f"object fills {fill:.0%}"
            elif cut:
                reason = "cut off at " + "+".join(cut)
            else:
                reason = ""
            decision = "drop" if reason else "keep"
            writer.writerow({"file": path.name, "clip_best": best, "p_class": f"{p_class:.3f}",
                             "p_best": f"{p_best:.3f}", "objects": objects, "fill": f"{fill:.3f}",
                             "cut_edges": "+".join(cut), "decision": decision, "reason": reason})
            fh.flush()             # write each row at once, so an interrupted run can carry on where it stopped
            if reason:
                shutil.move(str(path), str(rejected_dir / path.name))
                log_reason(out_dir, name, path.name, "auto", reason)
            key = reason.split(" '")[0].split(" at ")[0] if reason else "kept"
            counts[key] = counts.get(key, 0) + 1
            print(f"\r  {i}/{len(todo)}", end="", flush=True)
    print()
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:20s} {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir")
    parser.add_argument("--only", nargs="*", default=list(CLASS_PROMPTS), help="classes to check")
    parser.add_argument("--clip", default=CLIP_MODEL, help="Hugging Face CLIP model")
    parser.add_argument("--rembg", default=REMBG_MODEL, help="rembg model (u2netp = small and fast)")
    args = parser.parse_args()
    for name in args.only:
        filter_class(args.out_dir, name, args.clip, args.rembg)


if __name__ == "__main__":
    main()
