"""Background process that keeps PointLLM loaded and answers questions.

Why a separate process? PointLLM needs Python 3.10 + old library versions, but the
Colab notebook itself runs a newer Python. So the notebook starts this script with
the Python 3.10 environment and talks to it through stdin/stdout, one JSON per line:

  notebook -> worker : {"path": "...", "prompt": "...", "rotation": "None", "colour": "original",
                        "floor": "kept", "specks": "kept", "seed": 0, "sample": false}
  worker -> notebook : @@RESULT@@{"answer": "...", "has_color": true}
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import pointllm_helpers` works
sys.path.insert(0, os.environ.get("POINTLLM_DIR", "/content/PointLLM"))  # so `import pointllm` works

from pointllm_helpers import load_model, preprocess, ask  # noqa: E402

model, tokenizer = load_model()
print("@@READY@@", flush=True)

cache = {}  # (path, rotation, colour, floor, specks) -> preprocessed points, so 3 prompts about one scan only sample it once

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        key = (req["path"], req.get("rotation", "None"), req.get("colour", "original"), req.get("floor", "kept"),
               req.get("specks", "kept"))
        if key not in cache:
            cache[key] = preprocess(key[0], rotation=key[1], colour=key[2], floor=key[3], specks=key[4])
        pts, has_color = cache[key]
        ans = ask(model, tokenizer, pts, req["prompt"], req.get("sample", False), req.get("seed", 0))
        res = {"answer": ans, "has_color": has_color}
    except Exception:
        res = {"error": traceback.format_exc()}
    print("@@RESULT@@" + json.dumps(res), flush=True)
