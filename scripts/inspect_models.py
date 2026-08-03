"""Print each model's input and output tensors, before you write code around them.

Shapes and layouts are not guaranteed to match what a model card says, and in this
pipeline two of the three have a surprise in them: EyeGaze's input carries no channel
dimension, and its largest output is one no application reads.

    ./venv/bin/python scripts/inspect_models.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import Stage  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["face_det_lite", "facemap_3dmm", "eyegaze", "eyegaze_trimmed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("models_list", nargs="*", default=None)
    args = ap.parse_args()

    names = args.models_list or MODELS
    for name in names:
        path = os.path.join(args.models, f"{name}.tflite")
        if not os.path.exists(path):
            continue
        st = Stage(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"=== {name}   ({size_mb:.1f} MB on disk)")
        d = st.inp
        print(f"IN   {d['name'][:18]:<18} {str(d['shape']):<20} {d['dtype'].__name__}")
        total = 0.0
        for d in st.outs:
            mb = float(np.prod(d["shape"])) * 4 / 1e6
            total += mb
            note = f"  ({mb:.2f} MB)" if mb >= 0.01 else ""
            print(f"OUT  {d['name'][:18]:<18} {str(d['shape']):<20} "
                  f"{d['dtype'].__name__}{note}")
        print(f"     total output per inference: {total:.2f} MB\n")
        del st


if __name__ == "__main__":
    main()
