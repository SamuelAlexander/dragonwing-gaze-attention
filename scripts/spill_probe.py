"""Read each model's VTCM spill from the QNN delegate's own preparation log.

A graph that spills is moving activations to DDR because they do not fit in the NPU's
on-chip VTCM, which makes it bandwidth-bound rather than arithmetic-bound. That is the
difference between a model that is slow because it is big and one that is slow because
it is shaped badly for the hardware.

The delegate prints the summary to fd 2 during graph preparation, so this runs each
model in a child process and captures it.

    ./venv/bin/python scripts/spill_probe.py
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import sys, numpy as np
sys.path.insert(0, {scripts!r})
from gaze_common import Stage
st = Stage({path!r}, backend="htp")
shape = tuple(int(v) for v in st.input_shape)
st.run(np.zeros(shape, dtype=np.float32))
"""

MODELS = [
    ("face_det_lite", "face_det_lite.tflite"),
    ("facemap_3dmm", "facemap_3dmm.tflite"),
    ("eyegaze", "eyegaze.tflite"),
    ("eyegaze_trimmed", "eyegaze_trimmed.tflite"),
]

FIELDS = ("spill_bytes", "fill_bytes", "write_total_bytes", "read_total_bytes")


def main():
    scripts = os.path.join(ROOT, "scripts")
    models = os.path.join(ROOT, "models")
    print(f"{'model':<20}{'spill MB':>10}{'fill MB':>10}{'read MB':>10}{'write MB':>10}")
    print("-" * 60)
    for name, fname in MODELS:
        path = os.path.join(models, fname)
        if not os.path.exists(path):
            continue
        code = CHILD.format(scripts=scripts, path=path)
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        blob = proc.stdout + proc.stderr
        vals = {}
        for f in FIELDS:
            m = re.findall(rf"{f}=(\d+)", blob)
            vals[f] = sum(int(v) for v in m) if m else None
        if vals["spill_bytes"] is None:
            print(f"{name:<20}   no bandwidth summary in the prep log")
            continue
        print(f"{name:<20}"
              f"{vals['spill_bytes'] / 1e6:>10.2f}"
              f"{vals['fill_bytes'] / 1e6:>10.2f}"
              f"{vals['read_total_bytes'] / 1e6:>10.2f}"
              f"{vals['write_total_bytes'] / 1e6:>10.2f}")
    print("-" * 60)
    print("Nonzero spill means the graph does not fit in VTCM and is paying DDR traffic")
    print("per inference. Compare against each model's measured latency.")


if __name__ == "__main__":
    main()
