"""Measure what dropping EyeGaze's unused `heatmaps` output is worth.

Stock EyeGaze returns (heatmaps, landmarks, gaze_pitchyaw). The heatmaps tensor is
1.57 MB and no application reads it. This times the stock and trimmed graphs on the
same input and checks the gaze they return is identical.
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    RIGHT_EYE_CORNERS,
    FaceMesh,
    Stage,
    detect_faces,
    eye_patch,
    face_input,
    gaze_input,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(ROOT, "test-data/portrait-6471976.jpg"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    h, w = frame.shape[:2]
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"))
    net_in, transform = face_input(frame)
    faces = detect_faces(det.run(net_in), transform, w, h)
    crop_in, box_used = FaceMesh.crop_input(frame, faces[0]["box"])
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"))
    lmk, _pose = mesh.decode(mm.run(crop_in)["parameters_3dmm"], box_used)
    patch, _, _ = eye_patch(to_gray(frame), lmk, RIGHT_EYE_CORNERS, flip=True)
    x = gaze_input(patch)

    variants = [
        ("stock (3 outputs)", os.path.join(args.models, "eyegaze.tflite")),
        ("trimmed (2 outputs)", os.path.join(args.models, "eyegaze_trimmed.tflite")),
    ]

    print(f"{'variant':<22}{'backend':<10}{'median ms':>11}{'output MB':>11}")
    print("-" * 56)
    results = {}
    for label, path in variants:
        for backend in (None, "htp"):
            st = Stage(path, backend=backend)
            for _ in range(5):
                st.run(x)
            ts = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                st.run(x)
                ts.append((time.perf_counter() - t0) * 1000.0)
            med = statistics.median(ts)
            out_mb = sum(float(np.prod(d["shape"])) * 4 / 1e6 for d in st.outs)
            results[(label, backend)] = (med, st.run(x))
            print(f"{label:<22}{(backend or 'cpu'):<10}{med:>11.2f}{out_mb:>11.3f}")
            del st

    print("-" * 56)
    for backend in (None, "htp"):
        a = results[("stock (3 outputs)", backend)][0]
        b = results[("trimmed (2 outputs)", backend)][0]
        print(f"{(backend or 'cpu'):<10} {a - b:+.2f} ms per eye ({100 * (a - b) / a:+.1f}%), "
              f"{2 * (a - b):+.2f} ms per frame at two eyes")

    def find_gaze(outs):
        for v in outs.values():
            arr = np.asarray(v).reshape(-1)
            if arr.size == 2:
                return arr
        return None

    g_stock = find_gaze(results[("stock (3 outputs)", "htp")][1])
    g_trim = find_gaze(results[("trimmed (2 outputs)", "htp")][1])
    if g_stock is not None and g_trim is not None:
        print(f"\ngaze stock   {np.degrees(g_stock)} deg")
        print(f"gaze trimmed {np.degrees(g_trim)} deg")
        print(f"max delta    {np.degrees(np.abs(g_stock - g_trim)).max():.8f} deg "
              f"(same weights, so this should be ~0)")


if __name__ == "__main__":
    main()
