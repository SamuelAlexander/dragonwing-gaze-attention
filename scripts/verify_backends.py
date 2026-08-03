"""Per-model CPU vs NPU faithfulness, on byte-identical inputs.

Feeding each model the same tensor separates two things the end-to-end diff conflates:
whether the delegate computes the same answer, and whether tiny differences amplify as
they pass down a three-stage chain.

    python scripts/verify_backends.py test-data/portrait-6471976.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    LEFT_EYE_CORNERS,
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


def report(name, a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    absd = np.abs(a - b)
    rng = float(a.max() - a.min()) or 1.0
    print(f"  {name:<18} max |d| {absd.max():.6f}   mean |d| {absd.mean():.6f}   "
          f"max as % of range {100 * absd.max() / rng:.4f}%")
    return absd.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    h, w = frame.shape[:2]
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    paths = {
        "face_det_lite": os.path.join(args.models, "face_det_lite.tflite"),
        "facemap_3dmm": os.path.join(args.models, "facemap_3dmm.tflite"),
        "eyegaze": os.path.join(args.models, "eyegaze.tflite"),
    }
    cpu = {k: Stage(p) for k, p in paths.items()}
    npu = {k: Stage(p, backend="htp") for k, p in paths.items()}

    print(f"image: {args.image}  {w}x{h}\n")

    # --- stage 1 on identical input
    net_in, transform = face_input(frame)
    print("stage 1  face_det_lite (identical input tensor)")
    o_cpu, o_npu = cpu["face_det_lite"].run(net_in), npu["face_det_lite"].run(net_in)
    for k in o_cpu:
        report(k, o_cpu[k], o_npu[k])

    # Most of the 60x80 bbox grid is background, where the regression is unsupervised
    # and free to disagree. What matters is the cell the detection actually comes from.
    hm_cpu = 1.0 / (1.0 + np.exp(-o_cpu["heatmap"][0, :, :, 0]))
    cy, cx = np.unravel_index(int(np.argmax(hm_cpu)), hm_cpu.shape)
    print(f"  at the winning cell ({cy},{cx}), which is what the box is decoded from:")
    print(f"    heatmap  cpu {hm_cpu[cy, cx]:.6f}  npu "
          f"{1.0 / (1.0 + np.exp(-o_npu['heatmap'][0, cy, cx, 0])):.6f}")
    print(f"    bbox     cpu {np.round(o_cpu['bbox'][0, cy, cx, :], 4)}")
    print(f"             npu {np.round(o_npu['bbox'][0, cy, cx, :], 4)}")
    print(f"    bbox max delta at peak: "
          f"{np.abs(o_cpu['bbox'][0, cy, cx, :] - o_npu['bbox'][0, cy, cx, :]).max():.4f} "
          f"(x8 stride = px in the 640x480 input)")

    f_cpu = detect_faces(o_cpu, transform, w, h)
    f_npu = detect_faces(o_npu, transform, w, h)
    if f_cpu and f_npu:
        print(f"    decoded box delta: "
              f"{max(abs(a - b) for a, b in zip(f_cpu[0]['box'], f_npu[0]['box'])):.3f} px "
              f"in the {w}x{h} frame")

    # the CPU's boxes drive everything downstream, for both backends
    faces = f_cpu
    if not faces:
        sys.exit("no face")
    crop_in, box_used = FaceMesh.crop_input(frame, faces[0]["box"])

    print("\nstage 2  facemap_3dmm (identical input tensor)")
    p_cpu = cpu["facemap_3dmm"].run(crop_in)["parameters_3dmm"]
    p_npu = npu["facemap_3dmm"].run(crop_in)["parameters_3dmm"]
    report("parameters_3dmm", p_cpu, p_npu)

    lmk_cpu, pose_cpu = mesh.decode(p_cpu, box_used)
    lmk_npu, pose_npu = mesh.decode(p_npu, box_used)
    print(f"  -> landmarks max delta {np.abs(lmk_cpu - lmk_npu).max():.4f} px")
    print(f"  -> head pose max delta {np.degrees(np.abs(pose_cpu - pose_npu)).max():.4f} deg")

    print("\nstage 3  eyegaze (identical input tensor)")
    gray = to_gray(frame)
    for side, corners, flip in (("right", RIGHT_EYE_CORNERS, True),
                                ("left", LEFT_EYE_CORNERS, False)):
        patch, _, _ = eye_patch(gray, lmk_cpu, corners, flip=flip)
        if patch is None:
            continue
        x = gaze_input(patch)
        g_cpu = cpu["eyegaze"].run(x)
        g_npu = npu["eyegaze"].run(x)
        print(f"  {side} eye:")
        for k in ("gaze_pitchyaw", "landmarks", "heatmaps"):
            if k in g_cpu:
                d = report("  " + k, g_cpu[k], g_npu[k])
                if k == "gaze_pitchyaw":
                    print(f"      = {np.degrees(d):.4f} deg")

    print("\nInterpretation: per-model deltas are the delegate's own error. Compare with")
    print("the end-to-end delta from check_chain.py --compare to see how much a chain")
    print("amplifies them.")


if __name__ == "__main__":
    main()
