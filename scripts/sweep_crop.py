"""Sweep the eye-patch width factor and see what it does to gaze consistency.

The reference implementation sets the patch to cover 1.5x the corner-to-corner distance,
but that was paired with dlib's 5-point landmarks. Our corners come from a 3DMM mesh,
which may sit differently, so the right factor is an empirical question.

Two metrics, neither perfect on its own:
  agreement  - angle between the two eyes' fused directions. Cheap, but a degenerate
               crop that always returns the same answer would score perfectly.
  on-camera  - for subjects who are looking into the lens, the angle away from straight
               ahead. Only meaningful for images tagged as such.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gaze_common  # noqa: E402
from gaze_common import (  # noqa: E402
    LEFT_EYE_CORNERS,
    RIGHT_EYE_CORNERS,
    FaceMesh,
    Stage,
    angle_between,
    detect_faces,
    eye_patch,
    face_input,
    fuse,
    gaze_input,
    gaze_output,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# subjects that are looking into the lens, so the true direction is ~straight ahead
LOOKING_AT_CAMERA = {"portrait-10001467", "portrait-6471976", "portrait-3827677"}
STRAIGHT_AHEAD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--factors", default="1.0,1.25,1.5,1.75,2.0,2.5,3.0")
    args = ap.parse_args()

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"))
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"))
    gm = Stage(os.path.join(args.models, "eyegaze.tflite"))
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    paths = []
    for p in args.images:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    # cache the per-image geometry once; only the crop changes in the sweep
    cached = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        net_in, transform = face_input(frame)
        faces = detect_faces(det.run(net_in), transform, w, h)
        if not faces:
            continue
        crop_in, box_used = FaceMesh.crop_input(frame, faces[0]["box"])
        if crop_in is None:
            continue
        lmk, pose = mesh.decode(mm.run(crop_in)["parameters_3dmm"], box_used)
        cached.append((os.path.splitext(os.path.basename(path))[0],
                       to_gray(frame), lmk, pose))
    print(f"{len(cached)} faces\n")

    factors = [float(v) for v in args.factors.split(",")]
    print(f"{'factor':>8}{'mean agreement':>17}{'mean on-camera err':>21}{'n on-cam':>10}")
    print("-" * 58)
    original = gaze_common.EYE_WIDTH_FACTOR
    best = None
    for fct in factors:
        gaze_common.EYE_WIDTH_FACTOR = fct
        agrees, oncam = [], []
        for tag, gray, lmk, pose in cached:
            vecs = {}
            for side, corners, flip in (("right", RIGHT_EYE_CORNERS, True),
                                        ("left", LEFT_EYE_CORNERS, False)):
                p, _, _ = eye_patch(gray, lmk, corners, flip=flip)
                if p is not None:
                    vecs[side] = fuse(
                        pose, gaze_output(gm.run(gaze_input(p)), flipped=flip)
                    )
            if len(vecs) == 2:
                agrees.append(angle_between(vecs["left"], vecs["right"]))
                if tag in LOOKING_AT_CAMERA:
                    mean_v = vecs["left"] + vecs["right"]
                    oncam.append(angle_between(mean_v, STRAIGHT_AHEAD))
        a = float(np.mean(agrees)) if agrees else float("nan")
        o = float(np.mean(oncam)) if oncam else float("nan")
        print(f"{fct:>8.2f}{a:>17.1f}{o:>21.1f}{len(oncam):>10}")
        score = a + o
        if best is None or score < best[1]:
            best = (fct, score, a, o)
    gaze_common.EYE_WIDTH_FACTOR = original
    print("-" * 58)
    if best:
        print(f"best combined: factor {best[0]:.2f}  "
              f"(agreement {best[2]:.1f} deg, on-camera {best[3]:.1f} deg)")
        print(f"reference implementation uses 1.50")


if __name__ == "__main__":
    main()
