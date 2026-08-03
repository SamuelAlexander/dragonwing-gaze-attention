"""Dump what the pipeline actually sees: face box, landmarks, and both eye patches.

    python scripts/debug_crops.py test-data/portrait-6471976.jpg -o results/debug
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    LEFT_EYE_CORNERS,
    LEFT_EYE_IDX,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_IDX,
    FaceMesh,
    Stage,
    detect_faces,
    eye_patch,
    face_input,
    gaze_input,
    gaze_output,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIP_RIGHT = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "results/debug"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--flip-left", action="store_true",
                    help="flip the left eye instead of the right")
    args = ap.parse_args()
    global FLIP_RIGHT
    FLIP_RIGHT = not args.flip_left
    os.makedirs(args.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.image))[0]

    frame = cv2.imread(args.image)
    h, w = frame.shape[:2]

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"))
    mesh_m = Stage(os.path.join(args.models, "facemap_3dmm.tflite"))
    gaze_m = Stage(os.path.join(args.models, "eyegaze.tflite"))
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    net_in, transform = face_input(frame)
    faces = detect_faces(det.run(net_in), transform, w, h)
    if not faces:
        sys.exit("no face")
    face = faces[0]

    crop_in, box_used = FaceMesh.crop_input(frame, face["box"])
    lmk, pose = mesh.decode(mesh_m.run(crop_in)["parameters_3dmm"], box_used)

    # annotated overview
    vis = frame.copy()
    x0, y0, x1, y1 = (int(v) for v in face["box"])
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
    for n, (px, py) in enumerate(lmk):
        colour = (0, 0, 255)
        if n in RIGHT_EYE_IDX:
            colour = (255, 0, 0)   # blue = subject's right eye
        elif n in LEFT_EYE_IDX:
            colour = (0, 255, 255) # yellow = subject's left eye
        cv2.circle(vis, (int(px), int(py)), max(2, w // 400), colour, -1)
        if n % 4 == 0:
            cv2.putText(vis, str(n), (int(px) + 4, int(py) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, w / 2500, (255, 255, 255), 1)
    scale = 900 / max(vis.shape[:2])
    cv2.imwrite(f"{args.out}/{tag}-overview.jpg",
                cv2.resize(vis, None, fx=scale, fy=scale))

    # the exact tensors the gaze model receives, at 4x for inspection
    panels = []
    gray = to_gray(frame)
    for side, corners, flip in (("right", RIGHT_EYE_CORNERS, FLIP_RIGHT),
                                ("left", LEFT_EYE_CORNERS, not FLIP_RIGHT)):
        patch, _, _ = eye_patch(gray, lmk, corners, flip=flip)
        if patch is None:
            continue
        g = gaze_output(gaze_m.run(gaze_input(patch)), flipped=flip)
        panel = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (640, 384), interpolation=cv2.INTER_NEAREST)
        cv2.putText(panel, f"{side} (as fed to model)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(panel, f"pitch {np.degrees(g[0]):+.1f} yaw {np.degrees(g[1]):+.1f}",
                    (10, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        panels.append(panel)
    if panels:
        cv2.imwrite(f"{args.out}/{tag}-eyes.jpg", np.hstack(panels))
    print(f"wrote {args.out}/{tag}-overview.jpg and -eyes.jpg")


if __name__ == "__main__":
    main()
