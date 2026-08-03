"""Which eye positions should drive the gaze crop?

Two sources are available for free:
  A. face_det_lite's own 5-point landmark head (eye centres, coarse 8px stride)
  B. facemap_3dmm's 68-point mesh (eye corners)

They disagree, and the crop is where that disagreement turns into gaze error. This
draws both on the same face so the difference is visible, and reports left/right gaze
agreement under each, which is a cheap proxy for crop quality on a frontal face.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    DET5_LEFT_EYE,
    DET5_RIGHT_EYE,
    LEFT_EYE_CORNERS,
    LEFT_EYE_IDX,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_IDX,
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


def patch_from_centre(gray, centre, width, flip):
    """Same canonical warp as eye_patch, but from an explicit centre and eye width."""
    fake = np.array([[centre[0] - width / 2, centre[1]],
                     [centre[0] + width / 2, centre[1]]], dtype=np.float32)
    return eye_patch(gray, fake, (0, 1), flip=flip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "results/landmarks"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"), backend=args.backend)
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"), backend=args.backend)
    gm = Stage(os.path.join(args.models, "eyegaze.tflite"), backend=args.backend)
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    paths = []
    for p in args.images:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    print(f"{'image':<26}{'source':<12}{'L/R agreement':>15}{'eye sep px':>12}")
    print("-" * 66)
    totals = {}
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        net_in, transform = face_input(frame)
        faces = detect_faces(det.run(net_in), transform, w, h)
        if not faces:
            print(f"{os.path.basename(path):<26}no face")
            continue
        face = faces[0]
        crop_in, box_used = FaceMesh.crop_input(frame, face["box"])
        if crop_in is None:
            continue
        lmk, pose = mesh.decode(mm.run(crop_in)["parameters_3dmm"], box_used)
        gray = to_gray(frame)

        variants = {}

        # A: 68-point mesh corners (what the pipeline uses today)
        vecs = {}
        for side, corners, flip in (("right", RIGHT_EYE_CORNERS, True),
                                    ("left", LEFT_EYE_CORNERS, False)):
            p, _, _ = eye_patch(gray, lmk, corners, flip=flip)
            if p is not None:
                vecs[side] = fuse(pose, gaze_output(gm.run(gaze_input(p)), flipped=flip))
        sep = float(np.linalg.norm(lmk[LEFT_EYE_IDX].mean(0) - lmk[RIGHT_EYE_IDX].mean(0)))
        variants["mesh68"] = (vecs, sep)

        # B: detector's 5 points, eye width inferred from the inter-eye distance
        if "landmark5" in face:
            l5 = face["landmark5"]
            sep5 = float(np.linalg.norm(l5[DET5_LEFT_EYE] - l5[DET5_RIGHT_EYE]))
            eye_w = sep5 * 0.42  # a typical corner-to-corner span relative to eye spacing
            vecs5 = {}
            for side, idx, flip in (("right", DET5_RIGHT_EYE, True),
                                    ("left", DET5_LEFT_EYE, False)):
                p, _, _ = patch_from_centre(gray, l5[idx], eye_w, flip)
                if p is not None:
                    vecs5[side] = fuse(
                        pose, gaze_output(gm.run(gaze_input(p)), flipped=flip)
                    )
            variants["det5"] = (vecs5, sep5)

        for name, (vv, s) in variants.items():
            if len(vv) == 2:
                agree = angle_between(vv["left"], vv["right"])
                totals.setdefault(name, []).append(agree)
                print(f"{os.path.basename(path):<26}{name:<12}{agree:>13.1f} d{s:>12.1f}")

        # visual: both sources on one image
        vis = frame.copy()
        for n in RIGHT_EYE_IDX + LEFT_EYE_IDX:
            cv2.circle(vis, (int(lmk[n, 0]), int(lmk[n, 1])), max(2, w // 350),
                       (0, 255, 255), -1)
        if "landmark5" in face:
            for n in range(5):
                cv2.circle(vis, (int(face["landmark5"][n, 0]), int(face["landmark5"][n, 1])),
                           max(3, w // 250), (255, 0, 255), -1)
        cv2.putText(vis, "yellow = mesh68 eye pts   magenta = detector 5pt",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, w / 1600, (255, 255, 255), 2)
        sc = 900 / max(vis.shape[:2])
        cv2.imwrite(os.path.join(args.out, os.path.basename(path)),
                    cv2.resize(vis, None, fx=sc, fy=sc))

    print("-" * 66)
    for name, vals in totals.items():
        print(f"{name:<12} mean L/R agreement {np.mean(vals):6.1f} deg over {len(vals)} faces "
              f"(lower is better)")


if __name__ == "__main__":
    main()
