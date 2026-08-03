"""Does composing head pose with eye gaze actually help? Measured against ground truth.

MPIIGaze's original images carry a 3D gaze target and 3D eye centres in camera
coordinates, so the true gaze direction per eye is known. That lets us score the whole
pipeline, crop geometry and fusion included, rather than just the model on pre-normalised
patches.

Four variants are compared on identical frames:

  eye-only     the model's head-relative gaze, used directly
  fused        eye gaze rotated by the head pose, as the app does today
  fused-conj   same, but with the head rotation expressed in camera coordinates
               (P R P, where P is the y/z flip the mesh decode applies), which is the
               correction this script exists to test
  fused-inv    the transpose, in case the rotation direction is the thing that is wrong

    ./venv/bin/python scripts/check_fusion.py --limit 120
"""

import argparse
import glob
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
    gaze_output,
    pitchyaw_to_vector,
    rotation_from_pose,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# annotation.txt column layout, verified against interocular distance (~65 mm)
GAZE_TARGET = slice(26, 29)
RIGHT_EYE_3D = slice(35, 38)
LEFT_EYE_3D = slice(38, 41)

# the mesh decode's model-to-image flip
P_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def ang(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        ROOT, "test-data/mpiigaze/MPIIGaze/Data/Original"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"), backend=args.backend)
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"), backend=args.backend)
    gm = Stage(os.path.join(args.models, "eyegaze.tflite"), backend=args.backend)
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    days = sorted(glob.glob(os.path.join(args.root, "p*", "day*")))
    if not days:
        sys.exit(f"no original data under {args.root}")

    errs = {k: [] for k in ("eye-only", "fused", "fused-conj", "fused-inv")}
    head_yaw = []
    used = 0
    per_day = max(1, args.limit // max(1, min(len(days), 30)))

    for day in days:
        if used >= args.limit:
            break
        ann = os.path.join(day, "annotation.txt")
        if not os.path.exists(ann):
            continue
        lines = open(ann).read().strip().splitlines()
        for i, line in enumerate(lines[:per_day]):
            if used >= args.limit:
                break
            v = np.array(line.split(), dtype=np.float64)
            if len(v) < 41:
                continue
            img_path = os.path.join(day, f"{i + 1:04d}.jpg")
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            net_in, tr = face_input(frame)
            faces = detect_faces(det.run(net_in), tr, w, h)
            if not faces:
                continue
            crop_in, box_used = FaceMesh.crop_input(frame, faces[0]["box"])
            if crop_in is None:
                continue
            lmk, pose = mesh.decode(mm.run(crop_in)["parameters_3dmm"], box_used)
            gray = to_gray(frame)

            R = rotation_from_pose(*pose)
            R_conj = P_FLIP @ R @ P_FLIP
            target = v[GAZE_TARGET]

            got = False
            for corners, flip, eye3d in (
                (RIGHT_EYE_CORNERS, True, v[RIGHT_EYE_3D]),
                (LEFT_EYE_CORNERS, False, v[LEFT_EYE_3D]),
            ):
                patch, _, _ = eye_patch(gray, lmk, corners, flip=flip)
                if patch is None:
                    continue
                g = gaze_output(gm.run(gaze_input(patch)), flipped=flip)
                e = pitchyaw_to_vector(g[0], g[1])

                # MPIIGaze camera coordinates are already OpenCV style: x right,
                # y down, z into the scene. Our pitchyaw_to_vector uses the same
                # convention (neutral gaze is [0, 0, -1], back toward the camera),
                # so no axis flip is needed. Applying one puts the two vectors
                # roughly 180 degrees apart, which is how this was caught.
                gt = target - eye3d
                gt_img = gt / np.linalg.norm(gt)

                errs["eye-only"].append(ang(e, gt_img))
                errs["fused"].append(ang(R @ e, gt_img))
                errs["fused-conj"].append(ang(R_conj @ e, gt_img))
                errs["fused-inv"].append(ang(R_conj.T @ e, gt_img))
                head_yaw.append(abs(float(np.degrees(pose[1]))))
                got = True
            if got:
                used += 1

    if not errs["eye-only"]:
        sys.exit("nothing scored")

    print(f"scored {used} frames, {len(errs['eye-only'])} eyes\n")
    print(f"{'variant':<14}{'mean':>9}{'median':>9}{'p95':>9}")
    print("-" * 41)
    for k in ("eye-only", "fused", "fused-conj", "fused-inv"):
        a = sorted(errs[k])
        print(f"{k:<14}{np.mean(a):>9.2f}{np.median(a):>9.2f}"
              f"{a[int(0.95 * (len(a) - 1))]:>9.2f}")
    print("-" * 41)
    best = min(errs, key=lambda k: np.mean(errs[k]))
    print(f"best: {best}")

    # paired against eye-only, which is the do-nothing baseline
    base = np.array(errs["eye-only"])
    print(f"\npaired difference vs eye-only (negative = better):")
    for k in ("fused", "fused-conj", "fused-inv"):
        d = np.array(errs[k]) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {k:<12}{d.mean():+7.2f} deg  +/- {1.96 * se:.2f} (95% CI)"
              f"   {'SIGNIFICANT' if abs(d.mean()) > 1.96 * se else 'not significant'}")

    # the case Sam actually films is a big head turn, which MPIIGaze barely covers
    hy = np.array(head_yaw)
    hi = hy >= np.percentile(hy, 70)
    print(f"\nlargest-head-rotation 30% (|yaw| >= {np.percentile(hy, 70):.1f} deg, n={hi.sum()}):")
    for k in ("eye-only", "fused", "fused-inv"):
        print(f"  {k:<12}{np.array(errs[k])[hi].mean():>7.2f} deg")
    print("\n'fused' is what the app does today. If eye-only beats it, the head-pose")
    print("composition is hurting and must be fixed before it drives anything.")


if __name__ == "__main__":
    main()
