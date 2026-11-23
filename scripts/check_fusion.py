"""Is EyeGaze's output relative to the head or to the camera? Settled against ground truth.

The question decides whether head pose belongs in the gaze direction at all. If the model
reports gaze relative to the head, the direction in the room is that gaze rotated by the
head pose. If it already reports gaze relative to the camera, rotating it again counts
the head twice.

MPIIGaze's original images carry, per frame, a 3D gaze target and 3D eye centres in
camera coordinates (so the true gaze direction is known) and a fitted head pose (so the
true head rotation is known too). That allows a test with no estimated quantity in the
head-pose path:

  eye-only          EyeGaze's direction, used as it comes out
  + true head pose  the same, rotated by MPIIGaze's ground-truth head rotation
  + model head pose the same, rotated by facemap_3dmm's head rotation

If EyeGaze were head-relative, composing with the TRUE head pose would be the best of the
three. A regression of the model's angles against true gaze and true head angles gives the
same answer a second way: a head-relative model must respond to head rotation, a
camera-relative one must not.

    ./venv/bin/python scripts/check_fusion.py --per-day 3
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
    gaze_vector,
    head_rotation,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# annotation.txt columns (0-based): 24 landmark coords, 2 screen coords, then these
GAZE_TARGET = slice(26, 29)
HEAD_RVEC = slice(29, 32)        # Rodrigues vector, head model -> camera
RIGHT_EYE_3D = slice(35, 38)
LEFT_EYE_3D = slice(38, 41)

# facemap_3dmm's rotation maps its model space into camera space, so even a frontal head
# carries this flip. Composing a gaze vector needs the rotation relative to frontal.
P_FLIP = np.diag([1.0, -1.0, -1.0])
FORWARD = np.array([0.0, 0.0, -1.0])     # facing the lens


def ang(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def pitch_yaw(v):
    v = v / np.linalg.norm(v)
    return np.degrees([np.arcsin(v[1]), np.arctan2(-v[0], -v[2])])


def frame_list(root, per_day, seed=0):
    """A fixed random handful of frames from every recording day, so runs repeat."""
    rng = np.random.default_rng(seed)
    out = []
    for day in sorted(glob.glob(os.path.join(root, "p*", "day*"))):
        ann = os.path.join(day, "annotation.txt")
        if not os.path.exists(ann):
            continue
        lines = open(ann).read().strip().splitlines()
        for i in sorted(rng.choice(len(lines), min(per_day, len(lines)), replace=False)):
            out.append((day, int(i), lines[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        ROOT, "test-data/mpiigaze/MPIIGaze/Data/Original"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--per-day", type=int, default=3)
    ap.add_argument("--backend", default="htp")
    ap.add_argument("--list", action="store_true",
                    help="print the image paths this run uses, then exit")
    args = ap.parse_args()

    frames = frame_list(args.root, args.per_day)
    if not frames:
        sys.exit(f"no original data under {args.root}")
    if args.list:
        for day, i, _ in frames:
            print(os.path.relpath(os.path.join(day, f"{i + 1:04d}.jpg"), args.root))
        return

    backend = None if args.backend == "cpu" else args.backend
    det = Stage(os.path.join(args.models, "face_det_lite.tflite"), backend=backend)
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"), backend=backend)
    gm = Stage(os.path.join(args.models, "eyegaze.tflite"), backend=backend)
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    rows = []
    for day, i, line in frames:
        v = np.array(line.split(), dtype=np.float64)
        frame = cv2.imread(os.path.join(day, f"{i + 1:04d}.jpg"))
        if len(v) < 41 or frame is None:
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
        r_true, _ = cv2.Rodrigues(v[HEAD_RVEC])
        r_model = head_rotation(pose) @ P_FLIP
        gray = to_gray(frame)
        for corners, flip, eye3d in ((RIGHT_EYE_CORNERS, True, v[RIGHT_EYE_3D]),
                                     (LEFT_EYE_CORNERS, False, v[LEFT_EYE_3D])):
            patch, _, _ = eye_patch(gray, lmk, corners, flip=flip)
            if patch is None:
                continue
            e = gaze_vector(gaze_output(gm.run(gaze_input(patch)), flipped=flip))
            rows.append({"e": e.astype(np.float64), "gt": v[GAZE_TARGET] - eye3d,
                         "r_true": r_true, "r_model": r_model})

    if not rows:
        sys.exit("nothing scored")

    variants = {
        "eye-only": lambda r: r["e"],
        "+ true head pose": lambda r: r["r_true"] @ r["e"],
        "+ model head pose": lambda r: r["r_model"] @ r["e"],
    }
    errs = {k: np.array([ang(f(r), r["gt"]) for r in rows]) for k, f in variants.items()}
    turn = np.array([ang(r["r_true"] @ FORWARD, FORWARD) for r in rows])
    big = turn >= np.percentile(turn, 75)

    print(f"scored {len(rows)} eyes from {len(frames)} frames\n")
    print(f"{'':<20}{'mean':>8}{'median':>8}   {'largest head turns':>18}")
    print("-" * 58)
    for k, e in errs.items():
        print(f"{k:<20}{e.mean():>8.2f}{np.median(e):>8.2f}   {e[big].mean():>18.2f}")
    print(f"\n(largest head turns: the quarter of eyes with the head turned "
          f"{np.percentile(turn, 75):.0f} deg or more)")

    # second way to ask the same question
    pred = np.array([pitch_yaw(r["e"]) for r in rows])
    true = np.array([pitch_yaw(r["gt"]) for r in rows])
    head = np.array([pitch_yaw(r["r_true"] @ FORWARD) for r in rows])
    print("\nmodel angle = a * true gaze angle + b * true head angle + c")
    for k, name in enumerate(("pitch", "yaw")):
        a, b, c = np.linalg.lstsq(np.c_[true[:, k], head[:, k], np.ones(len(rows))],
                                  pred[:, k], rcond=None)[0]
        print(f"  {name:<6} a = {a:.2f}   b = {b:+.2f}")
    print("  head-relative model: b near -a.  camera-relative model: b near 0.")
    print("  a below 1 means the model under-reads large angles.")


if __name__ == "__main__":
    main()
