"""Sanity-check the three-stage chain on one image. Prints, does not draw.

    python scripts/check_chain.py test-data/portrait-6471976.jpg [--backend htp]

With --compare it runs every stage on both CPU and NPU and reports the largest
per-tensor difference, which is the only way to know the delegate is honest.
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
    angle_between,
    detect_faces,
    eye_patch,
    face_input,
    fuse,
    gaze_input,
    gaze_output,
    pitchyaw_to_vector,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EYES = (("right", RIGHT_EYE_CORNERS, True), ("left", LEFT_EYE_CORNERS, False))


def build(models, backend):
    return (
        Stage(os.path.join(models, "face_det_lite.tflite"), backend=backend),
        Stage(os.path.join(models, "facemap_3dmm.tflite"), backend=backend),
        Stage(os.path.join(models, "eyegaze.tflite"), backend=backend),
    )


def run_chain(frame, det, mesh_m, gaze_m, mesh, verbose=True):
    h, w = frame.shape[:2]
    net_in, transform = face_input(frame)
    det_out = det.run(net_in)
    faces = detect_faces(det_out, transform, w, h)
    if verbose:
        print(f"faces        : {len(faces)}")
    if not faces:
        return None

    face = faces[0]
    x0, y0, x1, y1 = face["box"]
    if verbose:
        print(f"  best face  : score {face['score']:.3f}  "
              f"({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})  "
              f"{x1 - x0:.0f}x{y1 - y0:.0f}px  aspect {(x1 - x0) / max(y1 - y0, 1):.2f}")

    crop_in, box_used = FaceMesh.crop_input(frame, face["box"])
    if crop_in is None:
        return None
    params = mesh_m.run(crop_in)["parameters_3dmm"]
    lmk, pose = mesh.decode(params, box_used)

    inside = ((lmk[:, 0] >= x0 - 10) & (lmk[:, 0] <= x1 + 10)
              & (lmk[:, 1] >= y0 - 10) & (lmk[:, 1] <= y1 + 10)).mean()
    if verbose:
        print(f"head pose    : pitch {np.degrees(pose[0]):+6.1f}  "
              f"yaw {np.degrees(pose[1]):+6.1f}  roll {np.degrees(pose[2]):+6.1f}  (deg)")
        print(f"lmk in box   : {inside * 100:.0f}%")

    gray = to_gray(frame)
    vecs, gazes = {}, {}
    for side, corners, flip in EYES:
        patch, centre, _ = eye_patch(gray, lmk, corners, flip=flip)
        if patch is None:
            continue
        g = gaze_output(gaze_m.run(gaze_input(patch)), flipped=flip)
        gazes[side] = g
        vecs[side] = fuse(pose, g)
        if verbose:
            eye_only = pitchyaw_to_vector(g[0], g[1])
            print(f"  {side:5s} eye : centre ({centre[0]:.0f},{centre[1]:.0f})  "
                  f"gaze pitch {np.degrees(g[0]):+6.1f} yaw {np.degrees(g[1]):+6.1f}  "
                  f"head adds {angle_between(eye_only, vecs[side]):5.1f} deg")

    if verbose and len(vecs) == 2:
        print(f"eye agreement: {angle_between(vecs['left'], vecs['right']):.1f} deg "
              f"(low = consistent crops)")
    return {"faces": faces, "lmk": lmk, "pose": pose, "gaze": gazes, "vecs": vecs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--backend", default=None, help="htp for NPU, omit for CPU")
    ap.add_argument("--compare", action="store_true",
                    help="run CPU and NPU and diff every stage")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")
    print(f"image        : {args.image}  {frame.shape[1]}x{frame.shape[0]}")

    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    if not args.compare:
        print(f"backend      : {args.backend or 'cpu'}")
        det, mesh_m, gaze_m = build(args.models, args.backend)
        run_chain(frame, det, mesh_m, gaze_m, mesh)
        return

    print("\n--- CPU ---")
    cpu = run_chain(frame, *build(args.models, None), mesh)
    print("\n--- NPU (htp) ---")
    npu = run_chain(frame, *build(args.models, "htp"), mesh)

    if cpu is None or npu is None:
        sys.exit("one backend found no face; cannot compare")

    print("\n--- CPU vs NPU ---")
    print(f"face box max delta : "
          f"{max(abs(a - b) for a, b in zip(cpu['faces'][0]['box'], npu['faces'][0]['box'])):.3f} px")
    print(f"landmark max delta : {np.abs(cpu['lmk'] - npu['lmk']).max():.4f} px")
    print(f"head pose max delta: {np.degrees(np.abs(cpu['pose'] - npu['pose'])).max():.4f} deg")
    for side in cpu["gaze"]:
        if side in npu["gaze"]:
            d = np.degrees(np.abs(cpu["gaze"][side] - npu["gaze"][side])).max()
            ang = angle_between(cpu["vecs"][side], npu["vecs"][side])
            print(f"gaze {side:5s} delta   : {d:.4f} deg per angle, {ang:.4f} deg of direction")


if __name__ == "__main__":
    main()
