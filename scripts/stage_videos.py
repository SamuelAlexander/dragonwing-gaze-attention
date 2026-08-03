"""Render the whole clip once, writing one video per pipeline stage.

Four synchronised 1920x1080 videos, same length and framerate as the source, so they cut
together in an editor. Stage 5 is the application itself, which `attention.py` already
writes.

One pass over the source drives all four writers, because running the chain four times
would cost four times as long for identical results.

    ./venv/bin/python scripts/stage_videos.py --source test-data/sample03.mp4

The same code renders the README's four stage stills, so they match the videos exactly:

    ./venv/bin/python scripts/stage_videos.py --still 360 --gaze-still 790
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_render as R  # noqa: E402
from gaze_common import (  # noqa: E402
    FACE_H,
    FACE_W,
    LEFT_EYE_CORNERS,
    LEFT_EYE_IDX,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_IDX,
    FaceMesh,
    Stage,
    detect_faces,
    eye_landmarks_to_frame,
    eye_patch,
    face_input,
    fuse,
    gaze_input,
    gaze_output,
    iris_circle,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = [
    ("stage1-detect", 1, "face_det_lite",
     "Find the face. 878K parameters, 7.31 ms on the NPU.", "480x640 grayscale in"),
    ("stage2-landmarks", 2, "facemap_3dmm",
     "68 landmarks and the head's orientation. 5.42M parameters, 1.01 ms on the NPU.",
     "128x128 RGB in"),
    ("stage3-eye-patches", 3, "eyegaze input",
     "Warp each eye into the model's canonical frame. The right eye is mirrored.",
     "96x160 grayscale in"),
    ("stage4-gaze", 4, "eyegaze output",
     "Per-eye gaze, then both composed with head pose into one direction.",
     "2.58M parameters, 4.78 ms per eye"),
]


def crop_around(frame, box, pad_frac=0.55):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    return frame[max(0, y0 - pad) : min(h - 1, y1 + pad),
                 max(0, x0 - pad) : min(w - 1, x1 + pad)]


def blank(text="no face detected"):
    img = np.full((360, 640, 3), R.PAPER, dtype=np.uint8)
    tw = cv2.getTextSize(text, R.F, 0.8, 2)[0][0]
    cv2.putText(img, text, ((640 - tw) // 2, 190), R.F, 0.8, R.MUTED, 2, cv2.LINE_AA)
    return img


def render(cap, writers, det, mm, gm, mesh, limit=0, total=0):
    """Run the chain over cap and write one card per stage per frame to writers.

    writers maps stage name to anything with .write(img): a VideoWriter for the
    videos, or Keep for single stills. Returns (frames, start_time).
    """
    i, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # ---- stage 1
        net_in, transform = face_input(frame)
        faces = detect_faces(det.run(net_in), transform, w, h)
        seen = cv2.cvtColor((net_in[0, :, :, 0] * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)
        boxed = frame.copy()
        if faces:
            f0 = faces[0]
            bx0, by0, bx1, by1 = (int(v) for v in f0["box"])
            cv2.rectangle(boxed, (bx0, by0), (bx1, by1), R.GREEN, 3)
            cv2.putText(boxed, f"face {f0['score']:.2f}", (bx0, max(30, by0 - 12)),
                        R.F, 1.0, R.GREEN, 2, cv2.LINE_AA)
            for px, py in f0.get("landmark5", []):
                cv2.circle(boxed, (int(px), int(py)), 7, (200, 120, 40), -1, cv2.LINE_AA)
            boxed = crop_around(boxed, f0["box"])
        name, no, model, what, note = STAGES[0]
        img, rect = R.card(no, model, what, note)
        R.place(img, rect, [(seen, "what the model receives",
                             "letterboxed to 4:3, padding included"),
                            (boxed, "what it returns",
                             "one box per face, plus five coarse points")])
        writers[name].write(img)

        if not faces:
            for name, no, model, what, note in STAGES[1:]:
                img, rect = R.card(no, model, what, note)
                R.place(img, rect, [(blank(), "waiting for a face", None)])
                writers[name].write(img)
            i += 1
            if limit and i >= limit:
                break
            continue

        # ---- stage 2
        crop_in, box_used = FaceMesh.crop_input(frame, faces[0]["box"])
        lmk, pose, geom = mesh.decode3d(mm.run(crop_in)["parameters_3dmm"], box_used)
        crop_vis = (crop_in[0][:, :, ::-1] * 255).astype(np.uint8).copy()
        crop_vis = cv2.resize(crop_vis, (512, 512), interpolation=cv2.INTER_NEAREST)
        lm_vis = frame.copy()
        for n, (px, py) in enumerate(lmk):
            c = R.IRIS_R if n in RIGHT_EYE_IDX else (
                R.IRIS_L if n in LEFT_EYE_IDX else R.DOT)
            cv2.circle(lm_vis, (int(px), int(py)), 4, c, -1, cv2.LINE_AA)
        lm_vis = crop_around(lm_vis, faces[0]["box"], 0.35)
        pose_panel = R.readout(
            [("pitch", f"{np.degrees(pose[0]):+6.1f}"),
             ("yaw", f"{np.degrees(pose[1]):+6.1f}"),
             ("roll", f"{np.degrees(pose[2]):+6.1f}")], 420, 190)
        name, no, model, what, note = STAGES[1]
        img, rect = R.card(no, model, what, note)
        R.place(img, rect, [(crop_vis, "what the model receives",
                             "the box, squeezed to 128x128"),
                            (lm_vis, "what it returns",
                             "68 landmarks, eyes in blue and green"),
                            (pose_panel, "head pose", "degrees", 1.0)],
              weights=[3, 4, 2])
        writers[name].write(img)

        # ---- stages 3 and 4
        gray = to_gray(frame)
        patches, vecs, eyes = [], [], []
        for side, corners, idx, flip in (
                ("right", RIGHT_EYE_CORNERS, RIGHT_EYE_IDX, True),
                ("left", LEFT_EYE_CORNERS, LEFT_EYE_IDX, False)):
            patch, centre, inv = eye_patch(gray, lmk, corners, flip=flip)
            out = gm.run(gaze_input(patch))
            g = gaze_output(out, flipped=flip)
            v = fuse(pose, g)
            vecs.append(v)
            ic, ir = iris_circle(eye_landmarks_to_frame(out, inv, flip))
            if ic is None:
                ic, ir = np.array(centre), 8.0
            eyes.append((side, np.asarray(ic, float), float(ir), g, v))
            p = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            patches.append((cv2.resize(p, (640, 384), interpolation=cv2.INTER_NEAREST),
                            f"{side} eye" + ("  (mirrored in)" if flip else ""),
                            f"pitch {np.degrees(g[0]):+.1f}   yaw {np.degrees(g[1]):+.1f}"))
        name, no, model, what, note = STAGES[2]
        img, rect = R.card(no, model, what, note)
        R.place(img, rect, patches)
        writers[name].write(img)

        mean_v = np.mean(vecs, axis=0)
        mean_v = mean_v / max(np.linalg.norm(mean_v), 1e-6)
        g_vis = frame.copy()
        inter = float(np.linalg.norm(eyes[0][1] - eyes[1][1])) if len(eyes) == 2 else 60.0
        for side, ic, ir, g, v in eyes:
            c = R.IRIS_R if side == "right" else R.IRIS_L
            cv2.circle(g_vis, (int(ic[0]), int(ic[1])), int(max(7, ir)), c, 3, cv2.LINE_AA)
            d = np.array([v[0], v[1]]) * 1.5 * inter
            cv2.arrowedLine(g_vis, (int(ic[0]), int(ic[1])),
                            (int(ic[0] + d[0]), int(ic[1] + d[1])), c, 3,
                            cv2.LINE_AA, tipLength=0.25)
        mid = np.mean([e[1] for e in eyes], axis=0)
        d = np.array([mean_v[0], mean_v[1]]) * 2.0 * inter
        cv2.arrowedLine(g_vis, (int(mid[0]), int(mid[1])),
                        (int(mid[0] + d[0]), int(mid[1] + d[1])),
                        (255, 255, 255), 6, cv2.LINE_AA, tipLength=0.22)
        g_vis = crop_around(g_vis, faces[0]["box"], 0.45)
        gaze_panel = R.readout(
            [("right eye", f"{np.degrees(eyes[0][3][0]):+.1f} / {np.degrees(eyes[0][3][1]):+.1f}"),
             ("left eye", f"{np.degrees(eyes[1][3][0]):+.1f} / {np.degrees(eyes[1][3][1]):+.1f}"),
             ("composed", f"{np.degrees(np.arctan2(mean_v[0], -mean_v[2])):+.1f} yaw")],
            520, 190)
        name, no, model, what, note = STAGES[3]
        img, rect = R.card(no, model, what, note)
        R.place(img, rect, [(g_vis, "blue and green: one estimate per eye",
                             "white: their mean, carried through the head's rotation"),
                            (gaze_panel, "pitch / yaw, degrees", None, 1.0)],
              weights=[5, 2])
        writers[name].write(img)

        i += 1
        if limit and i >= limit:
            break
        if i % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{total}  {i / el:.1f} FPS  eta {(total - i) / (i / el) / 60:.1f} min")

    return i, t0


class Keep:
    """Stand-in for a VideoWriter that keeps the last frame, for rendering stills."""

    def __init__(self):
        self.img = None

    def write(self, img):
        self.img = img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "test-data/sample03.mp4"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default=None, help="htp on the board, omit for CPU")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "results/stages"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--still", type=int, default=None,
                    help="render this one frame as four JPG cards instead of videos")
    ap.add_argument("--gaze-still", type=int, default=None,
                    help="separate frame for the stage 4 still; near-zero gaze makes "
                         "both per-eye arrows vanish, which hides the point it makes")
    ap.add_argument("--stills-out", default=os.path.join(ROOT, "images"))
    args = ap.parse_args()

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"), backend=args.backend)
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"), backend=args.backend)
    gm = Stage(os.path.join(args.models, "eyegaze.tflite"), backend=args.backend)
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    if args.still is not None:
        names = [n for n, *_ in STAGES]
        jobs = [(args.still, names)]
        if args.gaze_still is not None:
            jobs = [(args.still, names[:3]), (args.gaze_still, names[3:])]
        os.makedirs(args.stills_out, exist_ok=True)
        for frame_no, wanted in jobs:
            keep = {n: Keep() for n in names}
            cap = cv2.VideoCapture(args.source)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            render(cap, keep, det, mm, gm, mesh, limit=1)
            cap.release()
            for n in wanted:
                out = os.path.join(args.stills_out, f"{n}.jpg")
                cv2.imwrite(out, keep[n].img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"  {out}  (frame {frame_no})")
        return

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writers = {
        name: cv2.VideoWriter(os.path.join(args.out, f"{name}.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (R.W, R.H))
        for name, _, _, _, _ in STAGES
    }
    print(f"{total} frames at {fps:.0f} fps -> {len(writers)} videos in {args.out}")

    i, t0 = render(cap, writers, det, mm, gm, mesh, args.max_frames, total)
    cap.release()
    for wr in writers.values():
        wr.release()
    print(f"\n{i} frames in {(time.perf_counter() - t0) / 60:.1f} min")
    for name, *_ in STAGES:
        p = os.path.join(args.out, f"{name}.mp4")
        print(f"  {p}  {os.path.getsize(p) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
