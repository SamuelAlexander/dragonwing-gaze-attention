"""Compose the four stage cards into one overview figure.

Stitching four independently captioned images side by side gives four different type
sizes and four different margins. This instead pulls one frame from each stage video,
crops away each card's own header, and lays the imagery on a single grid with one set of
labels, so the strip reads as one figure.

    ./venv/bin/python scripts/pipeline_strip.py --frame 360 --gaze-frame 790
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_render as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANELS = [
    ("stage1-detect", "1", "find the face", "face_det_lite"),
    ("stage2-landmarks", "2", "landmarks and head pose", "facemap_3dmm"),
    ("stage3-eye-patches", "3", "warp the eye patches", "eyegaze input"),
    ("stage4-gaze", "4", "compose one direction", "eyegaze output"),
]

W, H = 1720, 1180
MARGIN, GUTTER = 44, 26
TOP = 176
COLS = 2


def grab(path, frame_no):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_no} of {path}")
    # drop the card's own header and its outer margin: the strip supplies its own
    return img[R.HEADER : img.shape[0] - 20, 20 : img.shape[1] - 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=os.path.join(ROOT, "results/stages"))
    ap.add_argument("--frame", type=int, default=360)
    ap.add_argument("--gaze-frame", type=int, default=790,
                    help="stage 4 needs visible gaze deflection or both arrows vanish")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "images/pipeline-strip.jpg"))
    args = ap.parse_args()

    img = np.full((H, W, 3), R.PAPER, dtype=np.uint8)
    R._text(img, "One frame, four models", (MARGIN, 58), (1.05, 3), R.INK)
    R._text(img, "One camera frame, read left to right and top to bottom. Every panel is a "
                 "real run: each model's input beside what it hands to the next stage.",
            (MARGIN, 92), R.SUBTITLE, R.MUTED)
    cv2.line(img, (MARGIN, 118), (W - MARGIN, 118), R.RULE, 2)

    n = len(PANELS)
    rows = (n + COLS - 1) // COLS
    pw = (W - 2 * MARGIN - GUTTER * (COLS - 1)) // COLS
    row_h = (H - TOP - 40) // rows
    avail_h = row_h - 62

    for i, (name, no, label, model) in enumerate(PANELS):
        r, c = divmod(i, COLS)
        px = MARGIN + c * (pw + GUTTER)
        py = TOP + r * row_h
        frame_no = args.gaze_frame if name == "stage4-gaze" else args.frame
        tile = grab(os.path.join(args.stages, f"{name}.mp4"), frame_no)

        sc = min(pw / tile.shape[1], avail_h / tile.shape[0])
        tile = cv2.resize(tile, (int(tile.shape[1] * sc), int(tile.shape[0] * sc)),
                          interpolation=cv2.INTER_AREA)
        tx = px + (pw - tile.shape[1]) // 2
        ty = py + 56 + (avail_h - tile.shape[0]) // 2
        img[ty : ty + tile.shape[0], tx : tx + tile.shape[1]] = tile
        cv2.rectangle(img, (tx - 1, ty - 1),
                      (tx + tile.shape[1], ty + tile.shape[0]), R.RULE, 1)

        R._text(img, no, (px, py + 26), (0.9, 2), R.ACCENT)
        R._text(img, label, (px + 28, py + 26), (0.74, 2), R.INK)
        R._text(img, model, (px + 28, py + 48), R.NOTE, R.MUTED)

    cv2.imwrite(args.out, img)
    print(f"  {args.out}  {W}x{H}")


if __name__ == "__main__":
    main()
