"""Charts for the guide, drawn from the result CSVs in the same style as the stage cards.

Two figures, both straight from measured data:

    images/chart-scheduling.png    what skipping the detector costs, desk vs moving footage
    images/chart-calibration.png   per-person error before and after calibration

Drawn with cv2 rather than a plotting library so they share the palette and type scale of
the stage figures. Consistency across a guide's images is worth more than plot features.

    ./venv/bin/python scripts/doc_figures.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_render as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES_A = (200, 130, 60)   # desk footage / after calibration
SERIES_B = (70, 90, 225)    # moving close-up / before calibration


def frame(w, h, title, subtitle):
    img = np.full((h, w, 3), R.PAPER, dtype=np.uint8)
    R._text(img, title, (R.MARGIN, 54), (0.92, 2), R.INK)
    for i, line in enumerate(R._wrap(subtitle, R.SUBTITLE, w - 2 * R.MARGIN)[:2]):
        R._text(img, line, (R.MARGIN, 84 + i * 24), R.SUBTITLE, R.MUTED)
    cv2.line(img, (R.MARGIN, 126), (w - R.MARGIN, 126), R.RULE, 2)
    return img


def legend(img, x, y, items):
    for label, colour in items:
        cv2.rectangle(img, (x, y - 11), (x + 22, y + 3), colour, -1)
        R._text(img, label, (x + 32, y), R.NOTE, R.INK)
        x += 42 + cv2.getTextSize(label, R.F, R.NOTE[0], R.NOTE[1])[0][0]


def grouped_bars(img, rect, groups, series, ymax, unit, ytitle):
    """groups: x labels. series: list of (name, colour, values)."""
    x0, y0, x1, y1 = rect
    R._text(img, ytitle, (x0 - 46, y0 - 14), R.NOTE, R.MUTED)
    for t in range(0, 5):
        v = ymax * t / 4
        yy = int(y1 - (y1 - y0) * (v / ymax))
        cv2.line(img, (x0, yy), (x1, yy), R.RULE, 1)
        R._text(img, f"{v:g}", (x0 - 46, yy + 5), R.NOTE, R.MUTED)

    n_g, n_s = len(groups), len(series)
    slot = (x1 - x0) / n_g
    bw = slot * 0.72 / n_s
    for gi, g in enumerate(groups):
        cx = x0 + slot * (gi + 0.5)
        for si, (_, colour, vals) in enumerate(series):
            v = vals[gi]
            bx = int(cx - slot * 0.36 + si * bw)
            bh = int((y1 - y0) * (v / ymax))
            cv2.rectangle(img, (bx, y1 - bh), (int(bx + bw - 4), y1), colour, -1)
            lab = f"{v:.2f}" if v < 10 else f"{v:.1f}"
            tw = cv2.getTextSize(lab, R.F, 0.5, 1)[0][0]
            dy = 8 if si % 2 == 0 else 24
            R._text(img, lab, (int(bx + (bw - 4 - tw) / 2), y1 - bh - dy), (0.5, 1), R.INK)
        tw = cv2.getTextSize(g, R.F, R.NOTE[0], R.NOTE[1])[0][0]
        R._text(img, g, (int(cx - tw / 2), y1 + 26), R.NOTE, R.INK)
    cv2.line(img, (x0, y1), (x1, y1), R.INK, 2)
    R._text(img, unit, (x1 - 120, y1 + 46), R.NOTE, R.MUTED)


def scheduling(out):
    desk = {r["detect_every"]: r for r in
            csv.DictReader(open(os.path.join(ROOT, "results/sched_cost.csv")))}
    close = {r["detect_every"]: r for r in
             csv.DictReader(open(os.path.join(ROOT, "results/sched_cost_closeup.csv")))}
    keys = [k for k in ("2", "3", "5", "10") if k in desk and k in close]
    img = frame(1280, 660, "What skipping the detector costs",
                "Gaze error introduced against running it on every frame, same code and "
                "board. The answer depends on the footage, not the pipeline.")
    grouped_bars(img, (110, 226, 1180, 520), [f"every {k}" for k in keys],
                 [("person at a desk", SERIES_A,
                   [float(desk[k]["mean_gaze_err_deg"]) for k in keys]),
                  ("handheld close-up", SERIES_B,
                   [float(close[k]["mean_gaze_err_deg"]) for k in keys])],
                 16.0, "detector runs", "mean gaze error, degrees")
    legend(img, 110, 172, [("person at a desk", SERIES_A),
                           ("handheld moving close-up", SERIES_B)])
    R._text(img, "the model's own error is about 15 degrees", (700, 172), R.NOTE, R.MUTED)
    cv2.imwrite(out, img)
    return out


def calibration(out):
    rows = list(csv.DictReader(
        open(os.path.join(ROOT, "results/mpiigaze_eval_calibration.csv"))))
    rows.sort(key=lambda r: float(r["raw_mean_deg"]))
    img = frame(1280, 660, "How much of the error a calibration removes",
                "Each MPIIGaze subject, before and after subtracting their own mean "
                "residual. The gap is fixed bias; what remains is noise.")
    grouped_bars(img, (110, 226, 1180, 520), [r["person"] for r in rows],
                 [("raw", SERIES_B, [float(r["raw_mean_deg"]) for r in rows]),
                  ("after calibration", SERIES_A,
                   [float(r["calibrated_mean_deg"]) for r in rows])],
                 24.0, "subject", "mean angular error, degrees")
    legend(img, 110, 172, [("raw", SERIES_B), ("after per-user calibration", SERIES_A)])
    n = sum(int(r["n"]) for r in rows)
    raw = sum(float(r["raw_mean_deg"]) * int(r["n"]) for r in rows) / n
    cal = sum(float(r["calibrated_mean_deg"]) * int(r["n"]) for r in rows) / n
    R._text(img, f"overall {raw:.2f} -> {cal:.2f} degrees, a {100 * (1 - cal / raw):.0f}% reduction",
            (700, 172), R.NOTE, R.INK)
    cv2.imwrite(out, img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "images"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (scheduling(os.path.join(args.out, "chart-scheduling.png")),
              calibration(os.path.join(args.out, "chart-calibration.png"))):
        im = cv2.imread(f)
        print(f"  {os.path.basename(f):26s} {im.shape[1]}x{im.shape[0]}")


if __name__ == "__main__":
    main()
