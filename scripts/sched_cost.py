"""What does running the face detector less often actually cost?

Skipping the detector is the obvious way to buy frame rate in a multi-model pipeline,
and it worked well elsewhere in this series. It behaves differently here, because every
downstream stage crops from the detector's box: a stale box does not merely delay
detection, it feeds the landmark model a misaligned crop, which moves the eye patches,
which moves the gaze.

This measures gaze error against the every-frame result, per frame, so the trade is a
number rather than an impression.

    ./venv/bin/python scripts/sched_cost.py --source test-data/face-closeup.mp4
"""

import argparse
import csv
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import GazePipeline  # noqa: E402
from gaze_common import angle_between  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(source, models, backend, every, limit=0):
    cap = cv2.VideoCapture(source)
    pipe = GazePipeline(models, backend=backend, detect_every=every)
    dirs, boxes, t0, n = [], [], time.perf_counter(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe(frame)
        dirs.append(None if res is None else res["direction"])
        boxes.append(None if res is None else res["box"])
        n += 1
        if limit and n >= limit:
            break
    cap.release()
    return dirs, boxes, (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "test-data/face-closeup.mp4"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default="htp")
    ap.add_argument("--every", default="1,2,3,5,10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default=os.path.join(ROOT, "results/sched_cost.csv"))
    args = ap.parse_args()

    backend = None if args.backend == "cpu" else args.backend
    everies = [int(v) for v in args.every.split(",")]

    print("baseline: detector on every frame")
    ref_dirs, ref_boxes, ref_t = run(args.source, args.models, backend, 1, args.limit)
    n = len(ref_dirs)
    got = sum(d is not None for d in ref_dirs)
    print(f"  {n} frames, {got} with a face, {n / ref_t:.1f} FPS\n")

    rows = []
    print(f"{'every':>6}{'FPS':>8}{'speedup':>9}{'mean err':>10}{'p95 err':>9}"
          f"{'max err':>9}{'>10deg':>8}{'lost':>6}")
    print("-" * 65)
    for every in everies:
        dirs, boxes, t = run(args.source, args.models, backend, every, args.limit)
        errs, lost, box_drift = [], 0, []
        for a, b, ba, bb in zip(ref_dirs, dirs, ref_boxes, boxes):
            if a is None:
                continue
            if b is None:
                lost += 1
                continue
            errs.append(angle_between(a, b))
            if ba and bb:
                box_drift.append(max(abs(u - v) for u, v in zip(ba, bb)))
        if not errs:
            continue
        errs_sorted = sorted(errs)
        p95 = errs_sorted[int(0.95 * (len(errs_sorted) - 1))]
        over = 100.0 * sum(e > 10 for e in errs) / len(errs)
        fps = n / t
        print(f"{every:>6}{fps:>8.1f}{fps / (n / ref_t):>8.2f}x"
              f"{statistics.mean(errs):>10.2f}{p95:>9.2f}{max(errs):>9.2f}"
              f"{over:>7.0f}%{lost:>6}")
        rows.append({
            "detect_every": every, "fps": round(fps, 2),
            "speedup": round(fps / (n / ref_t), 3),
            "mean_gaze_err_deg": round(statistics.mean(errs), 3),
            "p95_gaze_err_deg": round(p95, 3),
            "max_gaze_err_deg": round(max(errs), 3),
            "pct_over_10deg": round(over, 2),
            "frames_lost": lost,
            "mean_box_drift_px": round(statistics.mean(box_drift), 2) if box_drift else "",
            "frames": n, "backend": args.backend,
            "source": os.path.basename(args.source),
        })
    print("-" * 65)
    print("err is the angle between this setting's gaze and the every-frame gaze,")
    print("on the same frame. The model's own mean error is about 17 deg, so anything")
    print("approaching that is giving back everything the NPU bought.")

    if rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
