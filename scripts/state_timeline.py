"""Print the attention state over time for a clip, as a compact strip.

Reading state off a handful of stills is guesswork, because a single frame can land on a
transition. This replays the same pipeline and state machine and prints one character per
half second, so the whole run can be checked against what was actually staged.

    ./venv/bin/python scripts/state_timeline.py --source path/to/clip.mp4 \
        --calib-seconds 4
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import AttentionState, GazePipeline, calibrate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = {"ON": ".", "OFF": "o", "ALERT": "X", "NOFACE": "?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default="htp")
    ap.add_argument("--cone", type=float, default=30.0)
    ap.add_argument("--alert-after", type=float, default=2.0)
    ap.add_argument("--calib-seconds", type=float, default=4.0)
    ap.add_argument("--bucket", type=float, default=0.5, help="seconds per character")
    args = ap.parse_args()

    backend = None if args.backend == "cpu" else args.backend
    pipe = GazePipeline(args.models, backend=backend)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    calib, n = calibrate(cap, pipe, args.calib_seconds, False, "")
    if calib is None:
        sys.exit(f"calibration failed ({n} usable frames)")
    print(f"calibrated on {calib['samples']} samples, spread {calib['spread_deg']:.1f} deg")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pipe._frame_i, pipe._last_box = 0, None

    st = AttentionState(calib["axis"], cone_deg=args.cone, alert_after=args.alert_after)
    states, angles, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe(frame)
        s, a = st.update(res["direction"] if res else None, i / fps)
        states.append(s if res else "NOFACE")
        angles.append(a if a is not None else float("nan"))
        i += 1
    cap.release()

    per = max(1, int(round(args.bucket * fps)))
    print(f"\n{len(states)} frames, {len(states) / fps:.1f}s, "
          f"one character per {args.bucket:.1f}s")
    print("  . = on task    o = looking away    X = attention lost    ? = no face\n")

    strip, marks = [], []
    for b in range(0, len(states), per):
        chunk = states[b : b + per]
        # worst state in the bucket, so a real lapse is never hidden by averaging
        for key in ("NOFACE", "ALERT", "OFF", "ON"):
            if key in chunk:
                strip.append(MARK[key])
                break
        marks.append(b / fps)

    for row in range(0, len(strip), 60):
        t0 = marks[row]
        print(f"  {t0:5.1f}s |{''.join(strip[row:row + 60])}|")

    ang = np.array(angles, dtype=float)
    valid = ~np.isnan(ang)
    print(f"\nangle off axis: median {np.nanmedian(ang):.1f} deg, "
          f"p10 {np.nanpercentile(ang, 10):.1f}, p90 {np.nanpercentile(ang, 90):.1f}")
    for key in ("ON", "OFF", "ALERT", "NOFACE"):
        c = states.count(key)
        print(f"  {key:<7}{c:5d} frames  {100 * c / len(states):5.1f}%")
    print(f"  face found in {100 * valid.mean():.1f}% of frames")


if __name__ == "__main__":
    main()
