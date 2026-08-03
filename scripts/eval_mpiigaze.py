"""Score EyeGaze against MPIIGaze ground truth, and test whether calibration helps.

Two questions, and the second is the one that decides whether the application design
is sound:

  1. Does our inference path reproduce the published mean angular error on the
     dataset's own normalised patches? If not, something is wrong with us, not the model.
  2. How much of that error is a fixed per-person bias rather than random noise? A
     3-second calibration can subtract a bias; it cannot subtract noise. If most of the
     error is bias, a calibrated cone is far tighter than the headline 17 degrees implies.

    ./venv/bin/python scripts/eval_mpiigaze.py --root test-data/mpiigaze/MPIIGaze \
        --backend htp --limit 500
"""

import argparse
import csv
import glob
import os
import statistics
import sys

import cv2
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import Stage, gaze_input  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pitchyaw_to_vec(pitch, yaw):
    return np.array(
        [-np.cos(pitch) * np.sin(yaw), np.sin(pitch), -np.cos(pitch) * np.cos(yaw)],
        dtype=np.float64,
    )


def angular_error(a, b):
    va, vb = pitchyaw_to_vec(*a), pitchyaw_to_vec(*b)
    va /= np.linalg.norm(va)
    vb /= np.linalg.norm(vb)
    return float(np.degrees(np.arccos(np.clip(va @ vb, -1.0, 1.0))))


def load_entries(root):
    """The dataset's own evaluation subset: 15 people, 3000 samples, fixed list."""
    pattern = os.path.join(root, "Evaluation Subset", "sample list for eye image", "*.txt")
    entries = []
    for path in sorted(glob.glob(pattern)):
        person = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, side = (x.strip() for x in line.split())
                day, img = img_path.split("/")
                entries.append({"person": person, "day": day, "img": img, "side": side})
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "test-data/mpiigaze/MPIIGaze"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default="htp")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--csv", default=os.path.join(ROOT, "results/mpiigaze_eval.csv"))
    args = ap.parse_args()

    entries = load_entries(args.root)
    if not entries:
        sys.exit(f"no evaluation subset found under {args.root}")
    print(f"evaluation subset: {len(entries)} samples, "
          f"{len({e['person'] for e in entries})} people")

    # deterministic subsample, spread across people
    if args.limit and args.limit < len(entries):
        idx = np.linspace(0, len(entries) - 1, args.limit).astype(int)
        entries = [entries[i] for i in idx]
    print(f"scoring {len(entries)} samples on {args.backend}")

    backend = None if args.backend == "cpu" else args.backend
    model = Stage(os.path.join(args.models, "eyegaze.tflite"), backend=backend)

    mat_cache = {}
    bad_files = set()
    per_person = {}
    rows = []
    skipped = 0

    for k, e in enumerate(entries):
        key = (e["person"], e["day"])
        if key in bad_files:
            skipped += 1
            continue
        if key not in mat_cache:
            mp = os.path.join(args.root, "Data", "Normalized", e["person"],
                              f"{e['day']}.mat")
            if not os.path.exists(mp):
                bad_files.add(key)
                skipped += 1
                continue
            try:
                mat_cache[key] = sio.loadmat(mp)
            except Exception:
                # a truncated .mat, e.g. from a partially downloaded archive
                bad_files.add(key)
                skipped += 1
                continue
            if len(mat_cache) > 12:
                mat_cache.pop(next(iter(mat_cache)))
        mat = mat_cache[key]

        try:
            filenames = mat["filenames"]
            row = np.argwhere(filenames == e["img"])[0][0]
            side = e["side"]
            img = mat["data"][side][0, 0]["image"][0, 0][row]
            gx, gy, gz = mat["data"][side][0, 0]["gaze"][0, 0][row]
        except (KeyError, IndexError):
            skipped += 1
            continue

        # dataset gaze vector -> (pitch, yaw), exactly as the zoo's loader does
        theta = np.arcsin(-gy)
        phi = np.arctan2(-gx, -gz)
        gt = np.array([-theta, phi], dtype=np.float64)

        # the zoo mirrors the right eye and predicts a mirrored yaw
        patch = cv2.resize(img, (160, 96))
        patch = cv2.equalizeHist(patch)
        if side == "right":
            patch = np.fliplr(patch).copy()
            gt[1] = -gt[1]

        pred = model.run(gaze_input(patch))["gaze_pitchyaw"].reshape(-1)[:2].astype(np.float64)

        err = angular_error(pred, gt)
        per_person.setdefault(e["person"], []).append((pred, gt, err))
        rows.append({
            "person": e["person"], "day": e["day"], "img": e["img"], "side": side,
            "gt_pitch_deg": round(float(np.degrees(gt[0])), 3),
            "gt_yaw_deg": round(float(np.degrees(gt[1])), 3),
            "pred_pitch_deg": round(float(np.degrees(pred[0])), 3),
            "pred_yaw_deg": round(float(np.degrees(pred[1])), 3),
            "angular_error_deg": round(err, 3),
        })
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(entries)}")

    if not rows:
        sys.exit("nothing scored; check the dataset layout")

    all_err = [r["angular_error_deg"] for r in rows]
    print(f"\nscored {len(rows)} samples ({skipped} skipped)")
    print(f"\n{'':<22}{'mean':>9}{'median':>9}{'p95':>9}")
    print("-" * 49)
    print(f"{'as measured':<22}{statistics.mean(all_err):>9.2f}"
          f"{statistics.median(all_err):>9.2f}"
          f"{sorted(all_err)[int(0.95 * (len(all_err) - 1))]:>9.2f}")
    print(f"published (AI Hub)    {'16.80':>9}")

    # --- the calibration question
    # A 3-second calibration measures one offset per person and subtracts it. Simulate
    # exactly that: per person, take the mean (pitch, yaw) residual and remove it.
    print("\ncalibration test: subtract each person's mean residual")
    print(f"{'person':<10}{'n':>5}{'raw mean':>11}{'calibrated':>12}{'bias':>8}{'gain':>8}")
    print("-" * 54)
    raw_all, cal_all = [], []
    cal_rows = []
    for person, items in sorted(per_person.items()):
        if len(items) < 8:
            continue
        resid = np.array([p - g for p, g, _ in items])
        bias = resid.mean(axis=0)
        raw = [e for _, _, e in items]
        cal = [angular_error(p - bias, g) for p, g, _ in items]
        raw_all.extend(raw)
        cal_all.extend(cal)
        bias_mag = float(np.degrees(np.linalg.norm(bias)))
        print(f"{person:<10}{len(items):>5}{statistics.mean(raw):>11.2f}"
              f"{statistics.mean(cal):>12.2f}{bias_mag:>8.2f}"
              f"{100 * (1 - statistics.mean(cal) / statistics.mean(raw)):>7.0f}%")
        cal_rows.append({
            "person": person, "n": len(items),
            "raw_mean_deg": round(statistics.mean(raw), 3),
            "calibrated_mean_deg": round(statistics.mean(cal), 3),
            "bias_magnitude_deg": round(bias_mag, 3),
        })
    if raw_all:
        print("-" * 54)
        print(f"{'ALL':<10}{len(raw_all):>5}{statistics.mean(raw_all):>11.2f}"
              f"{statistics.mean(cal_all):>12.2f}{'':>8}"
              f"{100 * (1 - statistics.mean(cal_all) / statistics.mean(raw_all)):>7.0f}%")
        print("\nThe calibrated column is what a per-user calibration can reach. The gap")
        print("between the two columns is fixed bias; what remains is noise, and no")
        print("amount of calibration removes it.")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    if cal_rows:
        cpath = args.csv.replace(".csv", "_calibration.csv")
        with open(cpath, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(cal_rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(cal_rows)
        print(f"\nwrote {args.csv} and {cpath}")


if __name__ == "__main__":
    main()
