"""Per-stage and end-to-end timing for the gaze pipeline.

Pin the governor before running and reset it after:

    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ./venv/bin/python scripts/bench.py --image test-data/portrait-6471976.jpg
    echo schedutil   | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

Reports invoke-only medians per model per backend, plus a full-pipeline number that
includes host-side cropping and decoding, because those do not accelerate.
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
from gaze_common import (  # noqa: E402
    LEFT_EYE_CORNERS,
    RIGHT_EYE_CORNERS,
    FaceMesh,
    Stage,
    detect_faces,
    eye_patch,
    face_input,
    fuse,
    gaze_input,
    gaze_output,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def time_invoke(stage, x, runs, warmup=5):
    for _ in range(warmup):
        stage.run(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        stage.run(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(ROOT, "test-data/portrait-6471976.jpg"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--csv", default=os.path.join(ROOT, "results/bench_log.csv"))
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")
    h, w = frame.shape[:2]
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    gov = "unknown"
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            gov = f.read().strip()
    except OSError:
        pass
    print(f"image {os.path.basename(args.image)} {w}x{h}   governor={gov}   runs={args.runs}")
    if gov != "performance":
        print("WARNING: governor is not 'performance'; timings will be noisy")

    # build the real inputs once, from a CPU pass, so every backend times the same tensors
    cpu_det = Stage(os.path.join(args.models, "face_det_lite.tflite"))
    net_in, transform = face_input(frame)
    faces = detect_faces(cpu_det.run(net_in), transform, w, h)
    if not faces:
        sys.exit("no face in the benchmark image")
    crop_in, box_used = FaceMesh.crop_input(frame, faces[0]["box"])
    cpu_mesh = Stage(os.path.join(args.models, "facemap_3dmm.tflite"))
    lmk, pose = mesh.decode(cpu_mesh.run(crop_in)["parameters_3dmm"], box_used)
    gray = to_gray(frame)
    patch, _, _ = eye_patch(gray, lmk, RIGHT_EYE_CORNERS, flip=True)
    eye_in = gaze_input(patch)

    inputs = {
        "face_det_lite": net_in,
        "facemap_3dmm": crop_in,
        "eyegaze": eye_in,
    }
    configs = [
        ("cpu-1thread", {"num_threads": 1}),
        ("cpu-8thread", {"num_threads": 8}),
        ("npu-htp", {"backend": "htp"}),
    ]

    rows = []
    print(f"\n{'model':<16}{'config':<14}{'median ms':>11}{'min':>9}{'max':>9}{'FPS':>9}")
    print("-" * 68)
    per_config_total = {}
    for name, x in inputs.items():
        path = os.path.join(args.models, f"{name}.tflite")
        for cfg_name, kw in configs:
            st = Stage(path, **kw)
            med, lo, hi = time_invoke(st, x, args.runs)
            # eyegaze runs twice per frame, once per eye
            count = 2 if name == "eyegaze" else 1
            per_config_total[cfg_name] = per_config_total.get(cfg_name, 0.0) + med * count
            print(f"{name:<16}{cfg_name:<14}{med:>11.2f}{lo:>9.2f}{hi:>9.2f}{1000 / med:>9.1f}")
            rows.append({
                "model": name, "config": cfg_name, "median_ms": round(med, 3),
                "min_ms": round(lo, 3), "max_ms": round(hi, 3),
                "fps": round(1000 / med, 2), "runs": args.runs, "governor": gov,
                "image": os.path.basename(args.image),
            })
            del st

    print("-" * 68)
    for cfg_name, total in per_config_total.items():
        print(f"{'model time only':<16}{cfg_name:<14}{total:>11.2f}{'':>18}{1000 / total:>9.1f}"
              f"   (4 invokes: det + mesh + 2x gaze)")
        rows.append({
            "model": "all-4-invokes", "config": cfg_name, "median_ms": round(total, 3),
            "min_ms": "", "max_ms": "", "fps": round(1000 / total, 2),
            "runs": args.runs, "governor": gov, "image": os.path.basename(args.image),
        })

    # --- full pipeline, including host work
    print("\nfull pipeline (model + host crop/decode):")
    for cfg_name, kw in configs:
        det = Stage(os.path.join(args.models, "face_det_lite.tflite"), **kw)
        mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"), **kw)
        gm = Stage(os.path.join(args.models, "eyegaze.tflite"), **kw)

        def one_frame():
            ni, tr = face_input(frame)
            fs = detect_faces(det.run(ni), tr, w, h)
            if not fs:
                return
            ci, bu = FaceMesh.crop_input(frame, fs[0]["box"])
            lm, ps = mesh.decode(mm.run(ci)["parameters_3dmm"], bu)
            g = to_gray(frame)
            for corners, flip in ((RIGHT_EYE_CORNERS, True), (LEFT_EYE_CORNERS, False)):
                p, _, _ = eye_patch(g, lm, corners, flip=flip)
                if p is not None:
                    fuse(ps, gaze_output(gm.run(gaze_input(p)), flipped=flip))

        for _ in range(3):
            one_frame()
        ts = []
        for _ in range(max(10, args.runs // 2)):
            t0 = time.perf_counter()
            one_frame()
            ts.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(ts)
        host = med - per_config_total[cfg_name]
        print(f"  {cfg_name:<14}{med:>9.2f} ms  {1000 / med:>6.1f} FPS   "
              f"(host share {host:.2f} ms, {100 * host / med:.0f}%)")
        rows.append({
            "model": "full-pipeline", "config": cfg_name, "median_ms": round(med, 3),
            "min_ms": round(min(ts), 3), "max_ms": round(max(ts), 3),
            "fps": round(1000 / med, 2), "runs": len(ts), "governor": gov,
            "image": os.path.basename(args.image),
        })
        del det, mm, gm

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nappended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
