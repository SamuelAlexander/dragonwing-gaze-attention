"""Download the three basis arrays facemap_3dmm needs to turn its output into landmarks.

The model returns 265 coefficients, not coordinates. Reconstructing a face from them
needs the mean face plus the identity and expression bases, which ship as AI Hub assets
rather than inside the .tflite. Run once, on the machine with the AI Hub client.

    ./venv/bin/python scripts/fetch_3dmm_basis.py
"""

import argparse
import os

import numpy as np
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ("meanFace.npy", (204,)),
    ("shapeBasis.npy", (44676,)),
    ("blendShape.npy", (7956,)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models/3dmm"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, expected in FILES:
        asset = CachedWebModelAsset.from_asset_store("facemap_3dmm", 1, name)
        arr = load_numpy(asset)
        if arr.shape != expected:
            raise SystemExit(f"{name}: expected {expected}, got {arr.shape}")
        np.save(os.path.join(args.out, name), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
