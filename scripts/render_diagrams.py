"""Render the README's mermaid diagrams to PNG, for reposts that cannot render mermaid.

Uses headless Chrome on a local page that loads Mermaid itself, so the PNGs match what
GitHub draws, then trims the result to its content. Writes images/diagram-*.png in the
order the diagrams appear in the README.

    ./venv/bin/python scripts/render_diagrams.py
    ./venv/bin/python scripts/render_diagrams.py --chrome /usr/bin/chromium
"""

import argparse
import html
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["diagram-pipeline", "diagram-states"]
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>body{{margin:0;background:#fff;font-family:Helvetica,Arial,sans-serif}}
#d{{display:inline-block;padding:24px}}</style></head><body>
<div id="d"><pre class="mermaid">{src}</pre></div>
<script src="{mermaid}"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:"default",
themeVariables:{{fontSize:"18px"}}}});</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    ap.add_argument("--chrome",
                    default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "images"))
    args = ap.parse_args()

    blocks = re.findall(r"```mermaid\n(.*?)```", open(args.readme).read(), re.S)
    if len(blocks) != len(NAMES):
        raise SystemExit(f"expected {len(NAMES)} mermaid blocks, found {len(blocks)}")

    with tempfile.TemporaryDirectory() as tmp:
        for name, src in zip(NAMES, blocks):
            page = os.path.join(tmp, f"{name}.html")
            raw = os.path.join(tmp, f"{name}.png")
            with open(page, "w") as f:
                f.write(PAGE.format(src=html.escape(src), mermaid=MERMAID))
            # a tall window, because a top-to-bottom flowchart is long; the trim below
            # removes the excess, and the edge check catches a window that was too small
            subprocess.run([args.chrome, "--headless=new", "--disable-gpu",
                            "--hide-scrollbars", "--force-device-scale-factor=2",
                            "--window-size=1100,1500", "--virtual-time-budget=8000",
                            f"--screenshot={raw}", f"file://{page}"],
                           check=True, capture_output=True)
            im = cv2.imread(raw)
            ys, xs = np.where((im < 245).any(axis=2))
            if len(ys) == 0:
                raise SystemExit(f"{name}: blank render, Mermaid did not load")
            if ys.max() >= im.shape[0] - 3 or xs.max() >= im.shape[1] - 3:
                raise SystemExit(f"{name}: clipped by the window, raise --window-size")
            pad = 40
            crop = im[max(0, ys.min() - pad): ys.max() + pad,
                      max(0, xs.min() - pad): xs.max() + pad]
            out = os.path.join(args.out, f"{name}.png")
            cv2.imwrite(out, crop)
            print(f"  {out}  {crop.shape[1]}x{crop.shape[0]}")


if __name__ == "__main__":
    main()
