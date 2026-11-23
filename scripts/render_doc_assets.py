"""Render the README's mermaid diagrams, tables and project tree as PNGs.

For sites that cannot render mermaid or markdown tables (Hackster). Everything is read
from the README itself, so the images always match the text. Headless Chrome draws each
one on a white page and the result is trimmed to its content. Writes into images/:

    diagram-pipeline.png, diagram-states.png      the two mermaid diagrams
    table-<section>[-n].png                       every table, named by its heading
    project-structure.png                         the project tree

    ./venv/bin/python scripts/render_doc_assets.py
    ./venv/bin/python scripts/render_doc_assets.py --chrome /usr/bin/chromium
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
DIAGRAMS = ["diagram-pipeline", "diagram-states"]
# one name per table, in the order they appear in the README
TABLES = ["table-app-design", "table-terms", "table-letterbox", "table-head-pose-test",
          "table-npu-latency", "table-vtcm-spill", "table-trim-heatmaps",
          "table-detector-schedule", "table-detector-desk-vs-closeup",
          "table-thresholds", "table-accuracy", "table-calibration",
          "table-backend-parity"]
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.min.js"

# Sized for a page about 700 px wide: images are drawn at 2x, so 20 px text stays above
# 14 px once the site scales the image down to its column.
STYLE = """
body{margin:0;background:#fff;color:#1a1a1a;
     font-family:-apple-system,"Helvetica Neue",Helvetica,Arial,sans-serif}
#d{display:inline-block;padding:24px}
table{border-collapse:collapse;font-size:20px;line-height:1.4;max-width:860px}
th,td{border:1px solid #c8cdd3;padding:10px 16px;text-align:left;vertical-align:top}
th{background:#eef1f4;font-weight:700}
td.nw{white-space:nowrap}
code{font-family:Menlo,Consolas,monospace;font-size:0.9em;background:#f3f4f6;
     padding:1px 5px;border-radius:4px}
pre{margin:0;font-family:Menlo,Consolas,monospace;font-size:20px;line-height:1.5;
    background:#f6f8fa;border:1px solid #d8dee4;border-radius:8px;padding:20px 26px}
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>{style}</style></head>
<body><div id="d">{body}</div>{script}</body></html>"""

MERMAID_SCRIPT = """<script src="{mermaid}"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:"default",
themeVariables:{{fontSize:"18px"}}}});</script>"""


def inline(text):
    """The inline markdown these tables use: code, bold, italics."""
    out, parts = [], re.split(r"(`[^`]*`)", text)
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
            continue
        p = html.escape(p)
        p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
        p = re.sub(r"\*(.+?)\*", r"<i>\1</i>", p)
        out.append(p)
    return "".join(out)


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_html(lines):
    head, rows = cells(lines[0]), [cells(l) for l in lines[2:]]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    # short cells such as "278.46 ms (3.6 FPS)" read badly split over two lines
    def td(c):
        h = inline(c)
        # "17.88 ms (55.9 FPS)": the rate on its own line keeps the table narrow
        h = re.sub(r" \((\d+(?:\.\d+)? FPS)\)", r"<br>(\1)", h)
        return f'<td class="nw">{h}</td>' if len(c) <= 24 else f"<td>{h}</td>"
    b = "".join("<tr>" + "".join(td(c) for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def collect(readme):
    """(name, kind, payload) for every mermaid block, table and the project tree."""
    items, heading = [], "intro"
    lines = readme.split("\n")
    i, mermaid_n = 0, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("```"):
            lang = line[3:].strip()
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            block = "\n".join(lines[i + 1 : j])
            if lang == "mermaid":
                items.append((DIAGRAMS[mermaid_n], "mermaid", block))
                mermaid_n += 1
            elif heading == "Project structure":
                items.append(("project-structure", "tree", block))
            i = j + 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-| :]+\|$",
                                                                  lines[i + 1]):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            items.append((heading, "table", lines[i:j]))
            i = j
            continue
        i += 1
    tables = [it for it in items if it[1] == "table"]
    if len(tables) != len(TABLES):
        raise SystemExit(f"README has {len(tables)} tables, TABLES names {len(TABLES)}: "
                         "update the list")
    names = iter(TABLES)
    return [(next(names), k, p) if k == "table" else (n, k, p) for n, k, p in items]


def render(chrome, name, kind, payload, tmp, out_dir):
    if kind == "mermaid":
        body = f'<div class="mermaid">{html.escape(payload)}</div>'
        script = MERMAID_SCRIPT.format(mermaid=MERMAID)
        size, budget, scale = "800,1100", "8000", "3"
    elif kind == "table":
        body, script, size, budget, scale = table_html(payload), "", "1200,1600", "1000", "2"
    else:
        body, script = f"<pre>{html.escape(payload)}</pre>", ""
        size, budget, scale = "1100,1300", "1000", "2"
    page = os.path.join(tmp, f"{name}.html")
    raw = os.path.join(tmp, f"{name}.png")
    with open(page, "w") as f:
        f.write(PAGE.format(style=STYLE, body=body, script=script))
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--force-device-scale-factor={scale}", f"--window-size={size}",
                    f"--virtual-time-budget={budget}", f"--screenshot={raw}",
                    f"file://{page}"], check=True, capture_output=True)
    im = cv2.imread(raw)
    ys, xs = np.where((im < 250).any(axis=2))
    if len(ys) == 0:
        raise SystemExit(f"{name}: blank render")
    if ys.max() >= im.shape[0] - 3 or xs.max() >= im.shape[1] - 3:
        raise SystemExit(f"{name}: clipped by the window, raise its window size")
    pad = 32
    crop = im[max(0, ys.min() - pad): ys.max() + pad, max(0, xs.min() - pad): xs.max() + pad]
    out = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(out, crop)
    print(f"  {os.path.relpath(out, ROOT)}  {crop.shape[1]}x{crop.shape[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    ap.add_argument("--chrome",
                    default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "images"))
    args = ap.parse_args()

    items = collect(open(args.readme).read())
    with tempfile.TemporaryDirectory() as tmp:
        for name, kind, payload in items:
            render(args.chrome, name, kind, payload, tmp, args.out)


if __name__ == "__main__":
    main()
