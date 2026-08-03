"""Shared layout for the per-stage figures and videos.

Everything the guide shows about the pipeline is composed here, so the stills and the
video frames use one type scale, one grid and one palette. Composing each stage
independently and stitching afterwards is what made the first pipeline strip look like
four unrelated screenshots.

Layout is a fixed 1920x1080 card:

    +--------------------------------------------------+
    |  rule                                            |
    |  STAGE n   model_name                            |  header, 150 px
    |  one line of what it does        right-aligned   |
    +--------------------------------------------------+
    |                        |                         |
    |   panel: input         |   panel: output         |  body, 930 px
    |                        |                         |
    +--------------------------------------------------+
"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 1920, 1080
HEADER = 150
GUTTER = 28
MARGIN = 40

PAPER = (250, 249, 247)
INK = (32, 32, 32)
MUTED = (135, 133, 130)
RULE = (222, 220, 216)
ACCENT = (60, 120, 220)
GREEN = (80, 180, 90)
IRIS_R = (235, 170, 40)
IRIS_L = (90, 200, 90)
DOT = (60, 60, 235)
F = cv2.FONT_HERSHEY_SIMPLEX

# one type scale, used everywhere
TITLE = (1.15, 3)
SUBTITLE = (0.62, 1)
LABEL = (0.72, 2)
NOTE = (0.55, 1)
DATA = (0.68, 2)


def _text(img, s, org, style, colour):
    cv2.putText(img, s, org, F, style[0], colour, style[1], cv2.LINE_AA)


def _wrap(text, style, width):
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, F, style[0], style[1])[0][0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def card(stage_no, model, what, note=None):
    """Blank card with the header filled in. Returns (image, body_rect)."""
    img = np.full((H, W, 3), PAPER, dtype=np.uint8)
    _text(img, f"STAGE {stage_no}", (MARGIN, 62), (0.78, 2), ACCENT)
    x = MARGIN + cv2.getTextSize(f"STAGE {stage_no}", F, 0.78, 2)[0][0] + 24
    _text(img, model, (x, 64), TITLE, INK)
    for i, line in enumerate(_wrap(what, SUBTITLE, W - 2 * MARGIN)[:2]):
        _text(img, line, (MARGIN, 104 + i * 24), SUBTITLE, MUTED)
    if note:
        tw = cv2.getTextSize(note, F, NOTE[0], NOTE[1])[0][0]
        _text(img, note, (W - MARGIN - tw, 62), NOTE, MUTED)
    cv2.line(img, (MARGIN, HEADER - 16), (W - MARGIN, HEADER - 16), RULE, 2)
    return img, (MARGIN, HEADER, W - MARGIN, H - MARGIN)


def _fit_into(src, box_w, box_h, max_scale=None):
    s = min(box_w / src.shape[1], box_h / src.shape[0])
    if max_scale is not None:
        s = min(s, max_scale)
    return cv2.resize(src, (max(1, int(src.shape[1] * s)), max(1, int(src.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def place(img, rect, panels, weights=None):
    """Lay panels left to right inside rect.

    Each panel is (image, label, note) or (image, label, note, max_scale). Data readouts
    pass max_scale=1.0 so they are not blown up to fill a column; only imagery should
    stretch to the grid.
    """
    x0, y0, x1, y1 = rect
    n = len(panels)
    weights = weights or [1] * n
    avail_w = x1 - x0 - GUTTER * (n - 1)
    widths = [int(avail_w * wt / sum(weights)) for wt in weights]
    label_h = 46
    cursor = x0
    for i, panel in enumerate(panels):
        src, label, note = panel[0], panel[1], panel[2]
        max_scale = panel[3] if len(panel) > 3 else None
        pw = widths[i]
        px = cursor
        cursor += pw + GUTTER
        _text(img, label, (px, y0 + 26), LABEL, INK)
        if note:
            _text(img, note, (px, y0 + 46), NOTE, MUTED)
        avail_h = y1 - (y0 + label_h + 14)
        tile = _fit_into(src, pw, avail_h, max_scale)
        ty = y0 + label_h + 14 + (avail_h - tile.shape[0]) // 2
        tx = px + (pw - tile.shape[1]) // 2
        img[ty : ty + tile.shape[0], tx : tx + tile.shape[1]] = tile
        cv2.rectangle(img, (tx - 1, ty - 1),
                      (tx + tile.shape[1], ty + tile.shape[0]), RULE, 1)
    return img


def readout(lines, width, height):
    """A small panel of key/value data, for head pose and gaze angles."""
    img = np.full((height, width, 3), PAPER, dtype=np.uint8)
    y = 34
    for k, v in lines:
        _text(img, k, (10, y), NOTE, MUTED)
        _text(img, v, (150, y), DATA, INK)
        y += 34
    return img
