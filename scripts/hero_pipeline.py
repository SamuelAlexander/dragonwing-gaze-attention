"""Render the four pipeline stages as one 4:3 clip, for the README hero.

Same chain and the same outputs as the stage videos, composed for a different job: a short
loop that has to read at a glance, at GIF size, before anyone reads a word. So it differs
from the stage cards in three ways:

  - **fixed framing.** Each panel's crop is computed once over the whole clip, so the
    picture holds still and only the model outputs move. The stage cards re-crop around
    every frame's face box, which makes them jitter.
  - **one short label per panel.** Everything else is drawn instead of written: head pose
    as a box that turns with the head, and where each eye patch comes from as matching
    coloured outlines.
  - **4:3**, which holds up better than 16:9 when GitHub scales a README image down.

    ./venv/bin/python scripts/hero_pipeline.py --source test-data/sample03.mp4 \
        --start 1905 --end 2092 -o results/hero-pipeline.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    EYE_H,
    EYE_W,
    LEFT_EYE_CORNERS,
    LEFT_EYE_IDX,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_IDX,
    FaceMesh,
    Stage,
    detect_faces,
    eye_landmarks_to_frame,
    eye_patch,
    face_input,
    gaze_vector,
    head_rotation,
    gaze_input,
    gaze_output,
    iris_circle,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W, H = 1440, 1080                  # 4:3
G = 12                             # gutter
CW, CH = (W - 3 * G) // 2, (H - 3 * G) // 2

BG = (18, 18, 18)
WHITE = (245, 245, 245)
SHADOW = (20, 20, 20)
RIGHT_EYE = (235, 170, 40)         # blue, the subject's right eye
LEFT_EYE = (90, 200, 90)           # green, the subject's left eye
AMBER = (40, 190, 245)             # head pose
ORANGE = (60, 140, 240)            # stage numbers, detector points
F = cv2.FONT_HERSHEY_SIMPLEX

LABELS = ["Find the face", "Landmarks and head pose", "Cut the eye patches",
          "Gaze direction"]
LABEL_SCALE, LABEL_THICK = 0.95, 2


# --------------------------------------------------------------------------- analysis

def analyse(frame, det, mm, gm, mesh):
    """Run the chain on one frame and keep everything the four panels draw."""
    h, w = frame.shape[:2]
    net_in, transform = face_input(frame)
    faces = detect_faces(det.run(net_in), transform, w, h)
    if not faces:
        return None
    face = faces[0]
    crop_in, box_used = FaceMesh.crop_input(frame, face["box"])
    if crop_in is None:
        return None
    lmk, pose, _ = mesh.decode3d(mm.run(crop_in)["parameters_3dmm"], box_used)

    gray = to_gray(frame)
    eyes, vecs = [], []
    for side, corners, idx, flip, colour in (
            ("right", RIGHT_EYE_CORNERS, RIGHT_EYE_IDX, True, RIGHT_EYE),
            ("left", LEFT_EYE_CORNERS, LEFT_EYE_IDX, False, LEFT_EYE)):
        patch, centre, inv = eye_patch(gray, lmk, corners, flip=flip)
        if patch is None:
            return None
        out = gm.run(gaze_input(patch))
        v = gaze_vector(gaze_output(out, flipped=flip))
        vecs.append(v)
        ic, ir = iris_circle(eye_landmarks_to_frame(out, inv, flip))
        if ic is None:
            ic, ir = np.array(centre), 8.0
        # where the patch came from: its four corners carried back into the frame
        quad = np.array([[0, 0, 1], [EYE_W, 0, 1], [EYE_W, EYE_H, 1], [0, EYE_H, 1]],
                        dtype=np.float64) @ inv.T
        eyes.append({"side": side, "colour": colour, "patch": patch, "quad": quad[:, :2],
                     "iris": np.asarray(ic, float), "radius": float(ir), "v": v,
                     "idx": idx})
    mean_v = np.mean(vecs, axis=0)
    mean_v /= max(np.linalg.norm(mean_v), 1e-6)
    return {"box": face["box"], "points5": face.get("landmark5"), "lmk": lmk,
            "pose": pose, "eyes": eyes, "mean_v": mean_v}


# --------------------------------------------------------------------------- framing

def fit_aspect(x0, y0, x1, y1, aspect, fw, fh):
    """Grow a box to the target aspect ratio about its centre, then keep it in frame."""
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    # shrink to the frame without changing the aspect, or the panel would be stretched
    if bh > fh:
        bh, bw = fh, fh * aspect
    if bw > fw:
        bw, bh = fw, fw / aspect
    x0 = float(np.clip(cx - bw / 2, 0, fw - bw))
    y0 = float(np.clip(cy - bh / 2, 0, fh - bh))
    return x0, y0, x0 + bw, y0 + bh


def framing(data, fw, fh, strip_aspect):
    boxes = np.array([d["box"] for d in data if d])
    quads = np.concatenate([e["quad"] for d in data if d for e in d["eyes"]])
    med = np.median(boxes, axis=0)
    fx, fy = (med[0] + med[2]) / 2, (med[1] + med[3]) / 2
    face_h = med[3] - med[1]

    # 1: the room, with the subject large enough to see the box
    ph = min(fh, 3.0 * face_h)
    pw = ph * CW / CH
    scene = fit_aspect(fx - pw / 2, fy - 0.38 * ph, fx + pw / 2, fy + 0.62 * ph,
                       CW / CH, fw, fh)

    # 2 and 4: the head, sized to contain it across the whole clip
    ux0, uy0 = boxes[:, 0].min(), boxes[:, 1].min()
    ux1, uy1 = boxes[:, 2].max(), boxes[:, 3].max()
    pad = 0.28 * (uy1 - uy0)
    head = fit_aspect(ux0 - pad, uy0 - pad, ux1 + pad, uy1 + pad, CW / CH, fw, fh)

    # 3: a strip across both eyes, containing every patch outline in the clip
    qx0, qy0 = quads.min(axis=0)
    qx1, qy1 = quads.max(axis=0)
    qp = 0.12 * (qx1 - qx0)
    strip = fit_aspect(qx0 - qp, qy0 - qp, qx1 + qp, qy1 + qp, strip_aspect, fw, fh)
    return scene, head, strip


def view(frame, rect, size):
    """Crop rect out of frame at size, and return a frame-to-panel point mapper."""
    x0, y0, x1, y1 = rect
    tile = cv2.resize(frame[int(y0):int(y1), int(x0):int(x1)], size,
                      interpolation=cv2.INTER_AREA)
    sx, sy = size[0] / (int(x1) - int(x0)), size[1] / (int(y1) - int(y0))

    def at(p):
        return (int(round((p[0] - int(x0)) * sx)), int(round((p[1] - int(y0)) * sy)))
    return tile, at, sx


# --------------------------------------------------------------------------- drawing

def arrow(img, a, b, colour, thick, tip=0.25):
    """An arrow with a dark under-stroke, so it reads on skin, wall and shirt alike."""
    cv2.arrowedLine(img, a, b, SHADOW, thick + 4, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(img, a, b, colour, thick, cv2.LINE_AA, tipLength=tip)


def dot(img, p, r, colour):
    cv2.circle(img, p, r + 1, SHADOW, -1, cv2.LINE_AA)
    cv2.circle(img, p, r, colour, -1, cv2.LINE_AA)


def brackets(img, p0, p1, colour, thick):
    (x0, y0), (x1, y1) = p0, p1
    L = int(0.22 * min(x1 - x0, y1 - y0))
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        for t, c in ((thick + 3, SHADOW), (thick, colour)):
            cv2.line(img, (cx, cy), (cx + dx * L, cy), c, t, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), c, t, cv2.LINE_AA)


def head_box(img, centre, size, pose):
    """A box around the head in the head's own frame, so its turn and tilt are the pose."""
    R = head_rotation(pose)
    a = size / 2.0
    # model space: corners 0-3 are the face side (+z), which ends up nearest the camera.
    # The landmarks sit on the face's surface, so a box centred on them would stick half
    # out in front of the face. Push it back so its front face sits just ahead of the
    # nose and the head fills the rest, the way a bounding box around a head should.
    back = 0.8 * a
    corners = np.array([[x, y, z - back] for z in (a, -a) for y in (-a, a) for x in (-a, a)],
                       dtype=np.float64)
    cam = corners @ R.T
    depth = 4.0 * size

    def proj(p):
        k = depth / (depth + p[2])
        return (centre[0] + p[0] * k, centre[1] + p[1] * k)

    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    # Depth cue: anything in front of the face is solid, anything the head would hide
    # fades and goes dashed, the drafting convention for hidden lines. Without this every
    # edge reads as sitting in front of the face. Depth is measured in the head's own
    # frame, so the whole front square stays solid however the head is tilted: t is 0 at
    # the box's front face and 1 at its back, and the face surface sits near t = 0.1.
    zm = corners[:, 2]
    z_front, z_back = zm.max(), zm.min()
    layers = {"solid": [], "mid": [], "far": []}
    n = 28
    for i, j in edges:
        for k in range(n):
            t0, t1 = k / n, (k + 1) / n
            p0 = cam[i] + (cam[j] - cam[i]) * t0
            p1 = cam[i] + (cam[j] - cam[i]) * t1
            zmid = zm[i] + (zm[j] - zm[i]) * 0.5 * (t0 + t1)
            t = (z_front - zmid) / max(z_front - z_back, 1e-6)
            seg = (tuple(int(round(v)) for v in proj(p0)),
                   tuple(int(round(v)) for v in proj(p1)))
            if t < 0.15:
                layers["solid"].append(seg)
            elif k % 2 == 0:                        # dashed from here back
                layers["mid" if t < 0.6 else "far"].append(seg)
    for name, alpha, thick in (("far", 0.35, 2), ("mid", 0.6, 2)):
        over = img.copy()
        for p0, p1 in layers[name]:
            cv2.line(over, p0, p1, AMBER, thick, cv2.LINE_AA)
        cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
    for t, c in ((7, SHADOW), (4, AMBER)):           # the near side on top, heavier
        for p0, p1 in layers["solid"]:
            cv2.line(img, p0, p1, c, t, cv2.LINE_AA)


def label(img, n, text):
    """A numbered tag in the corner: the only text in the clip, so it has to be legible."""
    num = str(n)
    (nw, th), _ = cv2.getTextSize(num, F, LABEL_SCALE, LABEL_THICK + 1)
    (tw, _), _ = cv2.getTextSize(text, F, LABEL_SCALE, LABEL_THICK)
    pad, gap = 14, 14
    x0, y0 = 14, 14
    x1, y1 = x0 + pad + nw + gap + tw + pad, y0 + th + 2 * pad
    over = img.copy()
    cv2.rectangle(over, (x0, y0), (x1, y1), (12, 12, 12), -1)
    cv2.addWeighted(over, 0.72, img, 0.28, 0, img)
    base = y0 + pad + th
    cv2.putText(img, num, (x0 + pad, base), F, LABEL_SCALE, ORANGE, LABEL_THICK + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + pad + nw + gap, base), F, LABEL_SCALE, WHITE,
                LABEL_THICK, cv2.LINE_AA)


def compose(frame, d, crops, prev):
    d = d or prev
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)
    scene, head, strip = crops
    cells = [(G, G), (2 * G + CW, G), (G, 2 * G + CH), (2 * G + CW, 2 * G + CH)]
    inter_frame = float(np.linalg.norm(d["eyes"][0]["iris"] - d["eyes"][1]["iris"]))

    # 1 find the face
    p1, at, _ = view(frame, scene, (CW, CH))
    x0, y0, x1, y1 = d["box"]
    brackets(p1, at((x0, y0)), at((x1, y1)), WHITE, 3)
    if d["points5"] is not None:
        for p in d["points5"]:
            dot(p1, at(p), 4, ORANGE)

    # 2 landmarks and head pose
    p2, at, s2 = view(frame, head, (CW, CH))
    centre = np.mean([at(p) for p in d["lmk"]], axis=0)
    head_box(p2, centre, 2.5 * inter_frame * s2, d["pose"])
    eye_idx = {i: e["colour"] for e in d["eyes"] for i in e["idx"]}
    for i, p in enumerate(d["lmk"]):
        dot(p2, at(p), 4 if i in eye_idx else 3, eye_idx.get(i, WHITE))

    # 3 cut the eye patches: where each comes from, above what the model receives
    strip_h = CH - 2 * G - int((CW - 3 * G) / 2 * EYE_H / EYE_W)
    p3 = np.full((CH, CW, 3), BG, dtype=np.uint8)
    top, at, _ = view(frame, strip, (CW, strip_h))
    for e in d["eyes"]:
        q = np.array([at(p) for p in e["quad"]], dtype=np.int32)
        cv2.polylines(top, [q], True, SHADOW, 7, cv2.LINE_AA)
        cv2.polylines(top, [q], True, e["colour"], 3, cv2.LINE_AA)
    p3[:strip_h] = top
    pw = (CW - 3 * G) // 2
    ph = int(pw * EYE_H / EYE_W)
    for k, e in enumerate(d["eyes"]):               # right eye sits on the image left
        px, py = G + k * (pw + G), strip_h + G
        tile = cv2.resize(cv2.cvtColor(e["patch"], cv2.COLOR_GRAY2BGR), (pw, ph),
                          interpolation=cv2.INTER_NEAREST)
        p3[py:py + ph, px:px + pw] = tile
        cv2.rectangle(p3, (px - 2, py - 2), (px + pw + 1, py + ph + 1), e["colour"], 3)

    # 4 gaze direction: one estimate per eye, and their mean
    p4, at, s4 = view(frame, head, (CW, CH))
    inter = inter_frame * s4
    for e in d["eyes"]:
        c = at(e["iris"])
        cv2.circle(p4, c, max(7, int(e["radius"] * s4)) + 2, SHADOW, 6, cv2.LINE_AA)
        cv2.circle(p4, c, max(7, int(e["radius"] * s4)), e["colour"], 3, cv2.LINE_AA)
        v = e["v"]
        arrow(p4, c, (int(c[0] + v[0] * 1.6 * inter), int(c[1] + v[1] * 1.6 * inter)),
              e["colour"], 4)
    mid = np.mean([at(e["iris"]) for e in d["eyes"]], axis=0).astype(int)
    mv = d["mean_v"]
    arrow(p4, tuple(mid), (int(mid[0] + mv[0] * 2.2 * inter), int(mid[1] + mv[1] * 2.2 * inter)),
          WHITE, 7, tip=0.22)

    for n, (panel, (cx, cy)) in enumerate(zip((p1, p2, p3, p4), cells), start=1):
        label(panel, n, LABELS[n - 1])
        canvas[cy:cy + CH, cx:cx + CW] = panel
    return canvas, d


# --------------------------------------------------------------------------- main

def ffmpeg_exe():
    """A working ffmpeg: the system one if it actually runs, else imageio-ffmpeg's.

    A Homebrew ffmpeg can be on PATH and still abort on a missing dependency, so test it
    rather than trusting `which`.
    """
    if shutil.which("ffmpeg"):
        try:
            if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0:
                return "ffmpeg"
        except OSError:
            pass
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "test-data/sample03.mp4"))
    ap.add_argument("--start", type=int, default=1905, help="first source frame")
    ap.add_argument("--end", type=int, default=2092, help="last source frame, inclusive")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default=None, help="htp on the board, omit for CPU")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "results/hero-pipeline.mp4"))
    ap.add_argument("--still", default=None, help="also write the middle frame as a JPG")
    args = ap.parse_args()

    det = Stage(os.path.join(args.models, "face_det_lite.tflite"), backend=args.backend)
    mm = Stage(os.path.join(args.models, "facemap_3dmm.tflite"), backend=args.backend)
    gm = Stage(os.path.join(args.models, "eyegaze.tflite"), backend=args.backend)
    mesh = FaceMesh(os.path.join(args.models, "3dmm"))

    # pass 1: run the chain over the clip, so the framing can be fixed before drawing
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    data, fw, fh = [], None, None
    for _ in range(args.start, args.end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        data.append(analyse(frame, det, mm, gm, mesh))
    cap.release()
    found = sum(d is not None for d in data)
    print(f"frames {args.start}-{args.start + len(data) - 1}: face in {found}/{len(data)}")
    if found == 0:
        raise SystemExit("no face anywhere in that range")

    strip_h = CH - 2 * G - int((CW - 3 * G) / 2 * EYE_H / EYE_W)
    crops = framing(data, fw, fh, CW / strip_h)

    # pass 2: draw
    raw = args.out + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    prev = next(d for d in data if d)
    middle = None
    for i, d in enumerate(data):
        ok, frame = cap.read()
        if not ok:
            break
        img, prev = compose(frame, d, crops, prev)
        writer.write(img)
        if i == len(data) // 2:
            middle = img
    cap.release()
    writer.release()

    exe = ffmpeg_exe()
    if exe:
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264",
                        "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", args.out],
                       check=True)
        os.remove(raw)
    else:
        print("  no working ffmpeg: keeping OpenCV's mp4v encoding")
        os.replace(raw, args.out)
    if args.still and middle is not None:
        cv2.imwrite(args.still, middle, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  {args.out}  {W}x{H}  {len(data)} frames at {fps:.0f} fps "
          f"= {len(data) / fps:.2f} s")


if __name__ == "__main__":
    main()
