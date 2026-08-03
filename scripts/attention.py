"""Attention monitor: is the operator still looking at the thing they should be?

Runs the three-model chain on every frame, fuses head pose with eye gaze, and compares
the result against an on-task direction learned during a short calibration. Leaving the
cone starts a dwell timer; staying out past the threshold raises an alert.

    # calibrate against wherever the camera is pointed, then watch
    python scripts/attention.py --source 0 --display --calibrate

    # a video file, writing an annotated mp4
    python scripts/attention.py --source test-data/face-closeup.mp4 -o results/out.mp4

    # reuse a saved calibration
    python scripts/attention.py --source 0 --display --load-calib results/calib.json

The cone is deliberately wide. EyeGaze has a mean angular error near 17 degrees, so this
classifies attention into a region; it does not point at a pixel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    LEFT_EYE_CORNERS,
    LEFT_EYE_IDX,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_IDX,
    FaceMesh,
    Stage,
    angle_between,
    detect_faces,
    eye_landmarks_to_frame,
    eye_patch,
    face_input,
    fuse,
    gaze_input,
    gaze_output,
    iris_circle,
    to_gray,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GREEN = (80, 220, 100)
AMBER = (40, 190, 245)
RED = (60, 60, 240)
GREY = (150, 150, 150)
WHITE = (245, 245, 245)


# ---------------------------------------------------------------------------
# alert tone
# ---------------------------------------------------------------------------


class Beeper:
    """Short tone on the board's speaker, rate-limited, never loud.

    Sam's standing instruction is that this speaker is never run near full volume, so
    the amplitude is fixed low here and the mixer level is left to the caller.
    """

    def __init__(self, enabled=True, device="plughw:0,0", min_gap=1.5):
        self.enabled = enabled
        self.device = device
        self.min_gap = min_gap
        self._last = 0.0
        self._pcm = self._tone()
        self._proc = None

    @staticmethod
    def _tone(freq=880.0, seconds=0.18, rate=48000, amplitude=0.18):
        t = np.arange(int(rate * seconds), dtype=np.float32) / rate
        env = np.minimum(1.0, np.minimum(t * 60, (seconds - t) * 60))
        mono = np.sin(2 * np.pi * freq * t) * env * amplitude
        pcm = (mono * 32767).astype("<i2")
        return np.repeat(pcm, 2).tobytes()

    def maybe_beep(self):
        if not self.enabled:
            return
        now = time.time()
        if now - self._last < self.min_gap:
            return
        self._last = now
        if self._proc is not None and self._proc.poll() is None:
            return
        try:
            self._proc = subprocess.Popen(
                ["aplay", "-D", self.device, "-f", "S16_LE", "-r", "48000",
                 "-c", "2", "-q", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._proc.stdin.write(self._pcm)
            self._proc.stdin.close()
        except (OSError, BrokenPipeError):
            self.enabled = False  # no audio on this host; carry on silently


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


class GazePipeline:
    def __init__(self, models, backend="htp", detect_every=1, arrow_scale=1.9):
        self.det = Stage(os.path.join(models, "face_det_lite.tflite"), backend=backend)
        self.mesh_m = Stage(os.path.join(models, "facemap_3dmm.tflite"), backend=backend)
        self.gaze_m = Stage(os.path.join(models, "eyegaze.tflite"), backend=backend)
        self.mesh = FaceMesh(os.path.join(models, "3dmm"))
        self.detect_every = max(1, detect_every)
        self.arrow_scale = arrow_scale
        self._frame_i = 0
        self._last_box = None
        self.timings = {}

    def __call__(self, frame):
        h, w = frame.shape[:2]
        t = {}

        # A face does not teleport, so the detector can run at a lower rate than the
        # rest of the chain. The box is what everything downstream crops from.
        t0 = time.perf_counter()
        if self._frame_i % self.detect_every == 0 or self._last_box is None:
            net_in, transform = face_input(frame)
            faces = detect_faces(self.det.run(net_in), transform, w, h)
            self._last_box = faces[0] if faces else None
        t["detect"] = (time.perf_counter() - t0) * 1000
        self._frame_i += 1

        if self._last_box is None:
            self.timings = t
            return None

        t0 = time.perf_counter()
        crop_in, box_used = FaceMesh.crop_input(frame, self._last_box["box"])
        if crop_in is None:
            self.timings = t
            return None
        lmk, pose, geom = self.mesh.decode3d(
            self.mesh_m.run(crop_in)["parameters_3dmm"], box_used
        )
        t["mesh"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        gray = to_gray(frame)
        vecs, eyes = [], []
        for side, corners, idx, flip in (
            ("right", RIGHT_EYE_CORNERS, RIGHT_EYE_IDX, True),
            ("left", LEFT_EYE_CORNERS, LEFT_EYE_IDX, False),
        ):
            patch, centre, inv = eye_patch(gray, lmk, corners, flip=flip)
            if patch is None:
                continue
            out = self.gaze_m.run(gaze_input(patch))
            g = gaze_output(out, flipped=flip)
            v = fuse(pose, g)
            vecs.append(v)

            # iris from the model's own landmarks, falling back to the mesh corners
            frame_lm = eye_landmarks_to_frame(out, inv, flip)
            ic, ir = iris_circle(frame_lm)
            corner_span = float(np.linalg.norm(lmk[corners[0]] - lmk[corners[1]]))
            if ic is None:
                ic = np.array(centre, dtype=np.float64)
                ir = corner_span * 0.24
            else:
                # an iris is roughly a quarter of the corner-to-corner width; the
                # landmark fit sometimes comes back much smaller than that
                ir = max(float(ir), corner_span * 0.24)

            # project a 3D segment along the gaze so the arrow carries real perspective
            eyes.append({
                "side": side, "iris_centre": np.asarray(ic, dtype=np.float64),
                "iris_radius": float(ir), "arrow_tip": None, "gaze": v,
                "eye3d": (geom["verts"][idx].mean(axis=0).astype(np.float64)
                          if geom is not None else None),
            })
        t["gaze"] = (time.perf_counter() - t0) * 1000
        self.timings = t

        if not vecs:
            return None

        # Averaging the two eyes is the cheapest variance reduction available: the
        # per-eye errors are largely independent, so the mean is the better estimate.
        mean_v = np.mean(vecs, axis=0)
        n = np.linalg.norm(mean_v)
        if n < 1e-6:
            return None
        mean_v = mean_v / n

        # Both arrows are drawn from the averaged direction. Per-eye vectors carry
        # roughly 15 deg of independent error each, so drawing them separately makes the
        # two arrows point visibly different ways, which reads as a broken overlay
        # rather than as honest noise. The per-eye values are still returned.
        # One displacement for both eyes, from the averaged direction. Its length is
        # the projection of a unit vector onto the image plane, i.e. sin(off-axis
        # angle): zero looking straight at the lens, growing as the gaze turns away.
        arrow_delta = None
        if len(eyes) == 2:
            inter = float(np.linalg.norm(eyes[0]["iris_centre"]
                                         - eyes[1]["iris_centre"]))
        elif eyes:
            bx0, by0, bx1, by1 = self._last_box["box"]
            inter = 0.25 * float(np.hypot(bx1 - bx0, by1 - by0))
        else:
            inter = 0.0
        if inter > 0:
            arrow_delta = (np.array([mean_v[0], mean_v[1]], dtype=np.float64)
                           * self.arrow_scale * inter)

        # how nearly the gaze points down the camera axis, 0 at 45 deg, 1 head-on
        toward = float(np.clip((-mean_v[2] - 0.707) / (1.0 - 0.707), 0.0, 1.0))

        return {
            "box": self._last_box["box"],
            "score": self._last_box["score"],
            "landmarks": lmk,
            "pose": pose,
            "direction": mean_v,
            "per_eye": vecs,
            "eyes": eyes,
            "eye_centres": [tuple(e["iris_centre"]) for e in eyes],
            "toward_gain": toward,
            "arrow_delta": arrow_delta,
        }


# ---------------------------------------------------------------------------
# attention state
# ---------------------------------------------------------------------------


class AttentionState:
    """Cone test plus a dwell timer, hardened against a noisy gaze signal.

    Three things matter here, and the first two were bugs before they were features:

    - **Smooth the direction, not just the state.** A model with 17 degrees of error
      crosses a cone boundary constantly. An exponential moving average over the
      direction vector is what makes the boundary meaningful at all.
    - **Hysteresis in both directions.** Leaving the cone needs `grace` seconds, but
      re-entering it needs `clear_frames` consecutive inside frames. Without the
      second half, one lucky frame resets a dwell timer that should have kept running,
      and a genuinely inattentive subject never triggers.
    - **The clock is supplied by the caller.** On a camera that is wall time; on a file
      it must be the media clock, or the same clip gives different answers depending on
      how fast the pipeline happens to run.
    """

    def __init__(self, axis, cone_deg=25.0, alert_after=2.0, grace=0.3,
                 smoothing=0.35, clear_frames=3, exit_margin=5.0):
        self.axis = np.asarray(axis, dtype=np.float32)
        self.cone_deg = cone_deg
        self.alert_after = alert_after
        self.grace = grace
        self.smoothing = smoothing
        self.clear_frames = clear_frames
        self.exit_margin = exit_margin
        self.off_since = None
        self.away_for = 0.0
        self.state = "ON"
        self.last_angle = 0.0
        self._smoothed = None
        self._inside_run = 0

    def update(self, direction, now):
        if direction is None:
            # a lost face is not evidence of attention; hold the timer running
            angle = None
            self._inside_run = 0
            inside = False
        else:
            d = np.asarray(direction, dtype=np.float32)
            if self._smoothed is None:
                self._smoothed = d
            else:
                self._smoothed = (1 - self.smoothing) * self._smoothed + self.smoothing * d
                n = np.linalg.norm(self._smoothed)
                if n > 1e-6:
                    self._smoothed = self._smoothed / n
            angle = angle_between(self._smoothed, self.axis)
            self.last_angle = angle
            # asymmetric threshold: harder to re-enter than to leave
            threshold = (self.cone_deg if self.off_since is None
                         else self.cone_deg - self.exit_margin)
            inside = angle <= threshold
            self._inside_run = self._inside_run + 1 if inside else 0

        if inside and self._inside_run >= self.clear_frames:
            self.off_since = None
            self.away_for = 0.0
            self.state = "ON"
        elif inside and self.off_since is None:
            self.state = "ON"
        else:
            if self.off_since is None:
                self.off_since = now
            self.away_for = max(0.0, now - self.off_since - self.grace)
            if self.away_for >= self.alert_after:
                self.state = "ALERT"
            elif now - self.off_since >= self.grace:
                self.state = "OFF"
            else:
                self.state = "ON"
        return self.state, angle


def wait_for_key(cap, pipe, display, window, key="c"):
    """Idle preview with live tracking until the user presses the calibrate key.

    A timed countdown makes you race the clock, and if you are still settling when it
    expires you calibrate on the wrong direction and do not find out until you check
    the axis afterwards. Waiting for a keypress removes that whole failure mode.
    """
    print(f"press '{key}' in the window to calibrate, 'q' to quit")
    if not display:
        return True
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        res = pipe(frame)
        vis = frame.copy()
        h, w = vis.shape[:2]
        s = w / 1280.0
        if res is not None:
            for eye in res.get("eyes", []):
                draw_eye(vis, eye, s, res.get("arrow_delta"))
            x0, y0, x1, y1 = (int(v) for v in res["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), GREY, max(1, int(2 * s)))
        bh = int(96 * s)
        cv2.rectangle(vis, (0, 0), (w, bh), (24, 24, 24), -1)
        cv2.putText(vis, f"READY", (int(20 * s), int(62 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5 * s, WHITE, max(2, int(3 * s)))
        msg = ("look at your target, then press 'c'" if res is not None
               else "no face detected")
        cv2.putText(vis, msg, (int(230 * s), int(58 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    AMBER if res is not None else RED, max(1, int(2 * s)))
        cv2.imshow(window, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(key):
            return True
        if k in (27, ord("q")):
            return False


def calibrate(cap, pipe, seconds=3.0, display=False, window="attention"):
    """Collect a mean gaze direction while the subject looks at the target."""
    print(f"calibration: look at the thing you want monitored for {seconds:.0f} seconds")
    samples, t0 = [], time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe(frame)
        remaining = seconds - (time.time() - t0)
        if res is not None:
            samples.append(res["direction"])
        if display:
            vis = frame.copy()
            cv2.putText(vis, f"CALIBRATING  look at the target  {remaining:.1f}s",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, AMBER, 3)
            cv2.putText(vis, f"samples: {len(samples)}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            if res is not None:
                x0, y0, x1, y1 = (int(v) for v in res["box"])
                cv2.rectangle(vis, (x0, y0), (x1, y1), AMBER, 2)
            cv2.imshow(window, vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if len(samples) < 5:
        return None, len(samples)
    axis = np.mean(samples, axis=0)
    axis = axis / np.linalg.norm(axis)
    spread = float(np.mean([angle_between(s, axis) for s in samples]))
    return {"axis": axis.tolist(), "spread_deg": spread, "samples": len(samples)}, len(samples)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


IRIS_RIGHT = (235, 170, 40)   # blue, subject's right eye
IRIS_LEFT = (90, 200, 90)     # green, subject's left eye


def draw_eye(vis, eye, s, arrow_delta):
    """One eye: a single iris ring from the model's own landmarks, plus a gaze arrow.

    Both eyes are given the SAME 2D displacement. Computing it per eye through a
    perspective projection is geometrically correct but makes the two arrows splay
    apart when the gaze nears the camera axis, because each iris sits on a different
    side of the principal point. A shared displacement keeps them parallel always.

    The displacement length is sin(angle off the camera axis), so the arrow shrinks
    smoothly to nothing when the subject looks straight down the lens, and small gaze
    movements stay visible because sin is close to linear near zero.
    """
    centre = eye["iris_centre"]
    radius = max(7.0 * s, eye["iris_radius"])
    colour = IRIS_RIGHT if eye["side"] == "right" else IRIS_LEFT
    cx, cy = int(round(centre[0])), int(round(centre[1]))

    cv2.circle(vis, (cx, cy), int(round(radius)), colour,
               max(2, int(3 * s)), cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), max(2, int(radius * 0.22)), (255, 255, 255),
               -1, cv2.LINE_AA)

    if arrow_delta is None:
        return
    length = float((arrow_delta[0] ** 2 + arrow_delta[1] ** 2) ** 0.5)
    if length < 1.5:
        return  # genuinely dead centre
    ux, uy = arrow_delta[0] / length, arrow_delta[1] / length

    # Start at the edge of the iris ring rather than its centre. A short arrow drawn
    # from the centre disappears under the ring, which hides exactly the small
    # movements this overlay exists to show; from the edge it reads as a stub that
    # grows with the angle.
    sx_, sy_ = cx + ux * radius, cy + uy * radius
    tip = (sx_ + ux * length, sy_ + uy * length)
    cv2.arrowedLine(vis, (int(round(sx_)), int(round(sy_))),
                    (int(round(tip[0])), int(round(tip[1]))),
                    (255, 255, 255), max(2, int(3.5 * s)), cv2.LINE_AA,
                    tipLength=min(0.4, 12.0 * s / max(length, 1.0)))


def draw(frame, res, state, angle, st, fps, timings, cone_deg):
    vis = frame.copy()
    h, w = vis.shape[:2]
    s = w / 1280.0
    colour = {"ON": GREEN, "OFF": AMBER, "ALERT": RED}[state]

    if res is not None:
        x0, y0, x1, y1 = (int(v) for v in res["box"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour, max(1, int(2 * s)))
        for eye in res.get("eyes", []):
            draw_eye(vis, eye, s, res.get("arrow_delta"))

    # banner. Lay the right-hand column out after measuring the title, so a long
    # state name never collides with the readout.
    bh = int(96 * s)
    cv2.rectangle(vis, (0, 0), (w, bh), (24, 24, 24), -1)
    label = {"ON": "ON TASK", "OFF": "LOOKING AWAY", "ALERT": "ATTENTION LOST"}[state]
    title_scale = 1.5 * s
    title_thick = max(2, int(3 * s))
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick)
    cv2.putText(vis, label, (int(20 * s), int(62 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, colour, title_thick)

    col_x = int(20 * s) + tw + int(48 * s)
    if angle is None:
        txt = "no face"
    elif st.off_since is None:
        txt = f"{angle:5.1f} deg off axis  (cone {cone_deg:.0f})"
    else:
        # while off task the threshold to get back in is tighter, so show that number
        # rather than the nominal cone, which otherwise looks like a contradiction
        txt = (f"{angle:5.1f} deg off axis  "
               f"(back on below {cone_deg - st.exit_margin:.0f})")
    cv2.putText(vis, txt, (col_x, int(40 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72 * s, WHITE, max(1, int(2 * s)))

    if state in ("OFF", "ALERT"):
        remain = max(0.0, st.alert_after - st.away_for)
        bar_w = min(int(300 * s), max(int(120 * s), w - col_x - int(110 * s)))
        x, y = col_x, int(60 * s)
        frac = min(1.0, st.away_for / st.alert_after) if st.alert_after > 0 else 1.0
        cv2.rectangle(vis, (x, y), (x + bar_w, y + int(18 * s)), GREY, 1)
        cv2.rectangle(vis, (x, y), (x + int(bar_w * frac), y + int(18 * s)), colour, -1)
        cv2.putText(vis, f"{remain:4.1f}s", (x + bar_w + int(12 * s), y + int(16 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, colour, max(1, int(2 * s)))

    hud = f"{fps:4.1f} FPS"
    if timings:
        hud += "   " + "  ".join(f"{k} {v:.1f}ms" for k, v in timings.items())
    cv2.putText(vis, hud, (int(20 * s), h - int(18 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, WHITE, max(1, int(2 * s)))
    return vis


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or a video path")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--backend", default="htp", help="htp for NPU, 'cpu' for CPU")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("-o", "--out", default=None, help="write an annotated mp4")
    ap.add_argument("--cone", type=float, default=30.0,
                    help="on-task half-angle in degrees. 30 covers a typical monitor "
                         "at desk distance; measured on sample03")
    ap.add_argument("--alert-after", type=float, default=2.0)
    ap.add_argument("--detect-every", type=int, default=1)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-seconds", type=float, default=3.0)
    ap.add_argument("--start-delay", type=float, default=0.0,
                    help="deprecated, ignored: calibration now waits for the 'c' key")
    ap.add_argument("--save-calib", default=os.path.join(ROOT, "results/calib.json"))
    ap.add_argument("--load-calib", default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    backend = None if args.backend == "cpu" else args.backend
    pipe = GazePipeline(args.models, backend=backend, detect_every=args.detect_every)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        # MJPG before resolution, or the camera caps near 10 FPS at 720p
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"could not open source {args.source}")

    window = "attention"
    if args.display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # --- on-task axis
    calib = None
    if args.load_calib:
        with open(args.load_calib) as f:
            calib = json.load(f)
        print(f"loaded calibration from {args.load_calib}: axis {calib['axis']}")
    elif args.calibrate:
        if not wait_for_key(cap, pipe, args.display, window):
            cap.release()
            if args.display:
                cv2.destroyAllWindows()
            return
        calib, n = calibrate(cap, pipe, args.calib_seconds, args.display, window)
        if calib is None:
            sys.exit(f"calibration failed: only {n} usable frames. Is the face visible?")
        if not isinstance(src, int):
            # A file was rewound to the start after calibrating on its opening seconds,
            # so the annotated output covers the whole clip rather than starting after
            # the calibration window.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            pipe._frame_i = 0
            pipe._last_box = None
        print(f"calibrated on {calib['samples']} samples, "
              f"spread {calib['spread_deg']:.1f} deg")
        os.makedirs(os.path.dirname(args.save_calib), exist_ok=True)
        with open(args.save_calib, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"saved {args.save_calib}")
    if calib is None:
        # straight at the camera, the sensible default for a screen-mounted webcam
        calib = {"axis": [0.0, 0.0, -1.0], "spread_deg": 0.0, "samples": 0}
        print("no calibration: assuming the target is straight at the camera")

    st = AttentionState(calib["axis"], cone_deg=args.cone, alert_after=args.alert_after)
    beeper = Beeper(enabled=not args.no_audio)

    writer = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # A dwell timer needs the clock of the thing being watched. A live camera runs on
    # wall time; a file runs on its own timeline, and using wall time there makes the
    # result depend on how fast the pipeline is, so CPU and NPU would disagree about
    # the same clip.
    is_live = isinstance(src, int)
    media_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not is_live:
        print(f"file source: dwell timer uses the media clock at {media_fps:.1f} FPS")

    fps_ema, prev, i, alerts = 0.0, time.perf_counter(), 0, 0
    t_start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pipe(frame)
            now = time.time() if is_live else i / media_fps
            state, angle = st.update(res["direction"] if res else None, now)
            if state == "ALERT":
                beeper.maybe_beep()
                alerts += 1

            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst

            vis = draw(frame, res, state, angle, st, fps_ema, pipe.timings, args.cone)
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                        cap.get(cv2.CAP_PROP_FPS) or 30.0,
                        (vis.shape[1], vis.shape[0]),
                    )
                writer.write(vis)
            if args.display:
                cv2.imshow(window, vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
                if k == ord("r"):
                    # recalibrate without restarting, so a bad take costs seconds
                    new_calib, _ = calibrate(cap, pipe, args.calib_seconds,
                                             args.display, window)
                    if new_calib is not None:
                        calib = new_calib
                        st = AttentionState(calib["axis"], cone_deg=args.cone,
                                            alert_after=args.alert_after)
                        with open(args.save_calib, "w") as f:
                            json.dump(calib, f, indent=2)
                        print(f"recalibrated: spread {calib['spread_deg']:.1f} deg")
                    else:
                        print("recalibration failed, keeping the old axis")
            i += 1
            if args.max_frames and i >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    dur = time.time() - t_start
    print(f"\n{i} frames in {dur:.1f}s = {i / max(dur, 1e-6):.1f} FPS end-to-end")
    print(f"alert frames: {alerts}")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
