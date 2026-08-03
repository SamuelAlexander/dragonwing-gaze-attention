"""Draw what a given angular error means at a desk.

A mean angular error in degrees is hard to argue with and easy to misjudge. At a
typical 60 cm viewing distance, 17 degrees is about 18 cm: most of the way across a
laptop screen. This renders that as a picture, so the application's design constraint
is visible rather than asserted.

    ./venv/bin/python scripts/error_cone.py -o images/error-cone.png
"""

import argparse
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BG = (250, 248, 245)
INK = (40, 40, 40)
MUTED = (130, 130, 130)
CONE = (60, 60, 240)
CAL = (80, 170, 80)
SCREEN = (200, 205, 210)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error", type=float, default=16.8, help="mean angular error, deg")
    ap.add_argument("--calibrated", type=float, default=None,
                    help="calibrated error, deg, if measured")
    ap.add_argument("--distance", type=float, default=60.0, help="viewing distance, cm")
    ap.add_argument("--screen", type=float, default=60.0, help="screen width, cm")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "images/error-cone.png"))
    args = ap.parse_args()

    W, H = 1200, 640
    img = np.full((H, W, 3), BG, dtype=np.uint8)

    px_per_cm = (W - 260) / (args.distance * 1.25)
    eye = (140, H // 2)
    screen_x = int(eye[0] + args.distance * px_per_cm)

    # screen
    half = int(args.screen / 2 * px_per_cm)
    cv2.rectangle(img, (screen_x, eye[1] - half), (screen_x + 16, eye[1] + half),
                  SCREEN, -1)
    cv2.rectangle(img, (screen_x, eye[1] - half), (screen_x + 16, eye[1] + half),
                  MUTED, 2)

    def cone(deg, colour, label, side):
        spread = args.distance * np.tan(np.radians(deg))
        dy = int(spread * px_per_cm)
        overlay = img.copy()
        pts = np.array([[eye[0], eye[1]],
                        [screen_x, eye[1] - dy],
                        [screen_x, eye[1] + dy]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)
        cv2.line(img, (eye[0], eye[1]), (screen_x, eye[1] - dy), colour, 2, cv2.LINE_AA)
        cv2.line(img, (eye[0], eye[1]), (screen_x, eye[1] + dy), colour, 2, cv2.LINE_AA)
        cv2.line(img, (screen_x + 24, eye[1] - dy), (screen_x + 24, eye[1] + dy),
                 colour, 3, cv2.LINE_AA)
        ty = eye[1] - dy - 14 if side == "up" else eye[1] + dy + 30
        cv2.putText(img, f"{label}: +/-{deg:.1f} deg = {spread * 2:.0f} cm wide",
                    (screen_x - 330, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2,
                    cv2.LINE_AA)

    cone(args.error, CONE, "uncalibrated", "up")
    if args.calibrated:
        cone(args.calibrated, CAL, "after calibration", "down")

    cv2.line(img, (eye[0], eye[1]), (screen_x, eye[1]), INK, 2, cv2.LINE_AA)
    cv2.circle(img, eye, 12, INK, -1)
    cv2.putText(img, "eye", (eye[0] - 26, eye[1] + 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, INK, 2, cv2.LINE_AA)

    cv2.arrowedLine(img, (eye[0], H - 70), (screen_x, H - 70), MUTED, 2,
                    cv2.LINE_AA, tipLength=0.02)
    cv2.putText(img, f"{args.distance:.0f} cm", ((eye[0] + screen_x) // 2 - 40, H - 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, MUTED, 2, cv2.LINE_AA)
    cv2.putText(img, f"screen {args.screen:.0f} cm wide",
                (screen_x - 150, eye[1] + half + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, MUTED, 2, cv2.LINE_AA)
    cv2.putText(img, "What the gaze error means at a desk",
                (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, INK, 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}")
    for deg in (args.error, args.calibrated):
        if deg:
            s = args.distance * np.tan(np.radians(deg)) * 2
            print(f"  {deg:5.1f} deg at {args.distance:.0f} cm = {s:5.1f} cm wide "
                  f"({100 * s / args.screen:.0f}% of a {args.screen:.0f} cm screen)")


if __name__ == "__main__":
    main()
