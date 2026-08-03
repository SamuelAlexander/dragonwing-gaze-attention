"""Shared pieces for the gaze pipeline: model loading, decoding, and fusion.

Three models run per frame:

    face_det_lite  480x640 grayscale  -> face boxes
    facemap_3dmm   128x128 RGB        -> 68 landmarks + head pose
    eyegaze        96x160 grayscale   -> per-eye gaze, run twice

backend=None runs a model on the CPU; backend="htp" routes the same .tflite to the
Hexagon NPU through the QNN delegate. Nothing else changes.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:  # older board images
    from tflite_runtime.interpreter import Interpreter, load_delegate

QNN_DELEGATE = "/usr/lib/libQnnTFLiteDelegate.so"

# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------


class Stage:
    """One .tflite model, on the CPU or on the NPU.

    Outputs are returned as a dict keyed by tensor name, because all three models
    here have multiple outputs and index order is not guaranteed across exports.
    """

    def __init__(self, model_path, backend=None, num_threads=None):
        if backend:
            delegate = load_delegate(QNN_DELEGATE, options={"backend_type": backend})
            self.interp = Interpreter(
                model_path=model_path, experimental_delegates=[delegate]
            )
        elif num_threads:
            self.interp = Interpreter(model_path=model_path, num_threads=num_threads)
        else:
            self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.backend = backend

    @property
    def input_shape(self):
        return self.inp["shape"]

    def run(self, x):
        self.interp.set_tensor(self.inp["index"], x.astype(np.float32))
        self.interp.invoke()
        return {
            d["name"].split("/")[-1]: self.interp.get_tensor(d["index"])
            for d in self.outs
        }


# ---------------------------------------------------------------------------
# stage 1: face detection
# ---------------------------------------------------------------------------

# The zoo preprocesses with skimage.color.rgb2gray, whose luminance weights differ
# from OpenCV's COLOR_RGB2GRAY (0.299/0.587/0.114). Matching skimage keeps us on the
# distribution the model was trained on.
_RGB2GRAY_SKIMAGE = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)


FACE_W, FACE_H = 640, 480


def face_input(bgr):
    """BGR uint8 frame -> ((1, 480, 640, 1) float32 in [0, 1], letterbox transform).

    The detector's input is 4:3. Squashing a 16:9 webcam frame or a portrait photo
    into that distorts faces enough to wreck the boxes, so pad instead of stretch and
    hand back the (scale, pad_x, pad_y) needed to undo it.
    """
    h, w = bgr.shape[:2]
    scale = min(FACE_W / w, FACE_H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((FACE_H, FACE_W, 3), dtype=bgr.dtype)
    pad_x, pad_y = (FACE_W - nw) // 2, (FACE_H - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    gray = rgb @ _RGB2GRAY_SKIMAGE
    return gray[None, :, :, None], (scale, pad_x, pad_y)


def _peak_mask(hm):
    """3x3 max-pool peak suppression, equivalent to the zoo's F.max_pool2d(3, 1, 1)."""
    padded = np.pad(hm, 1, mode="constant", constant_values=-np.inf)
    best = hm.copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(best, padded[dy : dy + hm.shape[0], dx : dx + hm.shape[1]], out=best)
    return hm >= best


def detect_faces(out, transform, frame_w, frame_h, threshold=0.55, stride=8, max_faces=8):
    """Decode face_det_lite's three NHWC outputs into boxes in frame coordinates.

    `transform` is the (scale, pad_x, pad_y) returned by face_input.
    Returns a list of dicts with 'box' (x0, y0, x1, y1) and 'score', best first.
    """
    hm = 1.0 / (1.0 + np.exp(-out["heatmap"][0, :, :, 0]))
    box = out["bbox"][0]
    peaks = _peak_mask(hm) & (hm >= threshold)

    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return []
    order = np.argsort(hm[ys, xs])[::-1][:max_faces]

    scale, pad_x, pad_y = transform

    lmk5 = out.get("landmark")
    lmk5 = lmk5[0] if lmk5 is not None else None

    faces = []
    for i in order:
        cy, cx = int(ys[i]), int(xs[i])
        x, y, r, b = box[cy, cx, :]
        x0, y0, x1, y1 = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        # undo letterbox: remove padding, then undo the uniform scale
        x0, x1 = (x0 - pad_x) / scale, (x1 - pad_x) / scale
        y0, y1 = (y0 - pad_y) / scale, (y1 - pad_y) / scale

        face = {
            "box": (
                float(np.clip(x0, 0, frame_w - 1)),
                float(np.clip(y0, 0, frame_h - 1)),
                float(np.clip(x1, 0, frame_w - 1)),
                float(np.clip(y1, 0, frame_h - 1)),
            ),
            "score": float(hm[cy, cx]),
        }

        # The detector also regresses 5 points (two eyes, nose, two mouth corners),
        # stored as 5 x-offsets then 5 y-offsets, in cell units at the same stride.
        if lmk5 is not None:
            off = lmk5[cy, cx, :]
            px = (off[:5] + cx) * stride
            py = (off[5:] + cy) * stride
            face["landmark5"] = np.stack(
                [(px - pad_x) / scale, (py - pad_y) / scale], axis=1
            ).astype(np.float32)
        faces.append(face)
    return faces


# face_det_lite's 5 points, in its own order
DET5_RIGHT_EYE, DET5_LEFT_EYE, DET5_NOSE = 0, 1, 2


# ---------------------------------------------------------------------------
# stage 2: 68 landmarks + head pose
# ---------------------------------------------------------------------------

VERTEX_NUM, ALPHA_ID_SIZE, ALPHA_EXP_SIZE = 68, 219, 39


class FaceMesh:
    """Decodes facemap_3dmm's 265 parameters into 2D landmarks and head pose.

    Ported from the zoo's torch implementation to numpy so it runs on the board
    without pulling in torch.
    """

    def __init__(self, basis_dir):
        self.mean = np.load(os.path.join(basis_dir, "meanFace.npy")).reshape(
            3 * VERTEX_NUM, 1
        )
        self.basis_id = np.load(os.path.join(basis_dir, "shapeBasis.npy")).reshape(
            3 * VERTEX_NUM, ALPHA_ID_SIZE
        )
        self.basis_exp = np.load(os.path.join(basis_dir, "blendShape.npy")).reshape(
            3 * VERTEX_NUM, ALPHA_EXP_SIZE
        )

    @staticmethod
    def crop_input(bgr, box, size=128):
        """Face box -> (1, 128, 128, 3) float32 RGB in [0, 1], plus the integer box used."""
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        crop = bgr[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            return None, None
        resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        return rgb[None], (x0, y0, x1, y1)

    def decode(self, params, box, size=128):
        """265-vector -> (68x2 landmarks in frame coords, head pose in radians)."""
        lmk, pose, _ = self.decode3d(params, box, size)
        return lmk, pose

    def decode3d(self, params, box, size=128):
        """As decode(), plus the 3D geometry needed to draw in perspective.

        Returns (landmarks_2d, pose, geom) where geom carries the 3D vertices in the
        mesh's own camera space, the focal length, and the image-space scale that maps
        that space onto the frame.

        Note the depth here is not metric: the 3DMM pins tZ at a constant regardless of
        how far away the subject actually is. It is a canonical scale, good enough to
        draw convincing perspective and not good enough to quote as a distance.
        """
        p = np.asarray(params, dtype=np.float32).reshape(-1)

        alpha_id = p[0:219] * 3.0
        alpha_exp = p[219:258] * 0.5 + 0.5
        pitch, yaw, roll = p[258] * np.pi / 2, p[259] * np.pi / 2, p[260] * np.pi / 2
        tX, tY, tZ = p[261] * 60.0, p[262] * 60.0, 500.0
        f = p[263] * 150.0 + 450.0

        # p_matrix is a fixed rotation by -pi about x: the model's face basis is
        # y-up, the image is y-down.
        p_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        cr, sr = np.cos(-roll), np.sin(-roll)
        roll_m = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        cy_, sy_ = np.cos(-yaw), np.sin(-yaw)
        yaw_m = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], dtype=np.float32)
        cp, sp = np.cos(-pitch), np.sin(-pitch)
        pitch_m = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)

        r_matrix = yaw_m @ (pitch_m @ (p_matrix @ roll_m))

        shape = (
            self.mean
            + self.basis_id @ alpha_id.reshape(ALPHA_ID_SIZE, 1)
            + self.basis_exp @ alpha_exp.reshape(ALPHA_EXP_SIZE, 1)
        ).reshape(VERTEX_NUM, 3)

        verts = shape @ r_matrix.T
        verts[:, 0] += tX
        verts[:, 1] += tY
        verts[:, 2] += tZ

        lmk = verts[:, 0:2] * np.array([f, f], dtype=np.float32) / tZ

        # resized-crop space -> frame space
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        sx, sy = bw / size, bh / size
        lmk[:, 0] = (lmk[:, 0] + size / 2.0) * sx + x0
        lmk[:, 1] = (lmk[:, 1] + size / 2.0) * sy + y0

        geom = {
            "verts": verts,          # (68, 3) in the mesh's camera space
            "focal": float(f),
            "tZ": float(tZ),
            "to_frame": (sx, sy, float(x0), float(y0), size / 2.0),
        }
        return lmk, np.array([pitch, yaw, roll], dtype=np.float32), geom

    @staticmethod
    def project3d(p3, geom):
        """Project a point from the mesh's camera space into frame pixels.

        Uses the point's own depth rather than the constant tZ the landmark decode
        uses, so a point moved toward the camera grows the way a real one would. That
        is what gives the gaze arrow a usable Z axis.
        """
        sx, sy, x0, y0, half = geom["to_frame"]
        z = max(float(p3[2]), 1.0)
        u = float(p3[0]) * geom["focal"] / z
        v = float(p3[1]) * geom["focal"] / z
        return np.array([(u + half) * sx + x0, (v + half) * sy + y0], dtype=np.float64)


# iBUG-68 eye indices. 36-41 is the subject's RIGHT eye (image left).
RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX = list(range(42, 48))

# Outer and inner corner of each eye. The patch geometry is defined by these two
# points alone, not by the whole six-point contour.
RIGHT_EYE_CORNERS = (36, 39)
LEFT_EYE_CORNERS = (42, 45)


# ---------------------------------------------------------------------------
# stage 3: gaze
# ---------------------------------------------------------------------------

EYE_W, EYE_H = 160, 96


EYE_WIDTH_FACTOR = 1.5  # patch spans this multiple of the corner-to-corner distance


def eye_patch(gray, landmarks, corners, flip=False, ow=EYE_W, oh=EYE_H):
    """Warp an eye into the model's canonical 160x96 frame.

    This is the geometry EyeNet was trained and evaluated with: the eye centre is the
    midpoint of the two corners, it lands at exactly (ow/2, oh/2), and the patch width
    always covers 1.5x the corner-to-corner distance. A plain rectangular crop is NOT
    equivalent, because it leaves the eye at a scale and offset the model never saw.

    Returns (patch_uint8, centre_xy, inverse_transform) or (None, None, None).
    """
    p1 = np.asarray(landmarks[corners[0]], dtype=np.float64)
    p2 = np.asarray(landmarks[corners[1]], dtype=np.float64)
    eye_width = EYE_WIDTH_FACTOR * float(np.linalg.norm(p1 - p2))
    if eye_width <= 1e-3:
        return None, None, None

    cx, cy = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
    scale = ow / eye_width

    translate = np.eye(3)
    translate[:2, 2] = [-cx, -cy]
    scale_m = np.eye(3)
    scale_m[0, 0] = scale_m[1, 1] = scale
    centre = np.eye(3)
    centre[:2, 2] = [0.5 * ow, 0.5 * oh]

    transform = centre @ scale_m @ translate
    patch = cv2.warpAffine(gray, transform[:2, :], (ow, oh))
    patch = cv2.equalizeHist(patch)
    if flip:
        patch = np.fliplr(patch).copy()

    return patch, (float(cx), float(cy)), np.linalg.inv(transform)


def gaze_input(patch_uint8):
    """Canonical eye patch -> (1, 96, 160) float32. The model takes no channel dim."""
    return (patch_uint8.astype(np.float32) / 255.0)[None]


def gaze_output(out, flipped=False):
    """Raw model output -> (pitch, yaw) radians, un-mirrored if the patch was flipped."""
    g = np.asarray(out["gaze_pitchyaw"], dtype=np.float32).reshape(-1)[:2].copy()
    if flipped:
        g[1] = -g[1]
    return g


def to_gray(bgr):
    """BGR frame -> uint8 grayscale, cv2 weights, matching the reference eye pipeline."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# EyeNet's 34 landmarks, in the order its training data was built:
#   0-15  eyelid / interior margin
#   16-31 iris boundary
#   32    iris centre
#   33    eyeball centre
IRIS_BOUNDARY = slice(16, 32)
IRIS_CENTRE = 32
EYEBALL_CENTRE = 33

HEATMAP_H, HEATMAP_W = 48, 80


def eye_landmarks_to_frame(out, inv_transform, flipped):
    """EyeGaze's `landmarks` output -> (34, 2) points in frame coordinates.

    The network emits (y, x) in its 48x80 heatmap space, so this scales to the 160x96
    patch, swaps the axes, un-mirrors if the patch was flipped, then undoes the affine
    warp that produced the patch. Same sequence the reference implementation uses.

    This output is otherwise discarded; it is what makes the iris ring model-driven
    rather than a circle guessed from the face mesh.
    """
    lm = np.asarray(out["landmarks"], dtype=np.float64).reshape(-1, 2)
    if lm.shape[0] < 34:
        return None
    lm = lm * np.array([EYE_H / HEATMAP_H, EYE_W / HEATMAP_W])

    pts = np.empty((lm.shape[0], 3), dtype=np.float64)
    pts[:, 0] = (EYE_W - lm[:, 1]) if flipped else lm[:, 1]
    pts[:, 1] = lm[:, 0]
    pts[:, 2] = 1.0
    return (pts @ inv_transform.T)[:, :2]


def iris_circle(frame_landmarks):
    """Centre and radius of the iris, from the model's own iris boundary points."""
    if frame_landmarks is None:
        return None, None
    ring = frame_landmarks[IRIS_BOUNDARY]
    centre = frame_landmarks[IRIS_CENTRE]
    radius = float(np.median(np.linalg.norm(ring - centre, axis=1)))
    return centre, radius


def project_point(p3, focal, principal):
    """Pinhole projection of a camera-space point, y down, +z away from the camera."""
    z = max(float(p3[2]), 1e-3)
    return np.array([principal[0] + focal * p3[0] / z,
                     principal[1] + focal * p3[1] / z], dtype=np.float64)


# ---------------------------------------------------------------------------
# fusion: head pose + eye gaze -> one direction in camera space
# ---------------------------------------------------------------------------


def pitchyaw_to_vector(pitch, yaw):
    """Spherical (pitch, yaw) in radians -> unit vector, camera looking down +z.

    x right, y down, z forward, matching the image convention.
    """
    return np.array(
        [-np.cos(pitch) * np.sin(yaw), np.sin(pitch), -np.cos(pitch) * np.cos(yaw)],
        dtype=np.float32,
    )


def rotation_from_pose(pitch, yaw, roll):
    """Head rotation matrix from the 3DMM's pose angles, same order as the mesh decode."""
    cr, sr = np.cos(-roll), np.sin(-roll)
    roll_m = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    cy_, sy_ = np.cos(-yaw), np.sin(-yaw)
    yaw_m = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], dtype=np.float32)
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    pitch_m = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return yaw_m @ (pitch_m @ roll_m)


# The mesh decode maps its model space (y up, z back) into image space with this flip.
# The head rotation is built in model space, so to apply it to a vector that is already
# in camera convention it has to be conjugated by the same flip.
P_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def head_rotation_camera(head_pose):
    """Head rotation expressed in camera coordinates, ready to apply to a gaze vector.

    Getting this wrong is easy and quiet. Measured against MPIIGaze ground truth over
    800 eyes (scripts/check_fusion.py), applying the raw model-space rotation makes the
    fused direction 2.19 deg WORSE than ignoring head pose altogether; this form is
    0.68 deg better. Both differences are significant. Do not "simplify" this.
    """
    R = rotation_from_pose(*head_pose)
    return (P_FLIP @ R @ P_FLIP).T


def fuse(head_pose, eye_gaze):
    """Compose head orientation with eye-relative gaze into a camera-space direction.

    EyeGaze reports where the eye points relative to the head. Where the person is
    actually looking is that direction carried through the head's own rotation.
    """
    g_head = pitchyaw_to_vector(eye_gaze[0], eye_gaze[1])
    v = head_rotation_camera(head_pose) @ g_head
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def angle_between(a, b):
    """Angle between two direction vectors, in degrees."""
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))
