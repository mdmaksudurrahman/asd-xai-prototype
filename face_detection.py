# -*- coding: utf-8 -*-
"""
Face detection and the Part B / Task 2 face-check pipeline.

Primary detector: YuNet (cv2.FaceDetectorYN), which ships with OpenCV 4.12
but needs its weights downloaded separately (see README). It gives a
bounding box, a detection score, and 5 landmarks (right eye, left eye,
nose tip, right mouth corner, left mouth corner) per face.

Fallback detector: the Haar cascade already bundled with OpenCV, used
only if the YuNet weights aren't present. Haar gives a bounding box only
— no score, no landmarks — so score- and tilt-based checks are skipped
in fallback mode (count and minimum-width checks still apply).

This module is split into:
  - impure I/O: loading the detector, running it on a real image
    (needs real YuNet weights to do anything meaningful; can't be unit
    tested without them)
  - pure decision logic: given a list of detections and the image size,
    what should happen (reject / warn / proceed)? This half is fully
    unit-tested with hand-built fake detections — see
    tests/test_face_detection.py.
"""

import math
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration (Task 2's stated thresholds)
# --------------------------------------------------------------------------
YUNET_MODEL_PATH = os.environ.get(
    "ASD_YUNET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx"),
)
SCORE_THRESHOLD = 0.8  # "a score below 0.8" -> reject
MIN_FACE_WIDTH_PX = 80  # "a face narrower than 80 px" -> reject
NMS_THRESHOLD = 0.3
TOP_K = 5000
EYE_TILT_WARN_DEGREES = 20.0  # "tilted more than 20 degrees" -> warn, continue

# Fraction of the image's width/height the face should occupy after
# cropping, so an uploaded photo's crop looks like the (already tightly
# cropped) training images. Calibrated against the real 280 test images
# via tools/calibrate_face_crop_margin.py: width fraction median 0.811,
# height fraction median 0.887, averaged and rounded.
TARGET_FACE_FRACTION = 0.85
CROP_IF_FACE_FRACTION_BELOW = 0.5  # "fills less than about half" -> crop

# Minimum real file size for the YuNet ONNX weights. GitHub serves this
# file via Git LFS; a naive download (e.g. plain curl on the raw URL)
# can silently return a ~130-byte LFS *pointer* text file instead of the
# real ~230 KB binary. This catches that before OpenCV gives a confusing
# "failed to parse ONNX model" error.
YUNET_MIN_REAL_FILE_BYTES = 50_000

_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

_yunet_detector = None
_yunet_load_attempted = False
_yunet_load_lock = threading.Lock()
_yunet_load_error = None


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------
@dataclass
class FaceDetection:
    bbox: Tuple[float, float, float, float]  # (x, y, w, h), top-left origin
    score: Optional[float]  # None for Haar fallback (no score available)
    right_eye: Optional[Tuple[float, float]] = None
    left_eye: Optional[Tuple[float, float]] = None
    nose: Optional[Tuple[float, float]] = None
    right_mouth: Optional[Tuple[float, float]] = None
    left_mouth: Optional[Tuple[float, float]] = None

    @property
    def has_landmarks(self):
        return all(
            p is not None
            for p in (self.right_eye, self.left_eye, self.nose, self.right_mouth, self.left_mouth)
        )

    @property
    def has_eye_landmarks(self):
        """Just the two eyes — enough for the tilt check, even if the
        other 3 points are unavailable for some reason."""
        return self.right_eye is not None and self.left_eye is not None


@dataclass
class FaceCheckResult:
    ok: bool
    # None | "no_face" | "multiple_faces" | "low_score" | "too_small"
    reason: Optional[str]
    message: Optional[str]
    tilt_warning: bool
    tilt_message: Optional[str]
    tilt_degrees: Optional[float]
    detection: Optional[FaceDetection]  # the single accepted detection, if ok
    used_fallback: bool


# --------------------------------------------------------------------------
# YuNet loading (lazy, mirrors the main classifier's lazy-load pattern)
# --------------------------------------------------------------------------
def _verify_yunet_file(path):
    """Catch the Git-LFS-pointer-instead-of-real-file gotcha early, with a
    clear message, instead of a confusing OpenCV ONNX parse error."""
    if not os.path.exists(path):
        return f"No file at {path}."
    size = os.path.getsize(path)
    if size < YUNET_MIN_REAL_FILE_BYTES:
        with open(path, "rb") as f:
            head = f.read(200)
        if b"git-lfs" in head.lower() or b"version https" in head.lower():
            return (
                f"{path} is only {size} bytes and looks like a Git LFS pointer file, "
                "not the real model (~230 KB). A plain download (e.g. raw.githubusercontent.com "
                "via curl) can return this instead of the actual weights. Try downloading it "
                "through a regular browser, or from the Hugging Face mirror "
                "(huggingface.co/opencv/face_detection_yunet) instead."
            )
        return f"{path} is only {size} bytes — too small to be the real ~230 KB model file."
    return None


def load_yunet_detector(input_size=(320, 320)):
    """Load YuNet once. Returns True on success, False otherwise (caller
    falls back to Haar). Safe to call repeatedly; only actually loads
    once per process."""
    global _yunet_detector, _yunet_load_error

    problem = _verify_yunet_file(YUNET_MODEL_PATH)
    if problem:
        _yunet_load_error = problem
        return False

    try:
        _yunet_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            input_size,
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )
        return True
    except cv2.error as exc:
        _yunet_load_error = f"OpenCV could not load {YUNET_MODEL_PATH}: {exc}"
        _yunet_detector = None
        return False


def ensure_yunet_loaded():
    """Lazy-load, thread-safe, attempted at most once per process — same
    pattern as app.ensure_model_loaded() and for the same reason (must
    work when imported under gunicorn, not just `python app.py`)."""
    global _yunet_load_attempted
    if _yunet_detector is not None:
        return True
    with _yunet_load_lock:
        if _yunet_detector is not None:
            return True
        if _yunet_load_attempted:
            return False
        _yunet_load_attempted = True
        return load_yunet_detector()


def yunet_status():
    """Human-readable status for the About/model-card page (Task 6)."""
    if _yunet_detector is not None:
        return "YuNet loaded"
    if _yunet_load_error:
        return f"YuNet unavailable, using Haar cascade fallback: {_yunet_load_error}"
    return "YuNet not yet attempted"


# --------------------------------------------------------------------------
# Detection (impure — needs real weights to do anything meaningful)
# --------------------------------------------------------------------------
def _run_yunet(bgr_uint8):
    h, w = bgr_uint8.shape[:2]
    _yunet_detector.setInputSize((w, h))
    _, faces = _yunet_detector.detect(bgr_uint8)
    detections = []
    if faces is not None:
        for row in faces:
            x, y, fw, fh = row[0:4]
            detections.append(
                FaceDetection(
                    bbox=(float(x), float(y), float(fw), float(fh)),
                    score=float(row[14]),
                    right_eye=(float(row[4]), float(row[5])),
                    left_eye=(float(row[6]), float(row[7])),
                    nose=(float(row[8]), float(row[9])),
                    right_mouth=(float(row[10]), float(row[11])),
                    left_mouth=(float(row[12]), float(row[13])),
                )
            )
    return detections


def _run_haar_fallback(bgr_uint8):
    gray = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2GRAY)
    boxes = _haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [
        FaceDetection(bbox=(float(x), float(y), float(w), float(h)), score=None) for (x, y, w, h) in boxes
    ]


def detect_faces(rgb01_full_size):
    """Run face detection on a full-size image (float32 RGB, [0,1]).
    Returns (detections, used_fallback)."""
    bgr_uint8 = cv2.cvtColor((np.clip(rgb01_full_size, 0, 1) * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

    if ensure_yunet_loaded():
        return _run_yunet(bgr_uint8), False
    return _run_haar_fallback(bgr_uint8), True


# --------------------------------------------------------------------------
# Decision logic (pure — fully unit-testable without any real weights)
# --------------------------------------------------------------------------
def eye_tilt_degrees(detection: FaceDetection):
    """Angle of the eye line from horizontal, in degrees. None if the eye
    landmarks aren't available (Haar fallback)."""
    if not detection.has_eye_landmarks:
        return None
    (x1, y1), (x2, y2) = detection.right_eye, detection.left_eye
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def evaluate_face_check(
    detections: List[FaceDetection],
    used_fallback: bool,
    min_face_width=MIN_FACE_WIDTH_PX,
    score_threshold=SCORE_THRESHOLD,
    tilt_warn_degrees=EYE_TILT_WARN_DEGREES,
) -> FaceCheckResult:
    """The actual Task 2 policy: given what was detected, should this
    image be analysed, rejected, or analysed with a warning?"""

    def reject(reason, message):
        return FaceCheckResult(
            ok=False,
            reason=reason,
            message=message,
            tilt_warning=False,
            tilt_message=None,
            tilt_degrees=None,
            detection=None,
            used_fallback=used_fallback,
        )

    if len(detections) == 0:
        return reject("no_face", "No face found — please upload a clear, front-facing photo of one child.")

    if len(detections) > 1:
        return reject(
            "multiple_faces",
            "More than one face was detected — please upload a photo of one child only.",
        )

    det = detections[0]

    if det.score is not None and det.score < score_threshold:
        return reject(
            "low_score",
            f"The face in this photo wasn't detected clearly (confidence {det.score:.2f}, "
            f"need at least {score_threshold:.2f}) — please try a clearer, front-facing photo.",
        )

    face_w = det.bbox[2]
    if face_w < min_face_width:
        return reject(
            "too_small",
            f"The face in this photo is too small to analyse reliably "
            f"(about {face_w:.0f}px wide, need at least {min_face_width}px) — "
            "please use a closer photo.",
        )

    tilt = eye_tilt_degrees(det)
    tilt_warning = tilt is not None and abs(tilt) > tilt_warn_degrees
    tilt_message = (
        "This photo looks tilted — the result may be less reliable. " "Consider a level, front-facing photo."
        if tilt_warning
        else None
    )

    return FaceCheckResult(
        ok=True,
        reason=None,
        message=None,
        tilt_warning=tilt_warning,
        tilt_message=tilt_message,
        tilt_degrees=tilt,
        detection=det,
        used_fallback=used_fallback,
    )


# --------------------------------------------------------------------------
# Crop-to-training-distribution (pure geometry)
# --------------------------------------------------------------------------
def compute_crop_box(
    face_bbox,
    image_w,
    image_h,
    target_face_fraction=TARGET_FACE_FRACTION,
    crop_if_below=CROP_IF_FACE_FRACTION_BELOW,
):
    """If the detected face already fills roughly half or more of the
    image (already a tight crop, like the training images), returns None
    (no crop needed — use the original image). Otherwise returns a
    (x1, y1, x2, y2) box, centred on the face and sized so the face
    occupies about `target_face_fraction` of the crop, clamped to the
    image's bounds.
    """
    fx, fy, fw, fh = face_bbox
    face_fraction = max(fw / image_w, fh / image_h)
    if face_fraction >= crop_if_below:
        return None

    crop_w = fw / target_face_fraction
    crop_h = fh / target_face_fraction

    cx, cy = fx + fw / 2.0, fy + fh / 2.0
    x1, y1 = cx - crop_w / 2.0, cy - crop_h / 2.0
    x2, y2 = cx + crop_w / 2.0, cy + crop_h / 2.0

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_w), x2)
    y2 = min(float(image_h), y2)

    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def map_point_through_crop(point, crop_box, out_w, out_h):
    """Map a point in original-image coordinates into the coordinate space
    of a crop_box region that has been resized to (out_w, out_h) — used to
    carry YuNet's landmarks into the same 224x224 space as the heatmaps
    for Part B's region analysis (Phase 3)."""
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = (x2 - x1), (y2 - y1)
    px, py = point
    return (
        (px - x1) / crop_w * out_w if crop_w > 0 else 0.0,
        (py - y1) / crop_h * out_h if crop_h > 0 else 0.0,
    )
