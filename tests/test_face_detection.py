"""
Tests for face_detection.py's pure decision logic (evaluate_face_check,
compute_crop_box, eye_tilt_degrees) — none of these need real YuNet
weights, since they operate on hand-built FaceDetection objects rather
than calling the actual detector. The real detection call itself
(detect_faces) can't be tested here; verify it manually with real
weights and a real photo once you have the ONNX file in place.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import face_detection as fd  # noqa: E402


def _det(x=100, y=100, w=200, h=200, score=0.95, right_eye=None, left_eye=None):
    return fd.FaceDetection(bbox=(x, y, w, h), score=score, right_eye=right_eye, left_eye=left_eye)


# --------------------------------------------------------------------------
# evaluate_face_check — rejection paths
# --------------------------------------------------------------------------
def test_no_face_is_rejected():
    result = fd.evaluate_face_check([], used_fallback=False)
    assert result.ok is False
    assert result.reason == "no_face"
    assert "No face found" in result.message


def test_multiple_faces_is_rejected():
    result = fd.evaluate_face_check([_det(), _det(x=400)], used_fallback=False)
    assert result.ok is False
    assert result.reason == "multiple_faces"


def test_low_score_is_rejected():
    result = fd.evaluate_face_check([_det(score=0.5)], used_fallback=False)
    assert result.ok is False
    assert result.reason == "low_score"
    assert "0.50" in result.message


def test_score_exactly_at_threshold_is_accepted():
    """'a score below 0.8' rejects — 0.8 itself should pass."""
    result = fd.evaluate_face_check([_det(score=0.8)], used_fallback=False)
    assert result.ok is True


def test_too_small_is_rejected():
    result = fd.evaluate_face_check([_det(w=60)], used_fallback=False)
    assert result.ok is False
    assert result.reason == "too_small"


def test_face_width_exactly_at_threshold_is_accepted():
    """'narrower than 80px' rejects — exactly 80px should pass."""
    result = fd.evaluate_face_check([_det(w=80)], used_fallback=False)
    assert result.ok is True


# --------------------------------------------------------------------------
# evaluate_face_check — Haar fallback mode (no score, no landmarks)
# --------------------------------------------------------------------------
def test_haar_fallback_skips_score_check():
    """Haar gives no score at all (score=None) — can't reject on it."""
    det = fd.FaceDetection(bbox=(100, 100, 200, 200), score=None)
    result = fd.evaluate_face_check([det], used_fallback=True)
    assert result.ok is True
    assert result.used_fallback is True


def test_haar_fallback_still_rejects_too_small():
    det = fd.FaceDetection(bbox=(100, 100, 50, 50), score=None)
    result = fd.evaluate_face_check([det], used_fallback=True)
    assert result.ok is False
    assert result.reason == "too_small"


def test_haar_fallback_has_no_tilt_warning():
    """No landmarks in fallback mode -> tilt can't be computed, so no
    warning is possible (not a false negative, just not checkable)."""
    det = fd.FaceDetection(bbox=(100, 100, 200, 200), score=None)
    result = fd.evaluate_face_check([det], used_fallback=True)
    assert result.ok is True
    assert result.tilt_warning is False
    assert result.tilt_degrees is None


# --------------------------------------------------------------------------
# eye_tilt_degrees / tilt warning
# --------------------------------------------------------------------------
def test_level_eyes_no_tilt_warning():
    det = _det(right_eye=(140, 150), left_eye=(260, 150))  # same y -> 0 degrees
    result = fd.evaluate_face_check([det], used_fallback=False)
    assert result.ok is True
    assert result.tilt_warning is False
    assert abs(result.tilt_degrees) < 1e-6


def test_steeply_tilted_face_warns_but_still_accepted():
    # right_eye and left_eye separated mostly vertically -> large angle
    det = _det(right_eye=(150, 100), left_eye=(170, 250))
    result = fd.evaluate_face_check([det], used_fallback=False)
    assert result.ok is True  # tilt warns, doesn't reject
    assert result.tilt_warning is True
    assert "tilted" in result.tilt_message.lower()


def test_tilt_just_under_threshold_no_warning():
    # Construct eyes at just under 20 degrees
    dx = 120.0
    dy = dx * math.tan(math.radians(19.0))
    det = _det(right_eye=(140, 150), left_eye=(140 + dx, 150 + dy))
    result = fd.evaluate_face_check([det], used_fallback=False)
    assert result.tilt_warning is False


def test_tilt_just_over_threshold_warns():
    dx = 120.0
    dy = dx * math.tan(math.radians(21.0))
    det = _det(right_eye=(140, 150), left_eye=(140 + dx, 150 + dy))
    result = fd.evaluate_face_check([det], used_fallback=False)
    assert result.tilt_warning is True


# --------------------------------------------------------------------------
# compute_crop_box
# --------------------------------------------------------------------------
def test_no_crop_when_face_already_fills_image():
    # face width = 500 out of 800 wide image = 62.5%, above the 50% threshold
    box = fd.compute_crop_box(face_bbox=(150, 100, 500, 500), image_w=800, image_h=800)
    assert box is None


def test_crop_computed_when_face_is_small():
    # face is small relative to a big image -> should get a crop box
    box = fd.compute_crop_box(
        face_bbox=(900, 500, 100, 100), image_w=2000, image_h=1500, target_face_fraction=0.5
    )
    assert box is not None
    x1, y1, x2, y2 = box
    crop_w, crop_h = x2 - x1, y2 - y1
    # face (100px) should be ~50% of crop_w per target_face_fraction=0.5
    assert abs(100 / crop_w - 0.5) < 0.01
    assert abs(100 / crop_h - 0.5) < 0.01


def test_crop_box_is_centered_on_face():
    box = fd.compute_crop_box(
        face_bbox=(900, 500, 100, 100), image_w=2000, image_h=1500, target_face_fraction=0.5
    )
    x1, y1, x2, y2 = box
    face_cx, face_cy = 900 + 50, 500 + 50
    crop_cx, crop_cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert abs(crop_cx - face_cx) < 1.0
    assert abs(crop_cy - face_cy) < 1.0


def test_crop_box_clamped_to_image_bounds():
    # face near the top-left corner -> desired crop would go negative;
    # must be clamped to [0, image dimensions], never out of bounds
    box = fd.compute_crop_box(face_bbox=(5, 5, 50, 50), image_w=800, image_h=600, target_face_fraction=0.5)
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 800 and y2 <= 600


# --------------------------------------------------------------------------
# map_point_through_crop
# --------------------------------------------------------------------------
def test_map_point_through_crop_identity_when_crop_is_whole_image():
    # a point at the exact centre of a crop that equals the whole image,
    # resized to the same size, should land at the same relative position
    mapped = fd.map_point_through_crop((50, 50), crop_box=(0, 0, 100, 100), out_w=100, out_h=100)
    assert abs(mapped[0] - 50) < 1e-6
    assert abs(mapped[1] - 50) < 1e-6


def test_map_point_through_crop_scales_correctly():
    # crop is the right half of a 200x200 image (x in [100,200]), resized to 50x50
    # a point at x=150 (midpoint of the crop) should map to x=25 (midpoint of output)
    mapped = fd.map_point_through_crop((150, 100), crop_box=(100, 0, 200, 200), out_w=50, out_h=50)
    assert abs(mapped[0] - 25) < 1e-6
    assert abs(mapped[1] - 25) < 1e-6
