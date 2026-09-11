"""Unit tests for pure functions — no model, no images, run in well under a second."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


# --------------------------------------------------------------------------
# confidence_tier
# --------------------------------------------------------------------------
def test_confidence_tier_high():
    assert app.confidence_tier(0.90) == "high"
    assert app.confidence_tier(0.85) == "high"


def test_confidence_tier_moderate():
    assert app.confidence_tier(0.70) == "moderate"
    assert app.confidence_tier(0.65) == "moderate"


def test_confidence_tier_low():
    assert app.confidence_tier(0.50) == "low"
    assert app.confidence_tier(0.0) == "low"


# --------------------------------------------------------------------------
# infer_true_label_from_filename
# --------------------------------------------------------------------------
def test_infer_true_label_autistic():
    assert app.infer_true_label_from_filename("Autistic (12).jpg") == "Autistic"


def test_infer_true_label_non_autistic_underscore():
    assert app.infer_true_label_from_filename("Non_Autistic (3).png") == "Non_Autistic"


def test_infer_true_label_non_autistic_hyphen():
    assert app.infer_true_label_from_filename("non-autistic-042.jpg") == "Non_Autistic"


def test_infer_true_label_unrecognised():
    assert app.infer_true_label_from_filename("random_photo.jpg") is None


# --------------------------------------------------------------------------
# _norm01
# --------------------------------------------------------------------------
def test_norm01_scales_to_unit_range():
    m = np.array([[2.0, 4.0], [6.0, 8.0]])
    out = app._norm01(m)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_norm01_handles_flat_input():
    m = np.ones((4, 4))
    out = app._norm01(m)
    # a flat map shouldn't divide by ~0 and blow up
    assert np.all(out == 0.0)


# --------------------------------------------------------------------------
# topk_mask / iou_mask
# --------------------------------------------------------------------------
def test_topk_mask_selects_correct_fraction():
    m = np.arange(100).reshape(10, 10).astype(np.float32)
    mask = app.topk_mask(m, top_frac=0.10)
    assert mask.sum() == 10  # top 10% of 100 pixels


def test_iou_mask_identical_masks_is_one():
    m = np.zeros((5, 5), dtype=np.uint8)
    m[0:2, 0:2] = 1
    assert abs(app.iou_mask(m, m) - 1.0) < 1e-6


def test_iou_mask_disjoint_masks_is_zero():
    a = np.zeros((5, 5), dtype=np.uint8)
    b = np.zeros((5, 5), dtype=np.uint8)
    a[0, 0] = 1
    b[4, 4] = 1
    assert app.iou_mask(a, b) == 0.0


def test_iou_mask_partial_overlap():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    a[0:2, 0:2] = 1  # 4 pixels
    b[1:3, 1:3] = 1  # 4 pixels, overlapping at (1,1)
    # intersection = 1 pixel, union = 7 pixels
    assert abs(app.iou_mask(a, b) - (1 / 7)) < 1e-6


# --------------------------------------------------------------------------
# bbox_from_mask
# --------------------------------------------------------------------------
def test_bbox_from_mask_returns_correct_box():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 3:7] = 1
    box = app.bbox_from_mask(m)
    assert box == (3, 2, 6, 4)  # (x1, y1, x2, y2)


def test_bbox_from_mask_empty_returns_none():
    m = np.zeros((10, 10), dtype=np.uint8)
    assert app.bbox_from_mask(m) is None


# --------------------------------------------------------------------------
# spearman_corr
# --------------------------------------------------------------------------
def test_spearman_corr_perfect_positive():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(app.spearman_corr(a, b) - 1.0) < 1e-6


def test_spearman_corr_perfect_negative():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([40.0, 30.0, 20.0, 10.0])
    assert abs(app.spearman_corr(a, b) - (-1.0)) < 1e-6


# --------------------------------------------------------------------------
# describe_region
# --------------------------------------------------------------------------
def test_describe_region_upper_left():
    # 20x20 grid; an 8x8 hot block is 16% of the image, comfortably above
    # the 90th-percentile threshold so it's cleanly isolated.
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[0:8, 0:8] = 1.0  # hot spot near top-left
    face_box = (0, 0, 20, 20)
    text = app.describe_region(heat, face_box)
    assert "eye/upper-face" in text
    assert "left side" in text


def test_describe_region_lower_right():
    heat = np.zeros((20, 20), dtype=np.float32)
    heat[12:20, 12:20] = 1.0  # hot spot near bottom-right
    face_box = (0, 0, 20, 20)
    text = app.describe_region(heat, face_box)
    assert "mouth/lower-face" in text
    assert "right side" in text
