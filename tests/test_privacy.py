"""Tests for privacy.py — eye-band masking and pixelation. Pure numpy,
no model, detector, or image library needed."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import privacy  # noqa: E402

IMG = 224
FACE_BOX = (40, 30, 184, 194)
LANDMARKS = {
    "right_eye": (80, 85),
    "left_eye": (144, 85),
    "nose": (112, 125),
    "right_mouth": (92, 165),
    "left_mouth": (132, 165),
}


# --------------------------------------------------------------------------
# build_eye_band_mask
# --------------------------------------------------------------------------
def test_eye_band_mask_with_landmarks_is_centered_on_eyes():
    mask = privacy.build_eye_band_mask(LANDMARKS, FACE_BOX, IMG, IMG)
    ys, xs = np.where(mask)
    assert ys.size > 0
    band_center_y = (ys.min() + ys.max()) / 2
    # eyes are both at y=85 -> the band should be centred close to there
    assert abs(band_center_y - 85) < 5


def test_eye_band_mask_spans_full_face_width():
    mask = privacy.build_eye_band_mask(LANDMARKS, FACE_BOX, IMG, IMG)
    ys, xs = np.where(mask)
    fx1, _, fx2, _ = FACE_BOX
    assert xs.min() <= fx1 + 1
    assert xs.max() >= fx2 - 2


def test_eye_band_mask_is_within_image_bounds():
    mask = privacy.build_eye_band_mask(LANDMARKS, FACE_BOX, IMG, IMG)
    assert mask.shape == (IMG, IMG)
    assert mask.dtype == bool


def test_eye_band_mask_fallback_without_landmarks_uses_upper_face():
    mask = privacy.build_eye_band_mask(None, FACE_BOX, IMG, IMG)
    ys, xs = np.where(mask)
    assert ys.size > 0
    fx1, fy1, fx2, fy2 = FACE_BOX
    face_h = fy2 - fy1
    # should sit in the upper part of the face box (roughly 15%-45% down)
    assert ys.min() >= fy1
    assert ys.max() <= fy1 + face_h * 0.5


def test_eye_band_mask_none_when_no_face_box():
    assert privacy.build_eye_band_mask(LANDMARKS, None, IMG, IMG) is None


# --------------------------------------------------------------------------
# pixelate
# --------------------------------------------------------------------------
def test_pixelate_leaves_pixels_outside_mask_untouched():
    rgb01 = np.random.default_rng(0).random((IMG, IMG, 3)).astype(np.float32)
    mask = privacy.build_eye_band_mask(LANDMARKS, FACE_BOX, IMG, IMG)
    result = privacy.pixelate(rgb01, mask, block_size=10)
    outside = ~mask
    assert np.array_equal(result[outside], rgb01[outside])


def test_pixelate_blocks_are_locally_uniform():
    """The hallmark of pixelation: within the masked region, nearby
    pixels should collapse into identical-colored blocks rather than
    retaining the original per-pixel noise."""
    rgb01 = np.random.default_rng(1).random((IMG, IMG, 3)).astype(np.float32)
    mask = privacy.build_eye_band_mask(LANDMARKS, FACE_BOX, IMG, IMG)
    result = privacy.pixelate(rgb01, mask, block_size=10)

    ys, xs = np.where(mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    region = result[y1:y2, x1:x2]
    # the original random noise should NOT all be locally uniform -- but
    # the pixelated region should have many repeated block-identical rows
    first_row = region[0]
    n_rows_matching_first = sum(np.array_equal(region[i], first_row) for i in range(min(5, region.shape[0])))
    # several rows within the first block should be identical
    assert n_rows_matching_first >= 2


def test_pixelate_does_not_reveal_original_detail():
    """A strong sanity check: the pixelated region should have far fewer
    unique colors than the original (true pixel-level detail destroyed)."""
    rgb01 = np.random.default_rng(2).random((IMG, IMG, 3)).astype(np.float32)
    mask = privacy.build_eye_band_mask(LANDMARKS, FACE_BOX, IMG, IMG)
    result = privacy.pixelate(rgb01, mask, block_size=10)

    ys, xs = np.where(mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    original_region = rgb01[y1:y2, x1:x2]
    pixelated_region = result[y1:y2, x1:x2]

    original_unique = len(np.unique(original_region.reshape(-1, 3), axis=0))
    pixelated_unique = len(np.unique(pixelated_region.reshape(-1, 3), axis=0))
    assert pixelated_unique < original_unique / 4


def test_pixelate_with_none_mask_returns_unchanged_copy():
    rgb01 = np.random.default_rng(3).random((IMG, IMG, 3)).astype(np.float32)
    result = privacy.pixelate(rgb01, None)
    assert np.array_equal(result, rgb01)
    assert result is not rgb01  # still a copy, not the same object


def test_pixelate_with_empty_mask_returns_unchanged_copy():
    rgb01 = np.random.default_rng(4).random((IMG, IMG, 3)).astype(np.float32)
    empty_mask = np.zeros((IMG, IMG), dtype=bool)
    result = privacy.pixelate(rgb01, empty_mask)
    assert np.array_equal(result, rgb01)


def test_eye_blur_caption_mentions_original_image_was_analysed():
    assert "original image" in privacy.EYE_BLUR_CAPTION.lower()
    assert "display only" in privacy.EYE_BLUR_CAPTION.lower()
