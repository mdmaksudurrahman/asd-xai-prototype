"""
Tests for region_analysis.py — all pure geometry/arithmetic, no model or
detector needed. Uses a 224x224 canvas with hand-placed landmarks
resembling a real frontal face layout.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import region_analysis as ra  # noqa: E402

IMG = 224
# A face box roughly centred, landmarks placed in a plausible frontal layout
FACE_BOX = (40, 30, 184, 194)  # x1,y1,x2,y2 -> 144 wide, 164 tall
LANDMARKS = {
    "right_eye": (80, 90),
    "left_eye": (144, 90),
    "nose": (112, 120),
    "right_mouth": (90, 160),
    "left_mouth": (134, 160),
}


# --------------------------------------------------------------------------
# build_region_masks — detailed mode
# --------------------------------------------------------------------------
def test_detailed_masks_cover_expected_region_names():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    assert set(masks.keys()) == {"forehead", "eyes", "nose", "mouth", "rest_of_face", "outside_face"}


def test_detailed_masks_are_mutually_exclusive_inside_face():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    named = ["forehead", "eyes", "nose", "mouth", "rest_of_face"]
    total_true = sum(masks[n].astype(int) for n in named)
    # no pixel should be claimed by more than one named face-region
    assert total_true.max() <= 1


def test_detailed_masks_face_regions_are_within_face_box():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    fx1, fy1, fx2, fy2 = FACE_BOX
    face_mask = ra._rect_mask(IMG, IMG, fx1, fy1, fx2, fy2)
    for name in ("forehead", "eyes", "nose", "mouth", "rest_of_face"):
        assert not np.any(masks[name] & ~face_mask), f"{name} leaks outside the face box"


def test_outside_face_is_everything_not_in_face_box():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    fx1, fy1, fx2, fy2 = FACE_BOX
    face_mask = ra._rect_mask(IMG, IMG, fx1, fy1, fx2, fy2)
    assert np.array_equal(masks["outside_face"], ~face_mask)


def test_eyes_region_is_above_nose_region_is_above_mouth_region():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    eyes_rows = np.where(masks["eyes"].any(axis=1))[0]
    nose_rows = np.where(masks["nose"].any(axis=1))[0]
    mouth_rows = np.where(masks["mouth"].any(axis=1))[0]
    assert eyes_rows.max() <= nose_rows.min()
    assert nose_rows.max() <= mouth_rows.min()


def test_forehead_is_above_eyes():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    forehead_rows = np.where(masks["forehead"].any(axis=1))[0]
    eyes_rows = np.where(masks["eyes"].any(axis=1))[0]
    assert forehead_rows.max() <= eyes_rows.min()


def test_rest_of_face_captures_cheeks():
    """With landmarks roughly centred, the rest-of-face mask should be
    non-empty (the cheeks flanking the nose/mouth strip)."""
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    assert masks["rest_of_face"].sum() > 0


# --------------------------------------------------------------------------
# build_region_masks — coarse mode (no landmarks, Haar fallback)
# --------------------------------------------------------------------------
def test_coarse_mode_when_no_landmarks():
    masks = ra.build_region_masks(FACE_BOX, None, IMG, IMG)
    assert set(masks.keys()) == {"face", "outside_face"}
    assert not np.any(masks["face"] & masks["outside_face"])
    # covers the whole image
    assert np.all(masks["face"] | masks["outside_face"])


# --------------------------------------------------------------------------
# compute_attention_breakdown
# --------------------------------------------------------------------------
def test_attention_breakdown_sums_to_100():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    rng = np.random.default_rng(0)
    heatmap = rng.random((IMG, IMG)).astype(np.float32)
    percentages, _ = ra.compute_attention_breakdown(heatmap, masks)
    assert abs(sum(percentages.values()) - 100.0) < 0.5  # rounding tolerance


def test_attention_breakdown_all_in_one_region():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    heatmap = np.zeros((IMG, IMG), dtype=np.float32)
    heatmap[masks["eyes"]] = 1.0  # all energy concentrated in the eyes
    percentages, strongest = ra.compute_attention_breakdown(heatmap, masks)
    assert percentages["eyes"] == 100.0
    assert strongest == "eyes"
    for name in ("forehead", "nose", "mouth", "rest_of_face", "outside_face"):
        assert percentages[name] == 0.0


def test_strongest_region_identifies_single_peak_pixel():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    heatmap = np.full((IMG, IMG), 0.01, dtype=np.float32)
    # Put a single bright peak inside the mouth region
    mouth_ys, mouth_xs = np.where(masks["mouth"])
    heatmap[mouth_ys[0], mouth_xs[0]] = 5.0
    _, strongest = ra.compute_attention_breakdown(heatmap, masks)
    assert strongest == "mouth"


def test_degenerate_all_zero_heatmap_does_not_crash():
    masks = ra.build_region_masks(FACE_BOX, LANDMARKS, IMG, IMG)
    heatmap = np.zeros((IMG, IMG), dtype=np.float32)
    percentages, strongest = ra.compute_attention_breakdown(heatmap, masks)
    assert abs(sum(percentages.values()) - 100.0) < 0.5
    assert strongest is not None


# --------------------------------------------------------------------------
# on_face_percentage / off_face_warning
# --------------------------------------------------------------------------
def test_on_face_percentage():
    percentages = {
        "eyes": 30.0,
        "nose": 20.0,
        "mouth": 10.0,
        "forehead": 5.0,
        "rest_of_face": 5.0,
        "outside_face": 30.0,
    }
    assert ra.on_face_percentage(percentages) == 70.0


def test_off_face_warning_absent_when_mostly_on_face():
    percentages = {"outside_face": 30.0}
    assert ra.off_face_warning(percentages) is None


def test_off_face_warning_present_when_mostly_off_face():
    percentages = {"outside_face": 60.0}
    msg = ra.off_face_warning(percentages)
    assert msg is not None
    assert "outside the face" in msg
    assert "unreliable" in msg.lower()


def test_off_face_warning_boundary_exactly_50_is_not_warned():
    """'less than 50%' -> exactly 50% on-face should NOT warn."""
    percentages = {"outside_face": 50.0}
    assert ra.off_face_warning(percentages) is None


def test_off_face_warning_boundary_just_under_50_warns():
    percentages = {"outside_face": 50.1}
    assert ra.off_face_warning(percentages) is not None


# --------------------------------------------------------------------------
# describe_region_detailed / describe_region_coarse
# --------------------------------------------------------------------------
def test_describe_region_detailed_mentions_all_regions_and_percentages():
    percentages = {
        "forehead": 10.0,
        "eyes": 31.0,
        "nose": 18.0,
        "mouth": 9.0,
        "rest_of_face": 0.0,
        "outside_face": 32.0,
    }
    text = ra.describe_region_detailed(percentages, strongest_region="eyes")
    assert "68%" in text  # on-face = 100-32
    assert "31% eyes" in text
    assert "18% nose" in text
    assert "9% mouth" in text
    assert "32%" in text
    assert "approximate" in text
    assert "eyes" in text.lower()


def test_describe_region_coarse_mentions_approximate():
    percentages = {"face": 70.0, "outside_face": 30.0}
    text = ra.describe_region_coarse(percentages)
    assert "70%" in text
    assert "30%" in text
    assert "approximate" in text
    assert "YuNet" in text
