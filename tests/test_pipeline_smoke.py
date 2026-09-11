"""
End-to-end pipeline smoke tests, run against a tiny dummy model (see
conftest.py) rather than the real, private Xception_best.h5.

These exercise the actual Grad-CAM / Score-CAM / LIME / overlap-analysis
code paths — this is the test that would have caught the
"list indices must be integers or slices, not tuple" bug from an earlier
version of gradcam_heatmap() automatically, instead of only surfacing it
when a person clicked a button in the browser.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


def _make_image_bytes():
    """A tiny synthetic RGB image as raw bytes, like an uploaded file."""
    import io
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((96, 96, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _assert_valid_data_uri(uri):
    assert uri.startswith("data:image/png;base64,")
    # make sure it actually decodes to real image bytes, not garbage
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_quick_mode(loaded_app):
    result = app.explain_image(_make_image_bytes(), mode="quick")

    assert result["method"] == "Grad-CAM"
    assert result["prediction"] in ("Autistic", "Non-Autistic")
    assert 0 <= result["confidence"] <= 100
    assert result["confidence_tier"] in ("high", "moderate", "low")
    assert result["prob_autistic"] + result["prob_non_autistic"] == pytest.approx(100.0, abs=0.2)
    _assert_valid_data_uri(result["original_image"])
    _assert_valid_data_uri(result["gradcam_image"])
    assert "scorecam_image" not in result
    assert "lime_image" not in result
    assert isinstance(result["face_detected"], bool)
    assert isinstance(result["region_summary"], str) and len(result["region_summary"]) > 0


def test_cross_check_mode_adds_scorecam(loaded_app):
    result = app.explain_image(_make_image_bytes(), mode="cross_check")

    assert result["method"] == "Grad-CAM + Score-CAM"
    _assert_valid_data_uri(result["scorecam_image"])
    assert "lime_image" not in result
    assert "overlap" not in result


def test_full_mode_runs_lime_and_overlap_analysis(loaded_app, monkeypatch):
    # Use a tiny LIME sample count so this test runs in seconds, not minutes.
    monkeypatch.setattr(app, "LIME_NUM_SAMPLES", 20)

    result = app.explain_image(_make_image_bytes(), mode="full", true_label_display="Autistic")

    assert result["method"] == "Grad-CAM + Score-CAM + LIME + Overlap Analysis"
    _assert_valid_data_uri(result["scorecam_image"])
    _assert_valid_data_uri(result["lime_image"])
    _assert_valid_data_uri(result["panel_4_image"])

    overlap = result["overlap"]
    for key in ("grad_vs_score", "grad_vs_lime", "score_vs_lime"):
        block = overlap[key]
        _assert_valid_data_uri(block["image"])
        assert 0.0 <= block["iou"] <= 1.0
        assert -1.0 <= block["spearman"] <= 1.0

    three = overlap["three_way"]
    _assert_valid_data_uri(three["image"])
    for key in ("iou_grad_score", "iou_grad_lime", "iou_score_lime"):
        assert 0.0 <= three[key] <= 1.0


def test_gradcam_matches_model_input_size(loaded_app):
    """Regression guard for the double-wrapped-inputs Grad-CAM bug: the
    helper model must actually run and return a heatmap sized to the
    model's real input resolution, not silently produce garbage."""
    _, arr01 = app.load_preprocess(_make_image_bytes())
    heat = app.gradcam_heatmap(arr01, class_index=0)
    assert heat.shape == (app._img_h, app._img_w)
    assert heat.min() >= 0.0
    assert heat.max() <= 1.0 + 1e-6
