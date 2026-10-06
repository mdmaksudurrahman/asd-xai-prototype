"""
Route-level tests for the Part B / Task 2 face-check wiring in app.py.

These monkeypatch face_detection.detect_faces directly (same technique
as the `client` fixture's default in conftest.py) to simulate specific
scenarios — no face, multiple faces, a tilted face, a small face — none
of which need real YuNet weights.
"""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module  # noqa: E402
import face_detection  # noqa: E402


def _image_bytes(size=96):
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((size, size, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _patch_detection(monkeypatch, detections, used_fallback=False):
    monkeypatch.setattr(face_detection, "detect_faces", lambda rgb01: (detections, used_fallback))


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------
def test_upload_rejected_when_no_face(client, monkeypatch):
    _patch_detection(monkeypatch, [])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["rejected"] is True
    assert data["reason"] == "no_face"
    assert "No face found" in data["message"]


def test_upload_rejected_when_multiple_faces(client, monkeypatch):
    det = face_detection.FaceDetection(bbox=(0, 0, 50, 50), score=0.9)
    _patch_detection(monkeypatch, [det, det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 422
    assert resp.get_json()["reason"] == "multiple_faces"


def test_sample_rejected_when_no_face(client, monkeypatch):
    _patch_detection(monkeypatch, [])
    resp = client.post("/predict/sample", data={"mode": "quick"})
    assert resp.status_code == 422
    data = resp.get_json()
    assert data["rejected"] is True
    assert data["reason"] == "no_face"
    # the sample endpoint should still report which file it tried, even on rejection
    assert data["source"] == "sample"
    assert data["filename"] in ("Autistic (1).jpg", "Non_Autistic (1).jpg")


def test_rejection_does_not_run_any_model_inference(client, monkeypatch):
    """A rejected image should short-circuit before predict()/Grad-CAM/etc
    ever run — assert predict() is never called."""
    _patch_detection(monkeypatch, [])
    calls = []
    monkeypatch.setattr(app_module, "predict", lambda *a, **kw: calls.append(1))
    client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert calls == []


# --------------------------------------------------------------------------
# Accepted, with a tilt warning (not a rejection)
# --------------------------------------------------------------------------
def test_upload_accepted_with_tilt_warning(client, monkeypatch):
    # eyes mostly vertically separated -> steep tilt, but still a valid single face
    det = face_detection.FaceDetection(bbox=(0, 0, 96, 96), score=0.95, right_eye=(30, 20), left_eye=(40, 70))
    _patch_detection(monkeypatch, [det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "rejected" not in data
    assert data["tilt_warning"] is True
    assert "tilted" in data["tilt_message"].lower()
    # the prediction itself still ran normally
    assert data["prediction"] in ("Autistic", "Non-Autistic")


def test_upload_accepted_no_tilt_warning_when_level(client, monkeypatch):
    det = face_detection.FaceDetection(bbox=(0, 0, 96, 96), score=0.95, right_eye=(30, 48), left_eye=(66, 48))
    _patch_detection(monkeypatch, [det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["tilt_warning"] is False


# --------------------------------------------------------------------------
# Crop-box wiring (face_cropped reflects whether a crop was actually applied)
# --------------------------------------------------------------------------
def test_face_cropped_false_when_face_already_fills_frame(client, monkeypatch):
    # face bbox covers the whole 96x96 image -> no crop needed
    det = face_detection.FaceDetection(bbox=(0, 0, 96, 96), score=0.95)
    _patch_detection(monkeypatch, [det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(96), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["face_cropped"] is False


def test_face_cropped_true_when_face_is_small_in_frame(client, monkeypatch):
    # 100px-wide face (passes the separate 80px minimum-width check) inside
    # a 300x300 image -> only ~33% of the frame, well under the 50%
    # crop threshold, so this should trigger a crop.
    det = face_detection.FaceDetection(bbox=(50, 50, 100, 100), score=0.95)
    _patch_detection(monkeypatch, [det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(300), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["face_cropped"] is True


# --------------------------------------------------------------------------
# Task 3: region-of-interest analysis wiring
# --------------------------------------------------------------------------
def test_detailed_region_analysis_present_with_full_landmarks(client, monkeypatch):
    """All 5 landmarks given -> the detailed per-region breakdown, the
    region figure, and region_attention should all be present."""
    det = face_detection.FaceDetection(
        bbox=(20, 20, 160, 160),
        score=0.95,
        right_eye=(60, 60),
        left_eye=(130, 60),
        nose=(95, 95),
        right_mouth=(70, 130),
        left_mouth=(120, 130),
    )
    _patch_detection(monkeypatch, [det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(200), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["region_attention"] is not None
    gradcam_region = data["region_attention"]["gradcam"]
    assert set(gradcam_region["percentages"].keys()) == {
        "forehead",
        "eyes",
        "nose",
        "mouth",
        "rest_of_face",
        "outside_face",
    }
    assert gradcam_region["strongest_region"] in gradcam_region["percentages"]
    assert "approximate" in data["region_summary"].lower()
    assert "region_figure_image" in data
    assert data["region_figure_image"].startswith("data:image/png;base64,")
    # off_face_warning is either None or a string — just check the key exists
    assert "off_face_warning" in data


def test_coarse_region_analysis_when_landmarks_incomplete(client, monkeypatch):
    """Only eyes given (no nose/mouth) -> has_landmarks is False, so this
    should fall back to the coarse face/outside-face description rather
    than crash or silently produce a detailed breakdown from partial data."""
    det = face_detection.FaceDetection(
        bbox=(20, 20, 160, 160), score=0.95, right_eye=(60, 60), left_eye=(130, 60)
    )
    _patch_detection(monkeypatch, [det])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(200), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()

    assert set(data["region_attention"]["gradcam"]["percentages"].keys()) == {"face", "outside_face"}
    # the coarse-mode wording mentions this explicitly
    assert "YuNet" in data["region_summary"]


def test_region_analysis_absent_for_haar_fallback_with_no_landmarks_at_all(client, monkeypatch):
    det = face_detection.FaceDetection(bbox=(20, 20, 160, 160), score=None)  # pure Haar-style, no landmarks
    _patch_detection(monkeypatch, [det], used_fallback=True)
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(200), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data["region_attention"]["gradcam"]["percentages"].keys()) == {"face", "outside_face"}


def test_region_analysis_absent_when_no_face_box_given_at_all(loaded_app):
    """Direct explain_image() calls without going through the face-check
    routes (tests, reproduce_thesis_tables.py) must keep working exactly
    as before Phase 3 — no region_attention, no region_figure_image, the
    old coarse quadrant-based region_summary."""
    import io

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((96, 96, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")

    result = app_module.explain_image(buf.getvalue(), mode="quick")
    assert result["region_attention"] is None
    assert "region_figure_image" not in result
    assert result["off_face_warning"] is None


# --------------------------------------------------------------------------
# Task 4: privacy (eye-band pixelation) wiring
# --------------------------------------------------------------------------
def _detailed_detection():
    return face_detection.FaceDetection(
        bbox=(20, 20, 160, 160),
        score=0.95,
        right_eye=(60, 60),
        left_eye=(130, 60),
        nose=(95, 95),
        right_mouth=(70, 130),
        left_mouth=(120, 130),
    )


def test_sample_is_always_blurred(client, monkeypatch):
    _patch_detection(monkeypatch, [_detailed_detection()])
    resp = client.post("/predict/sample", data={"mode": "quick"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["eyes_blurred"] is True
    assert data["privacy_caption"] is not None
    assert "display only" in data["privacy_caption"].lower()


def test_upload_is_unblurred_by_default(client, monkeypatch):
    _patch_detection(monkeypatch, [_detailed_detection()])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(200), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["eyes_blurred"] is False
    assert data["privacy_caption"] is None


def test_upload_blurs_when_opted_in(client, monkeypatch):
    _patch_detection(monkeypatch, [_detailed_detection()])
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(200), "test.jpg"), "mode": "quick", "blur_eyes": "true"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["eyes_blurred"] is True
    assert data["privacy_caption"] is not None


def test_model_and_xai_always_see_the_real_unblurred_image(client, monkeypatch):
    """The prediction/confidence from the SAME image analysed with
    blur_eyes on vs. off must be identical — proving the model never
    saw the pixelated version, only the display images changed. Uses
    the upload endpoint's toggle twice on identical bytes, so the image
    itself is held constant and blur is the only variable."""
    det = _detailed_detection()
    _patch_detection(monkeypatch, [det])

    raw_bytes = _image_bytes(200).read()

    resp_unblurred = client.post(
        "/predict/upload",
        data={"image": (io.BytesIO(raw_bytes), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    resp_blurred = client.post(
        "/predict/upload",
        data={"image": (io.BytesIO(raw_bytes), "test.jpg"), "mode": "quick", "blur_eyes": "true"},
        content_type="multipart/form-data",
    )
    assert resp_unblurred.status_code == 200
    assert resp_blurred.status_code == 200
    data_unblurred = resp_unblurred.get_json()
    data_blurred = resp_blurred.get_json()

    assert data_blurred["prediction"] == data_unblurred["prediction"]
    assert data_blurred["confidence"] == data_unblurred["confidence"]


def test_blurred_images_differ_from_unblurred_at_pixel_level(client, monkeypatch):
    """Sanity check the blur is actually doing something visible: the
    returned original_image must differ between blurred and unblurred
    results for the exact same photo."""
    det = _detailed_detection()
    _patch_detection(monkeypatch, [det])

    raw_bytes = _image_bytes(200).read()

    resp_unblurred = client.post(
        "/predict/upload",
        data={"image": (io.BytesIO(raw_bytes), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    resp_blurred = client.post(
        "/predict/upload",
        data={"image": (io.BytesIO(raw_bytes), "test.jpg"), "mode": "quick", "blur_eyes": "true"},
        content_type="multipart/form-data",
    )
    assert resp_blurred.get_json()["original_image"] != resp_unblurred.get_json()["original_image"]


# --------------------------------------------------------------------------
# Haar-fallback flag surfaces through to the result
# --------------------------------------------------------------------------
def test_face_detector_used_fallback_flag_is_surfaced(client, monkeypatch):
    det = face_detection.FaceDetection(bbox=(0, 0, 96, 96), score=None)  # Haar-style: no score
    _patch_detection(monkeypatch, [det], used_fallback=True)
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["face_detector_used_fallback"] is True
