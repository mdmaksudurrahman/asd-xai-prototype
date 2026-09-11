"""HTTP-layer tests using Flask's test client — no real server process needed."""

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module  # noqa: E402


def _image_bytes():
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((96, 96, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------
# Behaviour when no model is loaded (the state a fresh checkout is in until
# someone adds their own Xception_best.h5 — this must fail helpfully, not
# crash with a 500).
# --------------------------------------------------------------------------
def test_home_page_shows_setup_banner_when_no_model(monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        resp = c.get("/")
        assert resp.status_code == 200
        assert b"No model loaded yet" in resp.data


def test_predict_upload_503_when_no_model(monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        resp = c.post(
            "/predict/upload",
            data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 503
        assert "not loaded" in resp.get_json()["error"]


def test_predict_sample_503_when_no_model(monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        resp = c.post("/predict/sample", data={"mode": "quick"})
        assert resp.status_code == 503


# --------------------------------------------------------------------------
# Behaviour with the dummy model loaded (via the `client` fixture in
# conftest.py, which also points TEST_DIR at synthetic sample images).
# --------------------------------------------------------------------------
def test_home_page_ok_with_model_loaded(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No model loaded yet" not in resp.data


def test_predict_upload_quick_returns_prediction(client):
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] in ("Autistic", "Non-Autistic")
    assert data["source"] == "upload"
    assert data["true_label"] is None


def test_predict_upload_rejects_bad_file_type(client):
    resp = client.post(
        "/predict/upload",
        data={"image": (io.BytesIO(b"not an image"), "test.txt"), "mode": "quick"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


def test_predict_upload_rejects_missing_file(client):
    resp = client.post("/predict/upload", data={"mode": "quick"}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "No image" in resp.get_json()["error"]


def test_predict_sample_returns_prediction_and_filename(client):
    resp = client.post("/predict/sample", data={"mode": "quick"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source"] == "sample"
    assert data["filename"] in ("Autistic (1).jpg", "Non_Autistic (1).jpg")
    assert data["true_label"] in ("Autistic", "Non-Autistic")


def test_predict_sample_pinned_filename_is_reused(client):
    first = client.post("/predict/sample", data={"mode": "quick"}).get_json()
    pinned_name = first["filename"]

    second = client.post("/predict/sample", data={"mode": "quick", "filename": pinned_name}).get_json()
    assert second["filename"] == pinned_name


def test_backward_compatible_detailed_flag_maps_to_cross_check(client):
    """Older clients sending detailed=true (before the 3-tier `mode` param
    existed) should still get the cross-check tier, not break."""
    resp = client.post(
        "/predict/upload",
        data={"image": (_image_bytes(), "test.jpg"), "detailed": "true"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["method"] == "Grad-CAM + Score-CAM"
