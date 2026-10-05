"""
Tests for the Part A / Task 1 "must-fix" items from the TRL 4 build
instructions: matching the thesis notebook's actual Grad-CAM/Score-CAM
layer, resize method, faithfulness (Deletion/Insertion AUC) metric, and
lazy model loading under a production server.

These don't (and can't, without the real model and 280 test images)
reproduce the thesis's exact published numbers — that's what
`tools/reproduce_thesis_tables.py` is for, run separately against the
real Xception_best.h5. What these verify is that each fix's *mechanism*
is actually in place and behaves correctly against a tiny dummy model.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module  # noqa: E402


# --------------------------------------------------------------------------
# A1 — Grad-CAM/Score-CAM target layer
# --------------------------------------------------------------------------
def test_a1_layer_picker_prefers_activation_over_raw_conv():
    """For a model shaped like Xception's final block (SeparableConv2D ->
    BatchNorm -> Activation, all with 4D output and "sepconv" in their
    name), the picker must choose the *activation* layer
    (`..._sepconv2_act`-style), matching the thesis's actual
    `pick_last_conv_name`, not the raw conv layer a naive type-based
    search would pick.
    """
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(32, 32, 3))
    x = layers.SeparableConv2D(8, 3, padding="same", name="block14_sepconv2")(inp)
    x = layers.BatchNormalization(name="block14_sepconv2_bn")(x)
    x = layers.Activation("relu", name="block14_sepconv2_act")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inp, x)

    picked = app_module._find_last_conv_layer(model)
    assert picked == "block14_sepconv2_act"
    assert picked != "block14_sepconv2"


def test_a1_layer_picker_falls_back_when_no_conv_name_present():
    """A model with 4D-output layers but none named with "conv"/"sepconv"
    should fall back to the last 4D layer rather than raising."""
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(16, 16, 3))
    x = layers.Lambda(lambda t: t * 2.0, name="scale")(inp)
    x = layers.Activation("relu", name="final_feature_map")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inp, x)

    picked = app_module._find_last_conv_layer(model)
    assert picked == "final_feature_map"


# --------------------------------------------------------------------------
# A3 — resize method (nearest-neighbor, matching Keras's load_img default)
# --------------------------------------------------------------------------
def test_a3_resize_uses_nearest_neighbor_not_interpolation(tmp_path):
    """A 2x2 image upscaled to 8x8 must contain only the 4 original colors
    if nearest-neighbor resizing was used. Bilinear/bicubic (PIL's
    implicit default) would blend adjacent pixels and introduce new,
    intermediate color values — this test would fail if that crept back
    in."""
    colors = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]],
        dtype=np.uint8,
    )
    img_path = tmp_path / "checker.png"
    Image.fromarray(colors).save(img_path)

    with open(img_path, "rb") as f:
        raw = f.read()

    app_module._img_h, app_module._img_w = 8, 8
    _, arr01 = app_module.load_preprocess(raw)

    seen_colors = {tuple(np.round(px * 255).astype(int)) for px in arr01.reshape(-1, 3)}
    expected_colors = {(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)}
    assert seen_colors == expected_colors, f"unexpected blended colors: {seen_colors - expected_colors}"


# --------------------------------------------------------------------------
# A2 — Deletion/Insertion faithfulness AUC
# --------------------------------------------------------------------------
def test_a2_faithfulness_auc_present_and_bounded(loaded_app, monkeypatch):
    # keep "full" mode fast for this test
    monkeypatch.setattr(app_module, "LIME_NUM_SAMPLES", 20)

    rng = np.random.default_rng(0)
    arr = (rng.random((96, 96, 3)) * 255).astype("uint8")
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")

    result = app_module.explain_image(buf.getvalue(), mode="full")

    assert "faithfulness" in result
    f = result["faithfulness"]
    for key in (
        "GradCAM_DeletionAUC",
        "GradCAM_InsertionAUC",
        "ScoreCAM_DeletionAUC",
        "ScoreCAM_InsertionAUC",
    ):
        assert key in f
        assert 0.0 <= f[key] <= 1.0, f"{key}={f[key]} out of bounds"


def test_a2_deletion_insertion_curves_match_sequential_reference(loaded_app):
    """The vectorised deletion_insertion_curves() must produce the exact
    same values as the notebook's original sequential loop — this test
    recomputes the sequential version independently and compares."""
    rng = np.random.default_rng(1)
    arr01 = rng.random((app_module._img_h, app_module._img_w, 3)).astype(np.float32)
    sal01 = rng.random((app_module._img_h, app_module._img_w)).astype(np.float32)
    class_index = 0
    steps = 10  # fewer steps = faster test; the equivalence holds at any step count

    vectorised_del, vectorised_ins = app_module.deletion_insertion_curves(
        arr01, sal01, class_index, steps=steps
    )

    # --- Independent, deliberately-sequential reference implementation ---
    h, w, _ = arr01.shape
    sal_flat = sal01.ravel()
    order = np.argsort(-sal_flat)
    base = app_module.blur_baseline(arr01, k=31)
    total = h * w

    del_img = arr01.copy()
    ins_img = base.copy()
    ref_del, ref_ins = [], []
    for s in range(steps + 1):
        k = int((s / steps) * total)
        if k > 0:
            idx = order[:k]
            rr, cc = idx // w, idx % w
            del_img[rr, cc, :] = base[rr, cc, :]
            ins_img[rr, cc, :] = arr01[rr, cc, :]
        p_del = app_module._model.predict(del_img[None, ...], verbose=0)[0][class_index]
        p_ins = app_module._model.predict(ins_img[None, ...], verbose=0)[0][class_index]
        ref_del.append(float(p_del))
        ref_ins.append(float(p_ins))

    np.testing.assert_allclose(vectorised_del, ref_del, atol=1e-5)
    np.testing.assert_allclose(vectorised_ins, ref_ins, atol=1e-5)


# --------------------------------------------------------------------------
# A4 — model loads lazily (the gunicorn fix)
# --------------------------------------------------------------------------
def test_a4_model_loads_on_first_request_without_explicit_call(dummy_model_path, monkeypatch):
    """Simulates how gunicorn actually runs the app: the module is
    imported and `app_module.load_asd_model()` is never called directly
    (unlike the `loaded_app` fixture, which calls it explicitly). The
    model should still end up loaded after the first HTTP request, via
    `before_request`."""
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_model_load_attempted", False)
    monkeypatch.setattr(app_module, "MODEL_PATH", dummy_model_path)

    assert not app_module.model_ready()

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        resp = c.get("/")
        assert resp.status_code == 200

    assert app_module.model_ready(), "model should have been loaded lazily by before_request"


def test_a4_ensure_model_loaded_only_attempts_once(monkeypatch):
    """If the model path is missing, ensure_model_loaded() shouldn't retry
    on every single request forever (wasteful) within one lazy-load
    cycle — it tries once and leaves `_model_load_attempted` set."""
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_model_load_attempted", False)
    monkeypatch.setattr(app_module, "MODEL_PATH", "/definitely/does/not/exist.h5")

    app_module.ensure_model_loaded()
    assert app_module._model_load_attempted is True
    assert app_module._model is None
