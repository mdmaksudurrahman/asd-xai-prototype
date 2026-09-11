"""
Shared fixtures for the test suite.

None of these tests use the real trained model or real test images —
those are private, sensitive data (children's facial photos) and are
deliberately excluded from version control (see .gitignore). Instead, a
tiny randomly-initialised CNN with the same layer types (Conv2D /
SeparableConv2D -> GlobalAveragePooling2D -> Dense(2, softmax)) stands in
for it, so the tests exercise the *real* code path (model loading, Grad-CAM,
Score-CAM, LIME, the overlap analysis) without needing anything private.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module  # noqa: E402


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory):
    """Build and save a tiny CNN, standing in for Xception_best.h5."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(0)
    inp = layers.Input(shape=(96, 96, 3))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(16, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inp, x)

    path = tmp_path_factory.mktemp("model") / "dummy_model.h5"
    model.save(str(path))
    return str(path)


@pytest.fixture(scope="session")
def synthetic_test_images(tmp_path_factory):
    """A couple of small synthetic 'test set' images with recognisable filenames."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("test_images")
    names = ["Autistic (1).jpg", "Non_Autistic (1).jpg"]
    for name in names:
        arr = (rng.random((120, 120, 3)) * 255).astype("uint8")
        img = Image.fromarray(arr)
        ImageDraw.Draw(img).ellipse((20, 20, 100, 100), outline=(0, 0, 0), width=3)
        img.save(d / name)
    return str(d)


@pytest.fixture
def loaded_app(dummy_model_path, synthetic_test_images, monkeypatch):
    """The Flask app module with the dummy model loaded and pointed at the
    synthetic test images, for tests that need a fully working pipeline."""
    monkeypatch.setattr(app_module, "MODEL_PATH", dummy_model_path)
    monkeypatch.setattr(app_module, "TEST_DIR", synthetic_test_images)
    ok = app_module.load_asd_model()
    assert ok, "dummy model failed to load"
    return app_module


@pytest.fixture
def client(loaded_app):
    loaded_app.app.config["TESTING"] = True
    with loaded_app.app.test_client() as c:
        yield c
