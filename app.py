# -*- coding: utf-8 -*-
"""
ASD-XAI Prototype — Flask backend
==================================

A small web app around an already-trained facial-image ASD classifier
(Xception / EfficientNet / MobileNet, matching the thesis notebook this
project is built from).

Three analysis tiers, matching three points in the thesis notebook
(ThesisXAIconsistency.ipynb):

  - "quick"       Grad-CAM only. Fast, always available.
  - "cross_check" + Score-CAM. A few seconds slower, gradient-free method.
  - "full"        The notebook's full "Paper-Ready XAI" pipeline for a
                   single image: Grad-CAM + Score-CAM + LIME, the same
                   4-panel comparison figure, and the same cross-method
                   overlap/agreement analysis (top-10% pixel IoU +
                   Spearman correlation, pairwise and 3-way) used to
                   produce RESULTS_PAPER_XAI_10/OVERLAP in the thesis.
                   This is slow (LIME alone runs 1,500 perturbed forward
                   passes, matching the notebook exactly) — expect
                   anywhere from ~20s to a few minutes on CPU.

SETUP
-----
1. Put your trained model at:      models/Xception_best.h5
   (or edit MODEL_PATH below / set the ASD_MODEL_PATH env var)
2. Put your 280 test images in:    data/test/
3. pip install -r requirements.txt
4. python app.py
5. Open http://127.0.0.1:5000

The app never assumes a specific architecture: it inspects the loaded
model to find its input size and last convolutional layer, so the same
code works for the Xception/EfficientNet/MobileNet variants mentioned in
the project proposal.
"""

import wording
import region_analysis
import privacy
import model_card
import face_detection
from skimage.segmentation import quickshift, mark_boundaries
from lime import lime_image
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import inspect
import io
import os
import random
import threading
from glob import glob

import cv2
import matplotlib

matplotlib.use("Agg")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("ASD_MODEL_PATH", os.path.join(BASE_DIR, "models", "Xception_best.h5"))
TEST_DIR = os.environ.get("ASD_TEST_DIR", os.path.join(BASE_DIR, "data", "test"))
REPORTS_DIR = os.environ.get("ASD_REPORTS_DIR", os.path.join(BASE_DIR, "reports"))
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ID_TO_LABEL = {0: "Non_Autistic", 1: "Autistic"}
LABEL_DISPLAY = {"Non_Autistic": "Non-Autistic", "Autistic": "Autistic"}

# LIME sample count — matches the notebook's `lime_vis(..., num_samples=1500)`
# exactly. This is the slow part of "full" mode; lower it (e.g. 500) in your
# own copy if you want a faster, slightly noisier LIME map.
LIME_NUM_SAMPLES = 1500
LIME_NUM_FEATURES = 10  # superpixels drawn in the green/red LIME figure
LIME_QUICKSHIFT = {"kernel_size": 4, "max_dist": 100, "ratio": 0.2}  # superpixel segmentation
TOPK_OVERLAP_FRACTION = 0.10  # matches the notebook's top_frac=0.10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload cap

# --------------------------------------------------------------------------
# Model loading (once, at startup)
# --------------------------------------------------------------------------
_model = None
_img_h, _img_w = 224, 224
_last_conv_name = None
_binary_output = False  # True if the model has a single sigmoid output
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_lime_explainer = lime_image.LimeImageExplainer()


def _find_last_conv_layer(m):
    """Pick the Grad-CAM/Score-CAM target layer the same way the thesis
    notebook's final pipeline does (`pick_last_conv_name`), NOT by layer
    *type*. This matters: for Xception it resolves to
    `block14_sepconv2_act` (the ReLU after the last separable conv's
    BatchNorm) rather than `block14_sepconv2` itself, because the matching
    is done on substrings of the layer *name* ("conv"/"sepconv"), and
    "sepconv" also appears inside "..._sepconv2_act". An earlier,
    type-based version of this function (`isinstance(layer, (Conv2D,
    SeparableConv2D))`) picked the wrong layer and produced heatmaps/IoU/
    Spearman values that didn't match the thesis figures or Tables 4.8-4.9.

    Falls back to the simpler type-based match only if no 4D-output layer
    has "conv"/"sepconv" in its name at all (e.g. a non-Xception-style
    architecture), to avoid ever raising on an otherwise-working model.
    """
    candidates = []
    for layer in m.layers:
        try:
            shape = tf.TensorShape(layer.output.shape)
        except (AttributeError, ValueError):
            continue
        if shape.rank == 4:
            candidates.append(layer.name)

    for name in reversed(candidates):
        low = name.lower()
        if "conv" in low or "sepconv" in low:
            return name

    if candidates:
        return candidates[-1]

    # Last-resort fallback: original type-based search.
    for layer in reversed(m.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            return layer.name
    raise ValueError("No 4D feature-map layer found in the model.")


def load_asd_model():
    """Load the classifier once and cache input size / last conv layer name."""
    global _model, _img_h, _img_w, _last_conv_name, _binary_output

    if not os.path.exists(MODEL_PATH):
        return False

    _model = load_model(MODEL_PATH, compile=False)

    shape = _model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _img_h = shape[1] or 224
    _img_w = shape[2] or 224

    out_shape = _model.output_shape
    if isinstance(out_shape, list):
        out_shape = out_shape[0]
    _binary_output = out_shape[-1] == 1

    _last_conv_name = _find_last_conv_layer(_model)
    return True


def model_ready():
    return _model is not None


_model_load_lock = threading.Lock()
_model_load_attempted = False


def ensure_model_loaded():
    """Load the model on first use if it isn't already.

    `if __name__ == "__main__":` only runs when the file is executed
    directly (`python app.py`). Under a production server such as
    `gunicorn app:app`, the module is only *imported* — that block never
    runs, so without this, the model would silently never load and every
    request would fail. Flask's `before_request` calls this on the first
    incoming request regardless of how the app was started.
    """
    global _model_load_attempted
    if _model is not None:
        return
    with _model_load_lock:
        if _model is not None or _model_load_attempted:
            return
        _model_load_attempted = True
        load_asd_model()


@app.before_request
def _load_model_before_first_request():
    ensure_model_loaded()


# --------------------------------------------------------------------------
# Pre/post-processing helpers
# --------------------------------------------------------------------------
def load_preprocess(path_or_bytes):
    """Load an image (path or raw bytes) -> (PIL RGB image, float32 array in [0,1])."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    else:
        pil = Image.open(path_or_bytes).convert("RGB")
    # Keras's `load_img` (used throughout the thesis notebook) defaults to
    # nearest-neighbor resizing. Match it explicitly so uploaded/sample
    # images get the same input pixels the model was trained and
    # evaluated on, rather than relying on PIL's own (different) default.
    pil_resized = pil.resize((_img_w, _img_h), resample=Image.NEAREST)
    arr = np.asarray(pil_resized).astype("float32") / 255.0
    return pil_resized, arr


def infer_true_label_from_filename(filename):
    """Best-effort ground-truth label parsed from the test-set filename (display only)."""
    name = os.path.basename(filename).lower().replace("_", "-")
    if "non-autistic" in name or "non autistic" in name:
        return "Non_Autistic"
    if "autistic" in name:
        return "Autistic"
    return None


def predict(arr01):
    """Run the classifier on a single preprocessed image array -> (pred_id, confidence, probs)."""
    x = np.expand_dims(arr01, axis=0)
    raw = _model.predict(x, verbose=0)[0]
    if _binary_output:
        p_autistic = float(raw[0])
        probs = np.array([1 - p_autistic, p_autistic])
    else:
        probs = raw
    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id])
    return pred_id, conf, probs


def detect_face_bbox(rgb01):
    """Rough face box (for the plain-language region description).

    Returns (box, found) — `found` is False when the cascade didn't detect
    a face at all, so callers can warn the user rather than silently
    explaining a heatmap over an arbitrary full-frame box.
    """
    gray = cv2.cvtColor((rgb01 * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    h, w = gray.shape[:2]
    if len(faces) == 0:
        return (0, 0, w, h), False
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (x, y, x + fw, y + fh), True


def run_face_check_on_bytes(image_bytes):
    """Part B / Task 2: run the face check on the image at full size,
    before any model inference. Returns a dict:

      {"ok": False, "reason": ..., "message": ...}                    -> reject, no analysis
      {"ok": True, "crop_box": ..., "tilt_warning": ..., ...}         -> proceed, maybe cropped

    `crop_box` is None when no crop is needed (face already fills
    roughly half the image or more — true for the real 280 test images,
    confirmed by tools/calibrate_face_crop_margin.py).
    """
    if isinstance(image_bytes, (bytes, bytearray)):
        pil_full = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        pil_full = Image.open(image_bytes).convert("RGB")
    rgb01_full = np.asarray(pil_full).astype("float32") / 255.0
    img_h_full, img_w_full = rgb01_full.shape[:2]

    detections, used_fallback = face_detection.detect_faces(rgb01_full)
    check = face_detection.evaluate_face_check(detections, used_fallback)

    if not check.ok:
        return {"ok": False, "reason": check.reason, "message": check.message}

    crop_box = None
    if check.detection is not None:
        crop_box = face_detection.compute_crop_box(check.detection.bbox, img_w_full, img_h_full)

    # Map the face box (and landmarks, if we have them) into the model's
    # 224x224 input space, so region_analysis's masks line up with the
    # heatmaps pixel-for-pixel (Part B / Task 3: "so the regions and
    # heatmap share coordinates"). If no crop was applied, the "effective"
    # crop is just the whole original image being resized to 224x224.
    effective_box = crop_box if crop_box is not None else (0, 0, img_w_full, img_h_full)

    def _map(pt):
        return face_detection.map_point_through_crop(pt, effective_box, _img_w, _img_h)

    det = check.detection
    fx, fy, fw, fh = det.bbox
    face_box_224 = _map((fx, fy)) + _map((fx + fw, fy + fh))

    landmarks_224 = None
    if det.has_landmarks:
        landmarks_224 = {
            "right_eye": _map(det.right_eye),
            "left_eye": _map(det.left_eye),
            "nose": _map(det.nose),
            "right_mouth": _map(det.right_mouth),
            "left_mouth": _map(det.left_mouth),
        }

    return {
        "ok": True,
        "crop_box": crop_box,
        "tilt_warning": check.tilt_warning,
        "tilt_message": check.tilt_message,
        "used_fallback": check.used_fallback,
        "face_box_224": face_box_224,
        "landmarks_224": landmarks_224,
    }


def confidence_tier(conf):
    """Map a raw softmax confidence to a plain-language tier for the UI.

    Softmax outputs aren't calibrated probabilities of correctness, so we
    deliberately show a qualifier alongside the number rather than letting
    e.g. "97.8%" read as near-certainty on its own.
    """
    pct = conf * 100
    if pct >= 85:
        return "high"
    if pct >= 65:
        return "moderate"
    return "low"


def _norm01(m):
    m = np.asarray(m, dtype=np.float32)
    m = m - m.min()
    mx = m.max()
    return m / mx if mx > 1e-8 else m


# --------------------------------------------------------------------------
# Grad-CAM  (notebook section 7 — same math, un-double-wrapped inputs)
# --------------------------------------------------------------------------
def gradcam_heatmap(arr01, class_index):
    conv_layer = _model.get_layer(_last_conv_name)
    grad_model = tf.keras.models.Model(_model.inputs, [conv_layer.output, _model.output])

    x = tf.convert_to_tensor(arr01[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        while isinstance(conv_out, (list, tuple)):
            conv_out = conv_out[0]
        while isinstance(preds, (list, tuple)):
            preds = preds[0]
        if _binary_output:
            loss = preds[:, 0] if class_index == 1 else (1 - preds[:, 0])
        else:
            loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heat = tf.reduce_sum(conv_out * pooled, axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    heat = heat.numpy()
    heat = tf.image.resize(heat[..., None], (_img_h, _img_w)).numpy()[..., 0]
    return _norm01(heat)


# --------------------------------------------------------------------------
# Score-CAM  (notebook section 8 — identical algorithm)
# --------------------------------------------------------------------------
def score_cam(arr01, class_index, max_maps=32, batch_size=16):
    act_model = tf.keras.models.Model(_model.inputs, _model.get_layer(_last_conv_name).output)
    x = tf.convert_to_tensor(arr01[None, ...], dtype=tf.float32)
    acts = act_model(x)[0].numpy()
    _, _, c = acts.shape
    k = min(max_maps, c)

    maps = acts[..., :k]
    maps = (maps - maps.min(axis=(0, 1), keepdims=True)) / (
        maps.max(axis=(0, 1), keepdims=True) - maps.min(axis=(0, 1), keepdims=True) + 1e-8
    )
    maps_up = tf.image.resize(maps, (_img_h, _img_w)).numpy()

    masked = np.stack([arr01 * maps_up[..., j : j + 1] for j in range(k)], axis=0).astype(np.float32)

    scores = []
    for s in range(0, k, batch_size):
        batch = masked[s : s + batch_size]
        raw = _model.predict(batch, verbose=0)
        if _binary_output:
            col = raw[:, 0] if class_index == 1 else (1 - raw[:, 0])
        else:
            col = raw[:, class_index]
        scores.append(col)
    scores = np.concatenate(scores, axis=0)
    scores = np.maximum(scores, 0)

    heat = np.sum(maps_up * scores[None, None, :], axis=-1)
    return _norm01(heat)


# --------------------------------------------------------------------------
# LIME  (notebook section 9 — same green=support / red=contradict style,
# with the quickshift boundaries used in the paper-ready figures)
# --------------------------------------------------------------------------
def _lime_classifier_fn(images_uint8):
    imgs = images_uint8.astype(np.float32) / 255.0
    return _model.predict(imgs, verbose=0)


def run_lime_explanation(arr01, num_samples=None):
    """Runs LIME's (expensive) explain_instance once. Reused by both
    lime_vis() and lime_saliency_map() so "full" mode doesn't pay the
    1,500-sample cost twice for the same image."""
    if num_samples is None:
        num_samples = LIME_NUM_SAMPLES  # read at call time, not import time
    img_uint8 = (arr01 * 255).astype(np.uint8)
    return _lime_explainer.explain_instance(
        img_uint8,
        _lime_classifier_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: quickshift(x, **LIME_QUICKSHIFT),
    )


def lime_vis(display_base01, arr01, pred_id, explanation):
    """Green = superpixels supporting the predicted class; red =
    contradicting it, tinted onto `display_base01` (the image shown to
    the user — may have its eye band pixelated for sample-image privacy).
    `arr01` is the real, unaltered image: used only for recomputing the
    quickshift segmentation boundaries so they match what LIME actually
    explained — the LIME masks themselves come from `explanation`
    (already computed from the real image) regardless of which image we
    tint for display."""
    img_uint8 = (arr01 * 255).astype(np.uint8)

    _, mask_pos = explanation.get_image_and_mask(
        pred_id, positive_only=True, num_features=LIME_NUM_FEATURES, hide_rest=False
    )
    _, mask_neg = explanation.get_image_and_mask(
        pred_id, positive_only=False, negative_only=True, num_features=LIME_NUM_FEATURES, hide_rest=False
    )

    pos = (mask_pos > 0).astype(np.float32)
    neg = (mask_neg > 0).astype(np.float32)

    vis = display_base01.copy()
    vis[..., 1] = np.clip(vis[..., 1] + 0.60 * pos, 0, 1)
    vis[..., 0] = np.clip(vis[..., 0] + 0.60 * neg, 0, 1)

    seg = quickshift(img_uint8, **LIME_QUICKSHIFT)
    vis = mark_boundaries(vis, seg, color=(1, 1, 1), mode="thick")
    return np.clip(vis, 0, 1)


def lime_saliency_map(pred_id, explanation):
    """Raw positive-support saliency map for the overlap/agreement analysis
    (notebook's `lime_saliency`) — a continuous map rather than the
    thresholded pos/neg mask `lime_vis` draws with."""
    seg = explanation.segments
    local_exp = dict(explanation.local_exp[pred_id])
    w = np.zeros_like(seg, dtype=np.float32)
    for sp_id, weight in local_exp.items():
        w[seg == sp_id] = weight
    return _norm01(np.maximum(w, 0))


# --------------------------------------------------------------------------
# Deletion/Insertion faithfulness AUC
# (notebook section 10 — thesis Table 4.7: 31x31 average-pool blur
# baseline, 30-step deletion/insertion curves, trapezoidal AUC)
# --------------------------------------------------------------------------
def blur_baseline(arr01, k=31):
    """Simple box-blur baseline (average pooling), matching the notebook's
    `blur_baseline` exactly — this is what deleted/un-inserted pixels get
    replaced with, instead of e.g. plain black or grey."""
    x = tf.convert_to_tensor(arr01[None, ...], dtype=tf.float32)
    x = tf.nn.avg_pool2d(x, ksize=k, strides=1, padding="SAME")
    return x[0].numpy()


def deletion_insertion_curves(arr01, sal01, class_index, steps=30):
    """Deletion: start from the original image, progressively replace the
    most-salient pixels with the blurred baseline — the predicted
    probability should fall. Insertion: the reverse, starting from the
    baseline and progressively restoring the most-salient original
    pixels — the probability should rise. Returns (del_probs, ins_probs),
    each of length steps+1.

    This reproduces the notebook's `deletion_insertion_curves` exactly,
    but computes all steps as one batched model call per curve instead of
    `2 * (steps+1)` sequential single-image calls: each step's image only
    depends on which pixels fall within that step's top-k-salient set
    (not on any previous step's image), so the two are numerically
    identical — this is purely a speed optimisation.
    """
    h, w, _ = arr01.shape
    total = h * w
    sal_flat = sal01.ravel()
    order = np.argsort(-sal_flat)  # most salient first

    rank = np.empty(total, dtype=np.int64)
    rank[order] = np.arange(total)
    rank2d = rank.reshape(h, w)

    base = blur_baseline(arr01, k=31)
    orig = arr01

    k_values = [int((s / steps) * total) for s in range(steps + 1)]
    # mask[s, y, x] True where pixel (y, x) is among the top-k_s most salient
    masks = rank2d[None, ...] < np.array(k_values)[:, None, None]
    masks = masks[..., None]  # broadcast over channel dim

    del_imgs = np.where(masks, base[None, ...], orig[None, ...]).astype(np.float32)
    ins_imgs = np.where(masks, orig[None, ...], base[None, ...]).astype(np.float32)

    del_probs = _model.predict(del_imgs, verbose=0)[:, class_index]
    ins_probs = _model.predict(ins_imgs, verbose=0)[:, class_index]
    return del_probs.astype(np.float32), ins_probs.astype(np.float32)


def auc_trapz(y):
    """Trapezoidal AUC over a uniform [0, 1] x-grid, matching the
    notebook's `auc_trapz`. Uses `np.trapezoid` rather than the notebook's
    `np.trapz` — the same computation under NumPy's newer, non-deprecated
    name (`np.trapz` still works but warns under NumPy 2.x)."""
    x = np.linspace(0, 1, len(y), dtype=np.float32)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(y, x))


# --------------------------------------------------------------------------
# Visualisation helpers
# --------------------------------------------------------------------------
def overlay_heatmap(rgb01, heat01, alpha=0.45, cmap_name="jet"):
    heat01 = np.clip(heat01, 0, 1)
    cmap = plt.colormaps[cmap_name]
    heat_rgb = cmap(heat01)[..., :3].astype(np.float32)
    return np.clip((1 - alpha) * rgb01 + alpha * heat_rgb, 0, 1)


def array_to_data_uri(rgb01):
    img = Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def figure_to_data_uri(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def describe_region(heat01, face_box):
    """Turn the heatmap into a one-sentence, plain-language region description."""
    fx1, fy1, fx2, fy2 = face_box
    fh, fw = max(fy2 - fy1, 1), max(fx2 - fx1, 1)

    thresh = np.percentile(heat01, 90)
    ys, xs = np.where(heat01 >= thresh)
    if len(xs) == 0:
        return "The model's attention was too diffuse to localise to a specific facial region."

    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    rel_x = (cx - fx1) / fw
    rel_y = (cy - fy1) / fh

    vertical = (
        "the eye/upper-face region"
        if rel_y < 0.45
        else ("the nose/mid-face region" if rel_y < 0.7 else "the mouth/lower-face region")
    )
    if rel_x < 0.4:
        horizontal = "on the left side"
    elif rel_x > 0.6:
        horizontal = "on the right side"
    else:
        horizontal = "centred"

    return f"The model's attention was concentrated mainly on {vertical}, {horizontal} of the face."


# --------------------------------------------------------------------------
# 4-panel figure (notebook: Original + Grad-CAM + Score-CAM + LIME)
# --------------------------------------------------------------------------
def build_4panel_figure(rgb01, grad_overlay, scam_overlay, lime_img, header):
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.12)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))

    ax1.imshow(rgb01)
    ax1.set_title("Original", fontsize=13)
    ax2.imshow(grad_overlay)
    ax2.set_title("Grad-CAM", fontsize=13)
    ax3.imshow(scam_overlay)
    ax3.set_title("Score-CAM", fontsize=13)
    ax4.imshow(lime_img)
    ax4.set_title("LIME (Green=Support, Red=Contradict)", fontsize=13)
    for ax in (ax1, ax2, ax3, ax4):
        ax.axis("off")

    fig.suptitle(header, fontsize=13, y=0.98)
    return figure_to_data_uri(fig)


# --------------------------------------------------------------------------
# Cross-method overlap / agreement analysis
# (notebook's OVERLAP section: top-10% pixel IoU + Spearman correlation,
# pairwise Grad-vs-Score / Grad-vs-LIME / Score-vs-LIME, plus a 3-way figure)
# --------------------------------------------------------------------------
def spearman_corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


def topk_mask(map01, top_frac=TOPK_OVERLAP_FRACTION):
    flat = map01.ravel()
    k = max(1, int(top_frac * flat.size))
    idx = np.argpartition(-flat, k - 1)[:k]
    m = np.zeros_like(flat, dtype=np.uint8)
    m[idx] = 1
    return m.reshape(map01.shape)


def iou_mask(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-8))


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def overlay_two_on_image(base01, a01, b01, alpha=0.55):
    out = base01.copy()
    out[..., 0] = np.clip(out[..., 0] + alpha * a01, 0, 1)
    out[..., 1] = np.clip(out[..., 1] + alpha * b01, 0, 1)
    return out


def build_overlap_figure(base01, map_a, map_b, name_a, name_b, title_prefix, top_frac=TOPK_OVERLAP_FRACTION):
    mask_a = topk_mask(map_a, top_frac)
    mask_b = topk_mask(map_b, top_frac)
    inter = (mask_a & mask_b).astype(np.uint8)

    iou = iou_mask(mask_a, mask_b)
    sp = spearman_corr(map_a, map_b)

    over = overlay_two_on_image(base01, mask_a.astype(np.float32), mask_b.astype(np.float32))

    fig = plt.figure(figsize=(5.4, 5.7))
    plt.imshow(over)
    plt.axis("off")
    plt.title(
        f"{title_prefix}\nOverlap: {name_a} vs {name_b} | Top{int(top_frac*100)}% pixels\nIoU={iou:.3f} | Spearman={sp:.3f}",
        fontsize=10,
    )
    ax = plt.gca()
    for bb, color in [
        (bbox_from_mask(mask_a), "red"),
        (bbox_from_mask(mask_b), "lime"),
        (bbox_from_mask(inter), "blue"),
    ]:
        if bb:
            x1, y1, x2, y2 = bb
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.2, edgecolor=color)
            )
    handles = [
        plt.Line2D([0], [0], color="red", lw=3, label=name_a),
        plt.Line2D([0], [0], color="lime", lw=3, label=name_b),
        plt.Line2D([0], [0], color="blue", lw=3, label="Intersection"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    return figure_to_data_uri(fig, dpi=220), iou, sp


def build_3way_overlap_figure(base01, map_g, map_s, map_l, title_prefix, top_frac=TOPK_OVERLAP_FRACTION):
    m_g, m_s, m_l = topk_mask(map_g, top_frac), topk_mask(map_s, top_frac), topk_mask(map_l, top_frac)
    inter3 = (m_g & m_s & m_l).astype(np.uint8)

    iou_gs, iou_gl, iou_sl = iou_mask(m_g, m_s), iou_mask(m_g, m_l), iou_mask(m_s, m_l)

    three = base01.copy()
    three[..., 0] = np.clip(three[..., 0] + 0.55 * m_g.astype(np.float32), 0, 1)
    three[..., 1] = np.clip(three[..., 1] + 0.55 * m_s.astype(np.float32), 0, 1)
    three[..., 2] = np.clip(three[..., 2] + 0.55 * m_l.astype(np.float32), 0, 1)

    fig = plt.figure(figsize=(5.4, 5.7))
    plt.imshow(three)
    plt.axis("off")
    plt.title(
        f"{title_prefix}\n3-way Top{int(top_frac*100)}% overlap (RGB = Grad/Score/LIME)\n"
        f"IoU(G,S)={iou_gs:.3f} | IoU(G,L)={iou_gl:.3f} | IoU(S,L)={iou_sl:.3f}",
        fontsize=10,
    )
    ax = plt.gca()
    for bb, color in [
        (bbox_from_mask(m_g), "red"),
        (bbox_from_mask(m_s), "lime"),
        (bbox_from_mask(m_l), "blue"),
        (bbox_from_mask(inter3), "white"),
    ]:
        if bb:
            x1, y1, x2, y2 = bb
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor=color)
            )
    handles = [
        plt.Line2D([0], [0], color="red", lw=3, label="Grad-CAM ROI"),
        plt.Line2D([0], [0], color="lime", lw=3, label="Score-CAM ROI"),
        plt.Line2D([0], [0], color="blue", lw=3, label="LIME ROI"),
        plt.Line2D([0], [0], color="white", lw=3, label="3-way intersection"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    return figure_to_data_uri(fig, dpi=220), {
        "iou_grad_score": iou_gs,
        "iou_grad_lime": iou_gl,
        "iou_score_lime": iou_sl,
    }


_REGION_COLORS = {
    "forehead": "#E8A87C",
    "eyes": "#6FA3F2",
    "nose": "#8FE38F",
    "mouth": "#F27C7C",
    "rest_of_face": "#D8D86A",
    "face": "#8FE38F",
    "outside_face": "#9A9A9A",
}


def build_region_figure(overlay01, masks, percentages, face_box_224, title):
    """Heatmap overlay with region outlines drawn on it (left) + a bar
    chart of attention % per region (right). Region boundaries are drawn
    from the masks directly, so they always match what was actually
    measured. Never crops/masks the heatmap itself — only draws outlines
    on top of it, per Task 3's explicit instruction."""
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})

    ax_img.imshow(overlay01)
    ax_img.axis("off")
    fx1, fy1, fx2, fy2 = face_box_224
    ax_img.add_patch(
        plt.Rectangle((fx1, fy1), fx2 - fx1, fy2 - fy1, fill=False, edgecolor="white", linewidth=1.5)
    )

    region_order = [r for r in ("forehead", "eyes", "nose", "mouth") if r in masks]
    for name in region_order:
        bb = bbox_from_mask(masks[name].astype(np.uint8))
        if bb:
            x1, y1, x2, y2 = bb
            ax_img.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=_REGION_COLORS[name], linewidth=1.8
                )
            )
    ax_img.set_title("Regions (approximate)", fontsize=10)

    bar_names = [n for n in region_order + ["rest_of_face", "outside_face"] if n in percentages]
    bar_values = [percentages[n] for n in bar_names]
    bar_colors = [_REGION_COLORS[n] for n in bar_names]
    bar_labels = [n.replace("_", " ") for n in bar_names]

    ax_bar.barh(bar_labels, bar_values, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax_bar.set_xlabel("% of attention", fontsize=9)
    ax_bar.set_xlim(0, max(100, max(bar_values) * 1.1 if bar_values else 100))
    for i, v in enumerate(bar_values):
        ax_bar.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)
    ax_bar.invert_yaxis()

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    return figure_to_data_uri(fig, dpi=180)


# --------------------------------------------------------------------------
# Core explanation pipeline shared by both entry points
# --------------------------------------------------------------------------
def explain_image(
    image_bytes,
    mode="quick",
    true_label_display=None,
    crop_box=None,
    face_box_224=None,
    landmarks_224=None,
    blur_eyes=False,
):
    """mode: 'quick' | 'cross_check' | 'full'

    blur_eyes: Part C / Task 4 privacy. When True, every image returned
    for display (original, overlays, LIME, 4-panel, overlap, region
    figure) gets its eye band pixelated — but the model and every XAI
    computation still run on the real, unaltered image (`arr01`), never
    on the blurred version. /predict/sample always passes True (these
    are real children's photos); /predict/upload defaults to False
    (the user's own photo) unless they opt in.

    crop_box, if given, is a (x1, y1, x2, y2) box in the *original*
    image's pixel coordinates (from face_detection.compute_crop_box) —
    applied before the resize to the model's input size, so an uploaded
    photo where the face is small gets cropped to match the training
    images' tight-crop framing (Part B / Task 2). None (the default)
    preserves the exact prior behaviour: resize the whole image as-is.

    face_box_224 / landmarks_224, if given, are already in the model's
    224x224 input coordinate space (see run_face_check_on_bytes) and
    drive the Task 3 region-of-interest analysis (region masks, %
    attention per region, the off-face warning, and the region figure).
    Without them, this falls back to the original coarse, quadrant-based
    describe_region() heuristic — used by direct callers that never ran
    the Part B face check (tests, tools/reproduce_thesis_tables.py).
    """
    if crop_box is None:
        pil, arr01 = load_preprocess(image_bytes)
    else:
        if isinstance(image_bytes, (bytes, bytearray)):
            pil_full = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            pil_full = Image.open(image_bytes).convert("RGB")
        pil_cropped = pil_full.crop(crop_box)
        pil = pil_cropped.resize((_img_w, _img_h), resample=Image.NEAREST)
        arr01 = np.asarray(pil).astype("float32") / 255.0
    rgb01 = np.asarray(pil).astype("float32") / 255.0

    pred_id, conf, probs = predict(arr01)
    pred_label = ID_TO_LABEL[pred_id]

    face_box, face_found = detect_face_bbox(rgb01)

    # Part C / Task 4: pixelate the eye band in every DISPLAY image when
    # requested — never in anything fed to the model or an XAI method,
    # all of which use arr01/rgb01 (the real image) elsewhere in this
    # function. rgb01_display is used only in the visualisation calls
    # below (array_to_data_uri, overlay_heatmap, lime_vis's tint base,
    # and the 4-panel/overlap/region figures).
    rgb01_display = rgb01
    eyes_blurred = False
    if blur_eyes and face_box_224 is not None:
        eye_mask = privacy.build_eye_band_mask(landmarks_224, face_box_224, _img_h, _img_w)
        if eye_mask is not None:
            rgb01_display = privacy.pixelate(rgb01, eye_mask)
            eyes_blurred = True

    heat_grad = gradcam_heatmap(arr01, pred_id)
    overlay_grad = overlay_heatmap(rgb01_display, heat_grad)

    # Part B / Task 3: region-of-interest analysis, when we have a face
    # box from the Task 2 check. Falls back to the old coarse, quadrant-
    # based describe_region() when we don't (direct callers that skipped
    # the face check — tests, tools/reproduce_thesis_tables.py).
    region_masks = None
    region_attention = None
    off_face_msg = None
    if face_box_224 is not None:
        region_masks = region_analysis.build_region_masks(face_box_224, landmarks_224, _img_h, _img_w)
        grad_pct, grad_strongest = region_analysis.compute_attention_breakdown(heat_grad, region_masks)
        region_attention = {"gradcam": {"percentages": grad_pct, "strongest_region": grad_strongest}}
        off_face_msg = region_analysis.off_face_warning(grad_pct)
        region_summary = (
            region_analysis.describe_region_detailed(grad_pct, grad_strongest)
            if landmarks_224 is not None
            else region_analysis.describe_region_coarse(grad_pct)
        )
    else:
        region_summary = describe_region(heat_grad, face_box)

    result = {
        "prediction": LABEL_DISPLAY[pred_label],
        "confidence": round(conf * 100, 1),
        "confidence_tier": confidence_tier(conf),
        "prob_non_autistic": round(float(probs[0]) * 100, 1),
        "prob_autistic": round(float(probs[1]) * 100, 1),
        "original_image": array_to_data_uri(rgb01_display),
        "gradcam_image": array_to_data_uri(overlay_grad),
        "region_summary": region_summary,
        "off_face_warning": off_face_msg,
        "region_attention": region_attention,
        "face_detected": face_found,
        "eyes_blurred": eyes_blurred,
        "privacy_caption": privacy.EYE_BLUR_CAPTION if eyes_blurred else None,
        "method": "Grad-CAM",
    }

    if region_masks is not None:
        result["region_figure_image"] = build_region_figure(
            overlay_grad, region_masks, region_attention["gradcam"]["percentages"], face_box_224, "Grad-CAM"
        )

    if mode in ("cross_check", "full"):
        heat_score = score_cam(arr01, pred_id)
        overlay_score = overlay_heatmap(rgb01_display, heat_score)
        result["scorecam_image"] = array_to_data_uri(overlay_score)
        result["method"] = "Grad-CAM + Score-CAM"

        if region_masks is not None:
            score_pct, score_strongest = region_analysis.compute_attention_breakdown(heat_score, region_masks)
            result["region_attention"]["scorecam"] = {
                "percentages": score_pct,
                "strongest_region": score_strongest,
            }

    if mode == "full":
        lime_explanation = run_lime_explanation(arr01)
        lime_img = lime_vis(rgb01_display, arr01, pred_id, lime_explanation)
        # continuous map, for overlap metrics
        lime_map = lime_saliency_map(pred_id, lime_explanation)
        result["lime_image"] = array_to_data_uri(lime_img)
        result["method"] = "Grad-CAM + Score-CAM + LIME + Overlap Analysis"

        if region_masks is not None:
            lime_pct, lime_strongest = region_analysis.compute_attention_breakdown(lime_map, region_masks)
            result["region_attention"]["lime"] = {"percentages": lime_pct, "strongest_region": lime_strongest}

        header = (
            f"True: {true_label_display or 'Unknown'} | Pred: {LABEL_DISPLAY[pred_label]} | "
            f"Conf: {conf:.3f} | Probs: [Non_Autistic={probs[0]:.3f}, Autistic={probs[1]:.3f}]"
        )
        result["panel_4_image"] = build_4panel_figure(
            rgb01_display, overlay_grad, overlay_score, lime_img, header
        )

        g_del, g_ins = deletion_insertion_curves(arr01, heat_grad, pred_id, steps=30)
        s_del, s_ins = deletion_insertion_curves(arr01, heat_score, pred_id, steps=30)
        result["faithfulness"] = {
            "GradCAM_DeletionAUC": round(auc_trapz(g_del), 4),
            "GradCAM_InsertionAUC": round(auc_trapz(g_ins), 4),
            "ScoreCAM_DeletionAUC": round(auc_trapz(s_del), 4),
            "ScoreCAM_InsertionAUC": round(auc_trapz(s_ins), 4),
        }

        gs_img, iou_gs, sp_gs = build_overlap_figure(
            rgb01_display, heat_grad, heat_score, "Grad-CAM ROI", "Score-CAM ROI", header
        )
        gl_img, iou_gl, sp_gl = build_overlap_figure(
            rgb01_display, heat_grad, lime_map, "Grad-CAM ROI", "LIME ROI", header
        )
        sl_img, iou_sl, sp_sl = build_overlap_figure(
            rgb01_display, heat_score, lime_map, "Score-CAM ROI", "LIME ROI", header
        )
        three_img, three_ious = build_3way_overlap_figure(
            rgb01_display, heat_grad, heat_score, lime_map, header
        )

        result["overlap"] = {
            "grad_vs_score": {"image": gs_img, "iou": round(iou_gs, 3), "spearman": round(sp_gs, 3)},
            "grad_vs_lime": {"image": gl_img, "iou": round(iou_gl, 3), "spearman": round(sp_gl, 3)},
            "score_vs_lime": {"image": sl_img, "iou": round(iou_sl, 3), "spearman": round(sp_sl, 3)},
            "three_way": {"image": three_img, **{k: round(v, 3) for k, v in three_ious.items()}},
        }

    return result


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    n_samples = len(_list_test_images())
    return render_template("index.html", model_ready=model_ready(), n_samples=n_samples)


@app.context_processor
def _inject_wording():
    """Make the shared banner/disclaimer wording available to every template."""
    return {"research_banner": wording.RESEARCH_BANNER, "result_disclaimer": wording.RESULT_DISCLAIMER}


def _xai_settings():
    """(label, description) rows for the About page, read from the live
    constants and function defaults so the page can't drift from the code."""
    score_params = inspect.signature(score_cam).parameters
    faith_params = inspect.signature(deletion_insertion_curves).parameters
    blur_params = inspect.signature(blur_baseline).parameters
    qs = LIME_QUICKSHIFT
    blur_k = blur_params["k"].default
    return [
        (
            "Grad-CAM",
            f"Gradient-weighted class activation map taken from layer {_last_conv_name or 'the last convolutional layer'}.",
        ),
        (
            "Score-CAM",
            f"Gradient-free. Weights {score_params['max_maps'].default} feature maps by how much "
            "each one raises the predicted class score.",
        ),
        (
            "LIME",
            f"{LIME_NUM_SAMPLES:,} perturbed samples over quickshift superpixels (kernel size "
            f"{qs['kernel_size']}, max distance {qs['max_dist']}, ratio {qs['ratio']}). The top "
            f"{LIME_NUM_FEATURES} superpixels are shown: green supports the prediction, red contradicts it.",
        ),
        (
            "Agreement between methods",
            f"Overlap (IoU) of each method's top {TOPK_OVERLAP_FRACTION:.0%} of pixels, and Spearman rank correlation.",
        ),
        (
            "Faithfulness",
            f"Deletion and insertion curves over {faith_params['steps'].default} steps with a "
            f"{blur_k}\u00d7{blur_k} average-blur baseline; the area under each curve is reported.",
        ),
        (
            "Face check",
            f"Exactly one face is required (detected with YuNet). A photo is rejected if the detection score "
            f"is below {face_detection.SCORE_THRESHOLD} or the face is narrower than "
            f"{face_detection.MIN_FACE_WIDTH_PX} px; a warning is shown if the eyes are tilted by more than "
            f"{face_detection.EYE_TILT_WARN_DEGREES:.0f}\u00b0.",
        ),
        (
            "Cropping",
            f"If the face fills less than {face_detection.CROP_IF_FACE_FRACTION_BELOW:.0%} of the photo, it is "
            f"cropped around the face so the face fills about {face_detection.TARGET_FACE_FRACTION:.0%} of the "
            "crop, matching the framing of the test images.",
        ),
        (
            "Face regions",
            "Forehead, eyes, nose, mouth and rest-of-face regions are approximate, built from five facial "
            f"landmarks. A warning is shown when less than {region_analysis.ON_FACE_WARNING_THRESHOLD:.0%} of "
            "the model's attention falls on the face.",
        ),
    ]


_FACT_FIELDS = (
    "dataset_name",
    "dataset_source",
    "training_set_size",
    "class_balance",
    "data_split",
    "augmentation",
    "training_notes",
)


def build_about_context():
    """Everything the About page needs. Performance figures come only from
    the measured report (and are withheld if it was measured on a
    different model file) — see model_card.load_performance."""
    sha = model_card.file_sha256(MODEL_PATH)
    facts = model_card.load_facts(os.path.join(BASE_DIR, "model_card_facts.json"))
    return {
        "model": {
            "file": os.path.basename(MODEL_PATH),
            "size_mb": model_card.file_size_mb(MODEL_PATH),
            "sha256": sha,
            "input_size": f"{_img_w}\u00d7{_img_h}",
            "gradcam_layer": _last_conv_name,
        },
        "performance": model_card.load_performance(os.path.join(REPORTS_DIR, "lab_validation.json"), sha),
        "xai_rows": _xai_settings(),
        "facts": facts,
        "fact_rows": {key: model_card.fact_or_placeholder(facts[key]) for key in _FACT_FIELDS},
        "training_facts_filled": any(facts[key] not in (None, "", []) for key in _FACT_FIELDS),
        "placeholder": model_card.TO_BE_COMPLETED,
    }


@app.route("/about")
def about():
    """Model card: what the model is, how it was measured, how the
    explanations are produced, and its limitations."""
    return render_template("about.html", **build_about_context())


@app.route("/lab")
def lab_report():
    """Lab validation report (Task 5): the web app's own code run over the
    whole test set and compared with the thesis numbers. Generated
    offline by `python tools/run_lab_validation.py` — a 280-image batch
    is far too slow to run inside a web request, so this only serves the
    finished file."""
    report_path = os.path.join(REPORTS_DIR, "lab_validation.html")
    if not os.path.exists(report_path):
        return (
            "<!doctype html><title>Lab validation</title>"
            "<p>No lab validation report has been generated yet.</p>"
            "<p>Run <code>python tools/run_lab_validation.py</code> on the machine "
            "that has the model and test images, then reload this page.</p>",
            404,
        )
    with open(report_path, encoding="utf-8") as f:
        return f.read()


def _list_test_images():
    if not os.path.isdir(TEST_DIR):
        return []
    files = []
    for f in glob(os.path.join(TEST_DIR, "*")):
        if os.path.splitext(f)[1].lower() in ALLOWED_EXT:
            files.append(f)
    return sorted(files)


def _resolve_mode():
    mode = request.form.get("mode")
    if mode in ("quick", "cross_check", "full"):
        return mode
    # backward-compatible fallback for the old boolean toggle
    return "cross_check" if request.form.get("detailed") == "true" else "quick"


@app.route("/predict/upload", methods=["POST"])
def predict_upload():
    if not model_ready():
        return jsonify({"error": "Model is not loaded. See setup instructions on the home page."}), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image was uploaded."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use JPG, PNG, BMP or WEBP."}), 400

    image_bytes = file.read()

    face_check = run_face_check_on_bytes(image_bytes)
    if not face_check["ok"]:
        return (
            jsonify({"rejected": True, "reason": face_check["reason"], "message": face_check["message"]}),
            422,
        )

    mode = _resolve_mode()
    # Privacy (Task 4): uploads are the user's own photo, so leave them
    # unmasked by default — only blur if they explicitly opt in.
    blur_eyes = request.form.get("blur_eyes") == "true"
    try:
        result = explain_image(
            image_bytes,
            mode=mode,
            crop_box=face_check["crop_box"],
            face_box_224=face_check["face_box_224"],
            landmarks_224=face_check["landmarks_224"],
            blur_eyes=blur_eyes,
        )
    except Exception as exc:  # noqa: BLE001 - surface a readable error to the UI
        return jsonify({"error": f"Could not process this image: {exc}"}), 500

    result["source"] = "upload"
    result["true_label"] = None
    result["tilt_warning"] = face_check["tilt_warning"]
    result["tilt_message"] = face_check["tilt_message"]
    result["face_cropped"] = face_check["crop_box"] is not None
    result["face_detector_used_fallback"] = face_check["used_fallback"]
    return jsonify(result)


@app.route("/predict/sample", methods=["POST"])
def predict_sample():
    if not model_ready():
        return jsonify({"error": "Model is not loaded. See setup instructions on the home page."}), 503

    samples = _list_test_images()
    if not samples:
        return jsonify({"error": f"No test images found in {TEST_DIR}."}), 404

    mode = _resolve_mode()
    # reuse the exact same image for a follow-up analysis
    pinned = request.form.get("filename")

    if pinned:
        matches = [p for p in samples if os.path.basename(p) == pinned]
        path = matches[0] if matches else random.choice(samples)
    else:
        path = random.choice(samples)

    true_label = infer_true_label_from_filename(path)
    true_label_display = LABEL_DISPLAY[true_label] if true_label else None

    with open(path, "rb") as f:
        image_bytes = f.read()

    face_check = run_face_check_on_bytes(image_bytes)
    if not face_check["ok"]:
        return (
            jsonify(
                {
                    "rejected": True,
                    "reason": face_check["reason"],
                    "message": face_check["message"],
                    "source": "sample",
                    "filename": os.path.basename(path),
                }
            ),
            422,
        )

    try:
        result = explain_image(
            image_bytes,
            mode=mode,
            true_label_display=true_label_display,
            crop_box=face_check["crop_box"],
            face_box_224=face_check["face_box_224"],
            landmarks_224=face_check["landmarks_224"],
            # Task 4: mandatory for sample images (real dataset photos)
            blur_eyes=True,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not process this image: {exc}"}), 500

    result["source"] = "sample"
    result["filename"] = os.path.basename(path)
    result["true_label"] = true_label_display
    result["tilt_warning"] = face_check["tilt_warning"]
    result["tilt_message"] = face_check["tilt_message"]
    result["face_cropped"] = face_check["crop_box"] is not None
    result["face_detector_used_fallback"] = face_check["used_fallback"]
    return jsonify(result)


if __name__ == "__main__":
    ok = load_asd_model()
    if ok:
        print(
            f"Model loaded from {MODEL_PATH} | input size: {_img_w}x{_img_h} | last conv layer: {_last_conv_name}"
        )
    else:
        print(f"WARNING: no model found at {MODEL_PATH}. The app will run, but predictions are disabled")
        print("until you place your trained .h5 file there (or set ASD_MODEL_PATH).")
    n = len(_list_test_images())
    print(f"Test images found in {TEST_DIR}: {n}")
    app.run(host="0.0.0.0", port=5000, debug=False)
