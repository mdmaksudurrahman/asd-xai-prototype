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
model to find its input size and its last convolutional layer, so the
same code works for the Xception/EfficientNet/MobileNet variants
mentioned in the project proposal.

A note on one deliberate difference from the notebook: the notebook
builds the Grad-CAM helper model with `tf.keras.models.Model([model.inputs], ...)`
— inputs wrapped in an extra list. On some TensorFlow/Keras versions that
double-wrapping is tolerated; on others it corrupts the model's returned
outputs and crashes. This file uses `tf.keras.models.Model(model.inputs, ...)`
(no extra wrapping) instead — the same computation, just without that
version-dependent landmine. Everything else (the Grad-CAM/Score-CAM math,
the LIME settings, the 4-panel figure, the overlap/IoU/Spearman analysis)
mirrors the notebook's final pipeline exactly.
"""

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
import io
import os
import random
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
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ID_TO_LABEL = {0: "Non_Autistic", 1: "Autistic"}
LABEL_DISPLAY = {"Non_Autistic": "Non-Autistic", "Autistic": "Autistic"}

# LIME sample count — matches the notebook's `lime_vis(..., num_samples=1500)`
# exactly. This is the slow part of "full" mode; lower it (e.g. 500) in your
# own copy if you want a faster, slightly noisier LIME map.
LIME_NUM_SAMPLES = 1500
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
    for layer in reversed(m.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            return layer.name
    raise ValueError("No Conv2D/SeparableConv2D layer found in the model.")


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


# --------------------------------------------------------------------------
# Pre/post-processing helpers
# --------------------------------------------------------------------------
def load_preprocess(path_or_bytes):
    """Load an image (path or raw bytes) -> (PIL RGB image, float32 array in [0,1])."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    else:
        pil = Image.open(path_or_bytes).convert("RGB")
    pil_resized = pil.resize((_img_w, _img_h))
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
        segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=100, ratio=0.2),
    )


def lime_vis(arr01, pred_id, explanation):
    """Green = superpixels supporting the predicted class; red = contradicting it."""
    img_uint8 = (arr01 * 255).astype(np.uint8)

    _, mask_pos = explanation.get_image_and_mask(
        pred_id, positive_only=True, num_features=10, hide_rest=False
    )
    _, mask_neg = explanation.get_image_and_mask(
        pred_id, positive_only=False, negative_only=True, num_features=10, hide_rest=False
    )

    pos = (mask_pos > 0).astype(np.float32)
    neg = (mask_neg > 0).astype(np.float32)

    vis = arr01.copy()
    vis[..., 1] = np.clip(vis[..., 1] + 0.60 * pos, 0, 1)
    vis[..., 0] = np.clip(vis[..., 0] + 0.60 * neg, 0, 1)

    seg = quickshift(img_uint8, kernel_size=4, max_dist=100, ratio=0.2)
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


# --------------------------------------------------------------------------
# Core explanation pipeline shared by both entry points
# --------------------------------------------------------------------------
def explain_image(image_bytes, mode="quick", true_label_display=None):
    """mode: 'quick' | 'cross_check' | 'full'"""
    pil, arr01 = load_preprocess(image_bytes)
    rgb01 = np.asarray(pil).astype("float32") / 255.0

    pred_id, conf, probs = predict(arr01)
    pred_label = ID_TO_LABEL[pred_id]

    face_box, face_found = detect_face_bbox(rgb01)

    heat_grad = gradcam_heatmap(arr01, pred_id)
    overlay_grad = overlay_heatmap(rgb01, heat_grad)

    result = {
        "prediction": LABEL_DISPLAY[pred_label],
        "confidence": round(conf * 100, 1),
        "confidence_tier": confidence_tier(conf),
        "prob_non_autistic": round(float(probs[0]) * 100, 1),
        "prob_autistic": round(float(probs[1]) * 100, 1),
        "original_image": array_to_data_uri(rgb01),
        "gradcam_image": array_to_data_uri(overlay_grad),
        "region_summary": describe_region(heat_grad, face_box),
        "face_detected": face_found,
        "method": "Grad-CAM",
    }

    if mode in ("cross_check", "full"):
        heat_score = score_cam(arr01, pred_id)
        overlay_score = overlay_heatmap(rgb01, heat_score)
        result["scorecam_image"] = array_to_data_uri(overlay_score)
        result["method"] = "Grad-CAM + Score-CAM"

    if mode == "full":
        lime_explanation = run_lime_explanation(arr01)
        lime_img = lime_vis(arr01, pred_id, lime_explanation)
        # continuous map, for overlap metrics
        lime_map = lime_saliency_map(pred_id, lime_explanation)
        result["lime_image"] = array_to_data_uri(lime_img)
        result["method"] = "Grad-CAM + Score-CAM + LIME + Overlap Analysis"

        header = (
            f"True: {true_label_display or 'Unknown'} | Pred: {LABEL_DISPLAY[pred_label]} | "
            f"Conf: {conf:.3f} | Probs: [Non_Autistic={probs[0]:.3f}, Autistic={probs[1]:.3f}]"
        )
        result["panel_4_image"] = build_4panel_figure(rgb01, overlay_grad, overlay_score, lime_img, header)

        gs_img, iou_gs, sp_gs = build_overlap_figure(
            rgb01, heat_grad, heat_score, "Grad-CAM ROI", "Score-CAM ROI", header
        )
        gl_img, iou_gl, sp_gl = build_overlap_figure(
            rgb01, heat_grad, lime_map, "Grad-CAM ROI", "LIME ROI", header
        )
        sl_img, iou_sl, sp_sl = build_overlap_figure(
            rgb01, heat_score, lime_map, "Score-CAM ROI", "LIME ROI", header
        )
        three_img, three_ious = build_3way_overlap_figure(rgb01, heat_grad, heat_score, lime_map, header)

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

    mode = _resolve_mode()
    try:
        result = explain_image(file.read(), mode=mode)
    except Exception as exc:  # noqa: BLE001 - surface a readable error to the UI
        return jsonify({"error": f"Could not process this image: {exc}"}), 500

    result["source"] = "upload"
    result["true_label"] = None
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

    try:
        with open(path, "rb") as f:
            result = explain_image(f.read(), mode=mode, true_label_display=true_label_display)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not process this image: {exc}"}), 500

    result["source"] = "sample"
    result["filename"] = os.path.basename(path)
    result["true_label"] = true_label_display
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
