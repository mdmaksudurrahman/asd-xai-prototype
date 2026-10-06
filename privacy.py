# -*- coding: utf-8 -*-
"""
Part C / Task 4: privacy for sample images.

The model and every XAI computation always run on the *original,
unaltered* image (Task 4's first rule — never give the model an image
with the eyes already blocked, since the thesis found the model relies
heavily on the eye/nose area, and blocking it would itself be a deletion
test). Only the images sent back to the browser for *display* get an
eye band pixelated, and only for sample images by default (real
children's photos from the dataset) — uploads are the user's own photo
and stay unmasked unless they explicitly opt in.

Pixelation (not a solid black bar) is used so the heatmap color blended
on top of it afterwards stays visible, per the build instructions.

Pure numpy here (no PIL/cv2) so both the geometry and the pixelation
itself are fully unit-tested without any image-library dependency.
"""

import numpy as np

EYE_BLUR_CAPTION = "Eyes blurred for privacy in the display only — the model analysed the original image."


def _rect_mask(img_h, img_w, x1, y1, x2, y2):
    mask = np.zeros((img_h, img_w), dtype=bool)
    x1c, x2c = int(round(max(0, x1))), int(round(min(img_w, x2)))
    y1c, y2c = int(round(max(0, y1))), int(round(min(img_h, y2)))
    if x2c > x1c and y2c > y1c:
        mask[y1c:y2c, x1c:x2c] = True
    return mask


def build_eye_band_mask(landmarks, face_box, img_h, img_w, height_factor=0.8):
    """A boolean mask covering the eye band, spanning the full face
    width. Uses the two eye landmarks if available (precise); falls
    back to the upper third of the face box if not (coarser, but still
    covers the eyes for a Haar-only detection with no landmarks).

    Returns None if face_box itself is None (nothing to blur).
    """
    if face_box is None:
        return None

    fx1, fy1, fx2, fy2 = face_box

    if landmarks is not None and "right_eye" in landmarks and "left_eye" in landmarks:
        rex, rey = landmarks["right_eye"]
        lex, ley = landmarks["left_eye"]
        eye_cy = (rey + ley) / 2.0
        eye_dist = max(1.0, ((rex - lex) ** 2 + (rey - ley) ** 2) ** 0.5)
        half = height_factor * eye_dist
        y1, y2 = eye_cy - half, eye_cy + half
    else:
        face_h = fy2 - fy1
        y1, y2 = fy1 + face_h * 0.15, fy1 + face_h * 0.45

    return _rect_mask(img_h, img_w, fx1, y1, fx2, y2)


def _resize_nearest(arr, out_h, out_w):
    """Nearest-neighbor resize on a raw numpy array — no PIL/cv2 needed,
    keeps this module dependency-free and simple to unit test."""
    h, w = arr.shape[:2]
    row_idx = np.clip((np.arange(out_h) * h / out_h).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(out_w) * w / out_w).astype(int), 0, w - 1)
    return arr[row_idx][:, col_idx]


def pixelate(rgb01, mask, block_size=10):
    """Return a copy of rgb01 with the masked region replaced by a
    blocky, heavily downsampled version of itself — classic pixelation,
    strong enough to be unidentifiable. Everything outside the mask is
    untouched. If the mask is empty (or None), returns rgb01 unchanged.
    """
    if mask is None or not np.any(mask):
        return rgb01.copy()

    out = rgb01.copy()
    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    region = out[y1:y2, x1:x2]
    h, w = region.shape[:2]
    small_h = max(1, h // block_size)
    small_w = max(1, w // block_size)

    small = _resize_nearest(region, small_h, small_w)
    pixelated_region = _resize_nearest(small, h, w)

    region_mask = mask[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = np.where(region_mask[..., None], pixelated_region, region)
    return out
