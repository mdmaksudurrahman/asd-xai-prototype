# -*- coding: utf-8 -*-
"""
Part B / Task 3: region-of-interest analysis.

Builds approximate facial-region masks (eyes, nose, mouth, forehead,
rest-of-face, outside-face) from the face box and 5 landmarks that
face_detection.py already produces, then measures what fraction of a
Grad-CAM/Score-CAM/LIME heatmap's total "energy" falls in each region —
the same approach as the Score-CAM paper's "energy-based pointing game".

Two modes, depending on what face_detection gave us:

  - "detailed" (YuNet succeeded, landmarks available): the full 5-region
    breakdown inside the face, plus outside-face.
  - "coarse" (Haar fallback, no landmarks): just face vs. outside-face —
    still useful (the off-face warning works either way), but can't say
    *which part* of the face mattered.

Everything here is pure geometry/arithmetic on numpy arrays — no model,
no detector call — so it's fully unit-tested with hand-built face boxes
and landmarks (tests/test_region_analysis.py).

Regions are deliberately simple (horizontal bands + a central strip),
not a precise anatomical segmentation — the build instructions call
them "approximate", and with only 5 landmark points a more elaborate
shape would be false precision.
"""

import numpy as np

REGION_NAMES = ["forehead", "eyes", "nose", "mouth", "rest_of_face"]
# "less than 50% of attention on the face" -> warn
ON_FACE_WARNING_THRESHOLD = 0.50


def _rect_mask(img_h, img_w, x1, y1, x2, y2):
    mask = np.zeros((img_h, img_w), dtype=bool)
    x1c, x2c = int(round(max(0, x1))), int(round(min(img_w, x2)))
    y1c, y2c = int(round(max(0, y1))), int(round(min(img_h, y2)))
    if x2c > x1c and y2c > y1c:
        mask[y1c:y2c, x1c:x2c] = True
    return mask


def build_region_masks(face_box, landmarks, img_h=224, img_w=224):
    """
    face_box: (x1, y1, x2, y2) in the same coordinate space as the
        heatmaps (i.e. already mapped into the model's 224x224 input
        space — see face_detection.map_point_through_crop).
    landmarks: dict with keys 'right_eye', 'left_eye', 'nose',
        'right_mouth', 'left_mouth', each an (x, y) tuple — or None if
        unavailable (Haar fallback), in which case only a coarse
        face/outside-face split is returned.

    Returns a dict of boolean masks, each shaped (img_h, img_w):
        detailed mode: forehead, eyes, nose, mouth, rest_of_face, outside_face
        coarse mode:   face, outside_face
    """
    fx1, fy1, fx2, fy2 = face_box
    face_mask = _rect_mask(img_h, img_w, fx1, fy1, fx2, fy2)
    outside_mask = ~face_mask

    if landmarks is None:
        return {"face": face_mask, "outside_face": outside_mask}

    rex, rey = landmarks["right_eye"]
    lex, ley = landmarks["left_eye"]
    nx, ny = landmarks["nose"]
    rmx, rmy = landmarks["right_mouth"]
    lmx, lmy = landmarks["left_mouth"]

    eye_cy = (rey + ley) / 2.0
    eye_dist = max(1.0, ((rex - lex) ** 2 + (rey - ley) ** 2) ** 0.5)
    mouth_cx, mouth_cy = (rmx + lmx) / 2.0, (rmy + lmy) / 2.0

    eye_half = 0.25 * eye_dist
    eye_top, eye_bot = eye_cy - eye_half, eye_cy + eye_half

    mouth_half = 0.25 * eye_dist
    mouth_top = mouth_cy - mouth_half

    nose_half_w = 0.40 * eye_dist
    mouth_span = abs(rmx - lmx)
    mouth_half_w = max(0.35 * eye_dist, 0.55 * mouth_span)

    forehead = _rect_mask(img_h, img_w, fx1, fy1, fx2, eye_top)
    eyes = _rect_mask(img_h, img_w, fx1, eye_top, fx2, eye_bot)
    nose = _rect_mask(img_h, img_w, nx - nose_half_w, eye_bot, nx + nose_half_w, mouth_top)
    mouth = _rect_mask(img_h, img_w, mouth_cx - mouth_half_w, mouth_top, mouth_cx + mouth_half_w, fy2)

    # Everything else inside the face box
    named_union = forehead | eyes | nose | mouth
    rest_of_face = face_mask & ~named_union

    return {
        "forehead": forehead & face_mask,
        "eyes": eyes & face_mask,
        "nose": nose & face_mask,
        "mouth": mouth & face_mask,
        "rest_of_face": rest_of_face,
        "outside_face": outside_mask,
    }


def compute_attention_breakdown(heatmap01, masks):
    """Returns (percentages, strongest_region):
    percentages: dict region_name -> % of the heatmap's total energy
                 falling inside that region (sums to ~100)
    strongest_region: name of the region containing the single
                 highest-valued pixel
    """
    total = float(heatmap01.sum())
    percentages = {}
    if total <= 1e-8:
        # a degenerate, all-zero heatmap — split evenly rather than divide by zero
        n = len(masks)
        return {k: round(100.0 / n, 1) for k in masks}, next(iter(masks))

    for name, mask in masks.items():
        percentages[name] = round(100.0 * float(heatmap01[mask].sum()) / total, 1)

    peak_idx = np.unravel_index(np.argmax(heatmap01), heatmap01.shape)
    strongest_region = next((name for name, mask in masks.items() if mask[peak_idx]), None)

    return percentages, strongest_region


def on_face_percentage(percentages):
    """Total % of attention inside the face (sum of everything except
    'outside_face'), for both detailed and coarse mode."""
    return round(100.0 - percentages.get("outside_face", 0.0), 1)


def off_face_warning(percentages):
    """The Task 3 "off-face" warning, or None if attention is mostly on
    the face."""
    on_face = on_face_percentage(percentages)
    if on_face < ON_FACE_WARNING_THRESHOLD * 100:
        return (
            "Most of the model's attention is outside the face (hair, background or clothing). "
            "This result may reflect image artefacts rather than facial features — "
            "treat it as unreliable."
        )
    return None


def describe_region_detailed(percentages, strongest_region):
    """Matches the build instructions' example phrasing:
    "68% of the model's attention was on the face: eyes 31%, nose 18%,
    mouth 9%, rest of face 10%. 32% was outside the face."
    Region masks are approximate (built from 5 landmark points), so the
    wording says so explicitly.
    """
    on_face = on_face_percentage(percentages)
    outside = percentages.get("outside_face", 0.0)

    parts = [f"{percentages.get(r, 0.0):.0f}% {r.replace('_', ' ')}" for r in REGION_NAMES]
    detail = ", ".join(parts)

    sentence = (
        f"{on_face:.0f}% of the model's attention was on the face ({detail} — "
        f"regions are approximate). {outside:.0f}% was outside the face."
    )
    if strongest_region:
        label = strongest_region.replace("_", " ")
        sentence += f" The single strongest point was in the {label} region."
    return sentence


def describe_region_coarse(percentages):
    """Fallback wording when only a face/outside-face split is available
    (Haar fallback mode, no landmarks)."""
    on_face = on_face_percentage(percentages)
    outside = percentages.get("outside_face", 0.0)
    return (
        f"{on_face:.0f}% of the model's attention was on the (approximate) face region; "
        f"{outside:.0f}% was outside it. A more detailed region breakdown needs YuNet landmarks, "
        "which weren't available for this image."
    )
