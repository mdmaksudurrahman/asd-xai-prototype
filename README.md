# Facial-Image ASD Analysis — Explainable AI Research Prototype

> **Research prototype – not a diagnostic or screening tool. Do not use it to make decisions about any child.**

A small web application built around an already-trained deep-learning classifier for facial images (an Xception model). Given one face photo, it shows the model's prediction **and** how the model reached it, using three explanation methods (Grad-CAM, Score-CAM and LIME), a breakdown of which facial regions the model attended to, and checks of how well the methods agree with each other and how faithful they are.

It was developed as part of the research project *Explainable AI for Autism Spectrum Disorder Detection Using Deep Learning-Based Facial Image Analysis* (GERAN PENYELIDIKAN SEPADAN 2025, Universiti Pendidikan Sultan Idris).

The model is **not retrained** here. This repository is the explanation, checking and presentation layer around a model that was trained elsewhere.

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [How an image is processed](#how-an-image-is-processed)
- [Privacy and data handling](#privacy-and-data-handling)
- [Pages and API](#pages-and-api)
- [Configuration](#configuration)
- [Tools](#tools)
- [Validation](#validation)
- [Development](#development)
- [Project layout](#project-layout)
- [Status](#status)
- [Limitations](#limitations)
- [Ethics, data and licence](#ethics-data-and-licence)

## What it does

- **Checks the photo first.** Exactly one clear, large enough face is required. Photos with no face, several faces, a low detection score or a very small face are rejected before any analysis, with a plain-language reason.
- **Predicts** Autistic / Non-Autistic with a confidence tier (high, moderate, low). The confidence is a softmax output, not a calibrated probability.
- **Explains** the prediction with Grad-CAM, Score-CAM (gradient-free) and LIME, in three tiers so that a quick answer is always fast.
- **Describes where the model looked**, by facial region (forehead, eyes, nose, mouth, rest of face, outside the face) as percentages. The regions are approximate, built from five facial landmarks. A warning appears if less than half of the attention falls on the face.
- **Checks the explanations themselves**: how much the three methods agree (top-10% pixel overlap and rank correlation) and how faithful each map is (deletion and insertion curves).
- **Protects sample images**: the eye region of every sample image is pixelated in what is displayed. The model always analyses the original image.
- **Documents itself**: an About page (model card) and a lab-validation page that measure and report the app's own performance.

## Quick start

Tested with Python 3.12 on Windows. Versions are pinned to the research environment, so please use them as pinned.

### 1. Install

```powershell
git clone <your-repository-url>
cd asd_xai_prototype

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Add the three things that are deliberately not in this repository

| Put it here | What it is | Why it is not in git |
|---|---|---|
| `models/Xception_best.h5` | The trained classifier | Research asset |
| `models/face_detection_yunet_2023mar.onnx` | The YuNet face detector (about 230 KB, public, from the OpenCV Zoo) | Binary file; see the download note below |
| `data/test/` | The 280 test images (140 per class) | Photos of children: sensitive |

Test-image filenames must contain `Autistic` or `Non-Autistic` / `Non_Autistic`, because the ground-truth label is read from the filename (for example `Autistic (10).jpg`, `Non-Autistic (7).jpg`).

**Downloading the YuNet file.** Download `face_detection_yunet_2023mar.onnx` through a normal web browser from the OpenCV Zoo (`models/face_detection_yunet`) or from the Hugging Face mirror `opencv/face_detection_yunet`. Do not fetch it with `curl` from the raw GitHub URL: that file is stored with Git LFS, and a plain download can silently return a tiny text "pointer" instead of the real file. Then check it:

```powershell
python tools/verify_yunet_model.py
```

It should end with `PASS`. If YuNet is missing, the app still works but falls back to OpenCV's older Haar face detector, which gives no detection score, no eye-tilt check and only a coarse face / not-face attention split.

### 3. Run

```powershell
python app.py
```

Open <http://127.0.0.1:5000>.

If the model file is missing, the app still starts and shows a setup message instead of failing.

> **Security note.** `python app.py` listens on all network interfaces (`0.0.0.0`), so anyone on the same network can reach it. Only run it on a trusted network. For any shared or hosted use, put it behind a production server and a password. Sample images are real children's photos (shown with the eyes pixelated).

## Using the app

- **Try a sample.** The primary button picks a random image from `data/test/`. Sample images always have their eyes pixelated in the display.
- **Upload a photo.** Drag one onto the upload area or click it. Uploads are not pixelated unless you tick **Blur the eyes in the displayed images** before choosing the photo (the model still analyses the unblurred photo). Photos the face check refuses get a clear explanation instead of a result.
- **Analysis tiers.** Every analysis starts as *quick*. Follow-up buttons upgrade the same image:

| Tier | Methods | Typical time on CPU |
|---|---|---|
| Quick | Grad-CAM | seconds |
| Cross-check | Grad-CAM + Score-CAM | about 10–20 s |
| Full report | Grad-CAM + Score-CAM + LIME, 4-panel figure, method-agreement metrics, deletion / insertion AUC | from about 30 s to a few minutes |

## How an image is processed

1. **Face check** (full-size image, YuNet). Rejected if: no face; more than one face; detection score below 0.8; face narrower than 80 px. A warning, not a rejection, if the line between the eyes is tilted by more than 20°.
2. **Crop to match the test images.** If the face fills less than half of the photo, it is cropped around the face so it fills about 85% of the crop (calibrated on the 280 test images, which are already tightly cropped and so are normally not cropped at all).
3. **Prediction.** Resize to the model's input size (224 × 224) with nearest-neighbour resampling, divide pixel values by 255, classify. This matches how the thesis notebook loads images.
4. **Explanations**, all computed on the *original* image:
   - **Grad-CAM** from the model's last convolutional feature layer (`block14_sepconv2_act` for Xception).
   - **Score-CAM** using 32 feature maps.
   - **LIME** with 1,500 perturbed samples over quickshift superpixels (kernel size 4, max distance 100, ratio 0.2); the top 10 superpixels are drawn, green supporting the prediction and red contradicting it.
5. **Region analysis.** Forehead, eyes, nose, mouth, rest-of-face and outside-face masks are built from the five YuNet landmarks. The share of each heat map's total attention in each region is reported. If less than 50% falls on the face, an off-face warning is shown.
6. **Method checks (full tier).** Top-10% pixel IoU and Spearman correlation between each pair of methods and all three; deletion and insertion AUC (30 steps, 31 × 31 average-blur baseline) for Grad-CAM and Score-CAM.
7. **Display.** For sample images (and uploads that opt in), the eye band is pixelated in every image sent to the browser. Heat maps are drawn on top of the pixelated image so the attention over the eyes stays visible.

## Privacy and data handling

- The model and the explanation methods always run on the original, unaltered image. Only the displayed copies are pixelated, so blurring never changes a prediction.
- Uploaded photos are processed in memory and are not saved by this application.
- Model weights, test images and generated reports are excluded from version control (see `.gitignore`).
- Reports in `reports/` contain filenames and numbers only, never images.
- Do not host this publicly, or show the sample images, until you have confirmed that your ethics approval covers it.

## Pages and API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | The prototype |
| `/about` | GET | Model card: model identity (including the model file's SHA-256), measured performance, how explanations are produced, limitations |
| `/lab` | GET | The lab-validation report (generated offline; see [Validation](#validation)) |
| `/predict/upload` | POST | Analyse an uploaded image. Form fields: `image` (file), `mode` (`quick` / `cross_check` / `full`), optional `blur_eyes=true` |
| `/predict/sample` | POST | Analyse a random test image, or a specific one with `filename`. Form fields: `mode`, optional `filename`. Always eye-pixelated |

Responses are JSON.

- `200`: the result — `prediction`, `confidence`, `confidence_tier`, class probabilities, the images as data URIs, `region_summary`, `region_attention`, `off_face_warning`, `tilt_warning`, `face_cropped`, `eyes_blurred`, `privacy_caption`, and in the full tier `faithfulness` and `overlap`.
- `422`: the photo was rejected before analysis: `{"rejected": true, "reason": ..., "message": ...}`. `reason` is one of `no_face`, `multiple_faces`, `low_score`, `too_small`.
- `400`, `500`, `503`: `{"error": ...}` for a bad request, an internal error, or no model loaded.

## Configuration

All optional; defaults assume the layout above.

| Environment variable | Default | Meaning |
|---|---|---|
| `ASD_MODEL_PATH` | `models/Xception_best.h5` | The trained classifier |
| `ASD_YUNET_PATH` | `models/face_detection_yunet_2023mar.onnx` | The YuNet face detector |
| `ASD_TEST_DIR` | `data/test` | Folder of test / sample images |
| `ASD_REPORTS_DIR` | `reports` | Where validation reports are written and read |

The About page also reads `model_card_facts.json`: edit it to fill in the training details (dataset, size, split, augmentation). Anything left `null` is shown as "to be completed", and nothing is invented.

## Tools

Run from the repository root. Tools that load the model need the files from step 2 of the quick start.

| Script | Purpose |
|---|---|
| `tools/verify_yunet_model.py` | Check the YuNet file is real (not a Git-LFS pointer) and loads |
| `tools/calibrate_face_crop_margin.py` | Measure how much of each test image the face fills; recommends the crop setting |
| `tools/reproduce_thesis_tables.py` | Re-compute the XAI metrics for the 10 reference images and compare them with the published values |
| `tools/run_lab_validation.py` | Run all test images through the app and write the lab-validation report (`--limit N` for a quick check) |
| `tools/diagnose_validation_gap.py` | If the validation results differ from the published ones: look for the cause (image sizes, resize methods, threshold sweep, most confident errors) |
| `tools/compare_with_notebook_outputs.py` | Compare the app's per-image probabilities with those saved in the thesis XAI notebook (no model needed) |

Reference values used by these tools are in `tools/reference/`.

## Validation

`tools/run_lab_validation.py` measures the app on the held-out test set and writes `reports/lab_validation.html` (served at `/lab`), `.csv` and `.json`.

- **Classification metrics** (accuracy, precision, recall, AUC, confusion matrix) use the same preprocessing as the thesis notebook, on **every** test image, with no face-check filtering. Positive class is *Autistic*.
- **Attention-on-face statistics** use the app's own pipeline, so they only cover images in which a face was found; the report lists the ones that were left out.
- The report records the SHA-256 of the model file it measured. The About page reads the `.json` and **withholds the figures if the model file has changed since**, so stale numbers are never shown.
- A partial run (for example `--limit 20`) is labelled as such.

Run it again whenever the model file or the test set changes.

**Known open item.** The lab validation compares the app's measured confusion matrix with the one published in the thesis, and these are not yet reconciled. The cause is being investigated (see `tools/diagnose_validation_gap.py` and `tools/compare_with_notebook_outputs.py`). The `/lab` page reports the difference as measured.

## Development

```powershell
pytest                # tests do not need the real model or any real image
ruff check .          # lint
black --check .       # formatting (line length 110)
```

The tests use a tiny randomly initialised network and synthetic images in place of the real model and photos, so they run anywhere. Face detection is replaced by hand-built detections in most tests.

GitHub Actions (`.github/workflows/ci.yml`) runs lint and the tests on every push and pull request to `main`.

**Windows line endings.** If `black --check .` complains after pasting code, run `black .` once. `.gitattributes` enforces LF endings in git; after the first reformat run `git add --renormalize .` so your commit does not show every line as changed.

## Project layout

```
app.py                     Flask app and the whole analysis pipeline
face_detection.py          YuNet / Haar face detection, face-check rules, crop geometry
region_analysis.py         Facial-region masks and attention percentages
privacy.py                 Eye-band pixelation for displayed images
lab_validation.py          Metrics, thesis comparison, and the validation report
model_card.py              About-page logic: model hash, measured figures, editable facts
wording.py                 The research banner and disclaimer text (single source)
model_card_facts.json      Editable training details for the About page
templates/                 index.html, about.html, _banner.html
static/                    style.css, app.js, banner.css, about.css
tools/                     Command-line scripts (see Tools) and tools/reference/
tests/                     Test suite
requirements.txt           Pinned runtime dependencies
requirements-dev.txt       pytest, ruff, black
pyproject.toml             ruff and black settings
models/                    NOT in git: classifier and face detector
data/test/                 NOT in git: the 280 test images
reports/                   NOT in git: generated validation reports
```

## Status

**Working**

- Thesis-consistent explanations (Grad-CAM layer, resize method, deletion / insertion AUC) with a reproduction script.
- Face check, crop and region analysis, with the eye-band privacy display.
- Lab-validation report, the About page, and the research banner on every page.

- A front end that shows the face-check rejections, privacy caption, off-face and tilt warnings, region figure, faithfulness table and an opt-in eye-blur switch.

**Planned**

- A curated set of 10 demo images with pre-computed reports.
- Hosting: a production server, a background job with a progress bar for the full tier, metadata-only logging, and password protection.
- Dataset-artefact checks (near-duplicate images between training and test, background-only and face-only tests).

## Limitations

- Performance was measured on one held-out test set. How the model behaves on other populations, ages, cameras or lighting has not been tested.
- The confidence value is not a calibrated probability of being correct.
- Explanation maps show where the model looked, not why a condition is or is not present. Facial regions are approximate.
- The test images are tightly cropped faces, so the face box covers most of the picture and "attention on the face" is high almost by construction. It is weak evidence about whether the model uses the background; the planned background-only and face-only tests are the stronger check.
- Photos without exactly one clear, large enough face are rejected.

## Ethics, data and licence

- The model weights and test images are intentionally not committed. Do not add them, or any photos of children, to this repository.
- Confirm that your ethics approval covers any hosted demo, and protect it with a password, before sharing a link.
- Licence: not yet specified. Add one before sharing the repository beyond the research team.