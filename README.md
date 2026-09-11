# ASD-XAI: Explainable AI for Facial-Image ASD Screening

![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)

A Flask web application that wraps a trained deep-learning classifier for
facial-image Autism Spectrum Disorder (ASD) screening with three tiers of
explainable AI (XAI): **Grad-CAM**, **Score-CAM**, and **LIME**, plus a
cross-method agreement analysis (IoU + Spearman correlation) between them.

This is a research prototype developed to accompany a thesis and grant
proposal on explainable AI for ASD detection. It is a **screening aid**,
not a diagnostic tool.

## Features

- **Two ways to get a result**: upload your own photo, or run on one of
  the held-out test images already on disk.
- **Three analysis tiers**, each building on the last, reusing the same
  image throughout:
  | Tier | Method(s) | Speed |
  |---|---|---|
  | Quick | Grad-CAM | Near-instant |
  | Cross-check | + Score-CAM | A few seconds |
  | Full report | + LIME, a 4-panel comparison figure, and pairwise/3-way cross-method overlap analysis | ~20s–a few minutes on CPU (LIME is the bottleneck) |
- **Plain-language explanations**: a face-region summary ("concentrated
  mainly on the eye/upper-face region"), confidence tiers instead of bare
  percentages, and a warning when no face is clearly detected in the
  photo.
- **No data leaves memory**: uploaded images are processed in RAM and
  never written to disk.
- **Architecture-agnostic**: automatically detects the model's input size
  and last convolutional layer, so it runs Xception, EfficientNet, or
  MobileNet checkpoints without code changes.

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask |
| Model inference | TensorFlow / Keras |
| Explainability | Grad-CAM & Score-CAM (custom), LIME, scikit-image (segmentation) |
| Face detection | OpenCV (Haar cascade) |
| Visualisation | Matplotlib |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework) |
| Testing | pytest |
| Linting / formatting | ruff, black |
| CI | GitHub Actions |

## Prerequisites

- Python 3.10+
- pip
- Your own trained model file (`Xception_best.h5` or equivalent) and test
  image set — **not included in this repository** (see [Data & privacy](#data--privacy))

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your model and test data

These are not tracked in git (see below). Place them at:

asd_xai_prototype/
├── models/
│ └── Xception_best.h5 <- your trained model
└── data/
└── test/
├── Autistic (1).jpg <- your held-out test images
├── Non_Autistic (1).jpg
└── ...


Ground-truth labels are inferred from filenames containing "autistic" /
"non_autistic" / "non-autistic" — for display only, they play no role in
prediction.

Alternatively, point at files elsewhere via environment variables instead
of moving them:

```bash
export ASD_MODEL_PATH=/path/to/Xception_best.h5
export ASD_TEST_DIR=/path/to/test_images
```

### 5. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000**. The terminal output confirms whether the
model loaded, its detected input size, and how many test images were
found.

## Project structure

.
├── app.py # Flask backend: model loading, Grad-CAM/
│ # Score-CAM/LIME, overlap analysis, routes
├── templates/
│ └── index.html
├── static/
│ ├── style.css
│ └── app.js
├── models/ # Your model goes here (git-ignored)
├── data/test/ # Your test images go here (git-ignored)
├── tests/ # pytest suite (uses a dummy model, no
│ # private data required)
├── .github/workflows/ci.yml # Lint + test on every push/PR
├── requirements.txt
├── requirements-dev.txt # pytest, ruff, black
└── pyproject.toml # ruff/black config


## Testing

The test suite runs against a tiny, randomly-initialised dummy model
(built on the fly in `tests/conftest.py`) — it needs none of your real
data to pass.

```bash
pip install -r requirements-dev.txt
pytest -v            # unit tests + full pipeline smoke tests + Flask routes
ruff check .          # lint
black --check .       # formatting
```

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `ASD_MODEL_PATH` | `models/Xception_best.h5` | Path to the trained model file |
| `ASD_TEST_DIR` | `data/test/` | Folder of held-out test images |

Other tunables live at the top of `app.py`:

| Constant | Default | Purpose |
|---|---|---|
| `LIME_NUM_SAMPLES` | `1500` | LIME perturbation count in "full" mode — lower for a faster, noisier map |
| `TOPK_OVERLAP_FRACTION` | `0.10` | Fraction of top pixels compared in the cross-method overlap analysis |

## Data & privacy

The model and test images are **children's facial photographs and a
model trained on them** — sensitive data that must never be committed.
`.gitignore` excludes `models/*` and `data/test/*` (only placeholder
files are tracked), and uploaded images are processed in memory only,
never written to disk. Keep any repository containing or configured to
use real data **private**.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Amber "No model loaded" banner on the home page | `models/Xception_best.h5` missing, or `ASD_MODEL_PATH` points somewhere empty |
| "Try a random test image" is disabled | `data/test/` has no `.jpg/.jpeg/.png/.bmp/.webp` files |
| "Full XAI report" takes a long time | Expected — LIME runs 1,500 forward passes per image on CPU |
| `ValueError: No Conv2D/SeparableConv2D layer found` | The loaded `.h5` file isn't a CNN classifier — confirm the path points to the actual trained model |
| `ModuleNotFoundError: No module named 'lime'` / `'skimage'` | Re-run `pip install -r requirements.txt` |

## License

Add your chosen license here (e.g. MIT) before making the repository
public, or state explicitly that all rights are reserved while private.

## Acknowledgments

Built to accompany a thesis and matching-grant proposal on explainable AI
for early ASD detection using facial image analysis.