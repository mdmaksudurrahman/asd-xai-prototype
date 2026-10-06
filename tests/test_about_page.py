"""
Route-level tests for Task 6: the research banner on every page, the
wording rules, and the About / model-card page.

The rules these protect:
  - the banner is on every page and prints;
  - nothing describes the prototype as a screening aid;
  - the About page shows only what the app measured — no thesis figures,
    no comparison, even when the report on disk contains them;
  - stale figures (measured on a different model file) are never shown.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from markupsafe import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as app_module  # noqa: E402
import lab_validation as lv  # noqa: E402
import model_card  # noqa: E402
import wording  # noqa: E402

THESIS_FIGURES = ("97.5", "98.54", "96.43", "99.14")


def _model_sha():
    return hashlib.sha256(Path(app_module.MODEL_PATH).read_bytes()).hexdigest()


def _write_report(tmp_path, sha, n=280):
    report = {
        "generated_at": "2026-10-06T17:41:00",
        "model": {"file": "model.h5", "sha256": sha, "input_size": "96x96"},
        "metrics": {
            "n": n,
            "accuracy": 0.9607,
            "precision": 0.9510,
            "recall": 0.9714,
            "auc": 0.9916,
            "confusion_matrix": [[133, 7], [4, 136]],
        },
        # what tools/run_lab_validation.py really writes — must never reach the page
        "comparison": {
            "metrics": [{"metric": "accuracy", "ours": 0.9607, "thesis": 0.975, "status": "DIFF"}],
            "confusion": {"ours": [[133, 7], [4, 136]], "thesis": [[138, 2], [5, 135]], "status": "DIFF"},
            "comparable": True,
            "all_match": False,
        },
    }
    (tmp_path / "lab_validation.json").write_text(json.dumps(report), encoding="utf-8")


def _lab_report():
    """Minimal report dict for lab_validation.render_html()."""
    metrics = lv.compute_metrics([0, 1, 0, 1], [0, 1, 1, 1], [0.1, 0.9, 0.7, 0.8])
    return {
        "generated_at": "2026-10-06T12:00:00",
        "model": {"file": "m.h5", "sha256": "abc", "input_size": "96x96"},
        "metrics": metrics,
        "comparison": lv.compare_to_thesis(metrics),
        "attention": lv.summarize_attention([]),
    }


def _about(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "REPORTS_DIR", str(tmp_path))
    resp = client.get("/about")
    assert resp.status_code == 200
    return resp.get_data(as_text=True)


# --------------------------------------------------------------------------
# wording
# --------------------------------------------------------------------------
def test_banner_text_says_what_it_must():
    banner = wording.RESEARCH_BANNER
    assert "Research prototype" in banner
    assert "not a diagnostic or screening tool" in banner
    assert "any child" in banner


# --------------------------------------------------------------------------
# banner + wording on every page
# --------------------------------------------------------------------------
def test_banner_on_the_main_page(client):
    page = client.get("/").get_data(as_text=True)
    assert str(escape(wording.RESEARCH_BANNER)) in page
    assert "banner.css" in page


def test_banner_on_the_about_page(client, monkeypatch, tmp_path):
    page = _about(client, monkeypatch, tmp_path)
    assert str(escape(wording.RESEARCH_BANNER)) in page


def test_banner_on_the_lab_report():
    assert wording.RESEARCH_BANNER in lv.render_html(_lab_report())


def test_banner_is_not_hidden_when_printing(client):
    page = client.get("/").get_data(as_text=True)
    opening_tag = re.search(r'<div class="research-banner"[^>]*>', page).group(0)
    assert "no-print" not in opening_tag
    assert "display:none" not in opening_tag.replace(" ", "")


def test_about_link_shown_on_main_page_but_not_on_about_itself(client, monkeypatch, tmp_path):
    assert "research-banner__link" in client.get("/").get_data(as_text=True)
    assert "research-banner__link" not in _about(client, monkeypatch, tmp_path)


def test_no_page_calls_the_prototype_a_screening_aid(client, monkeypatch, tmp_path):
    pages = {
        "main": client.get("/").get_data(as_text=True),
        "about": _about(client, monkeypatch, tmp_path),
        "lab": lv.render_html(_lab_report()),
    }
    for name, page in pages.items():
        for phrase in wording.FORBIDDEN_PHRASES:
            assert phrase.lower() not in page.lower(), f"{phrase!r} found on the {name} page"


def test_result_disclaimer_comes_from_the_shared_wording(client):
    page = client.get("/").get_data(as_text=True)
    assert str(escape(wording.RESULT_DISCLAIMER)) in page
    assert "not a diagnosis and not a screening tool" in page


def test_main_page_heading_no_longer_says_screening(client):
    page = client.get("/").get_data(as_text=True)
    assert "<h1>Facial-Image ASD Analysis</h1>" in page


# --------------------------------------------------------------------------
# About page: the model
# --------------------------------------------------------------------------
def test_about_shows_model_identity(client, monkeypatch, tmp_path):
    page = _about(client, monkeypatch, tmp_path)
    assert _model_sha() in page
    assert Path(app_module.MODEL_PATH).name in page
    assert "96\u00d796" in page  # the dummy model's input size
    assert "last_conv" in page  # the Grad-CAM layer actually in use


def test_about_works_without_a_model_file(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "_last_conv_name", None)
    monkeypatch.setattr(app_module, "MODEL_PATH", "/definitely/not/here/model.h5")
    monkeypatch.setattr(app_module, "REPORTS_DIR", str(tmp_path))
    page = app_module.app.test_client().get("/about").get_data(as_text=True)
    assert "Model file not found on this server" in page
    assert "Available once the model is loaded" in page


def test_about_xai_settings_are_read_from_the_live_code(client, monkeypatch, tmp_path):
    page = _about(client, monkeypatch, tmp_path)
    assert f"{app_module.LIME_NUM_SAMPLES:,}" in page
    assert str(app_module.face_detection.SCORE_THRESHOLD) in page
    assert f"{app_module.face_detection.MIN_FACE_WIDTH_PX} px" in page


def test_lime_settings_are_unchanged_by_the_refactor_into_constants():
    """These feed the thesis-consistent LIME maps. Moving them into named
    constants must not have changed a single value."""
    assert app_module.LIME_QUICKSHIFT == {"kernel_size": 4, "max_dist": 100, "ratio": 0.2}
    assert app_module.LIME_NUM_FEATURES == 10
    assert app_module.LIME_NUM_SAMPLES == 1500


def test_about_unfilled_facts_show_one_placeholder_not_an_invention(client, monkeypatch, tmp_path):
    page = _about(client, monkeypatch, tmp_path)
    assert model_card.TO_BE_COMPLETED in page
    # one notice, not a wall of placeholders
    assert page.count(model_card.TO_BE_COMPLETED) == 1
    assert "Training set size" not in page  # no empty rows pretending to be data


def test_about_partly_filled_facts_list_every_row_with_placeholders_for_the_gaps(
    client, monkeypatch, tmp_path
):
    (tmp_path / "model_card_facts.json").write_text(
        json.dumps({"training_set_size": "2,540 images"}), encoding="utf-8"
    )
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))
    page = _about(client, monkeypatch, tmp_path)
    assert "2,540 images" in page
    assert "Augmentation" in page
    assert model_card.TO_BE_COMPLETED in page


def test_about_filled_in_facts_are_shown(client, monkeypatch, tmp_path):
    (tmp_path / "model_card_facts.json").write_text(
        json.dumps({"dataset_name": "Example facial-image dataset"}), encoding="utf-8"
    )
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))
    page = _about(client, monkeypatch, tmp_path)
    assert "Example facial-image dataset" in page


# --------------------------------------------------------------------------
# About page: measured performance
# --------------------------------------------------------------------------
def test_about_without_a_report_says_figures_are_not_generated(client, monkeypatch, tmp_path):
    page = _about(client, monkeypatch, tmp_path)
    assert "have not been generated yet" in page


def test_about_shows_only_what_the_app_measured(client, monkeypatch, tmp_path):
    _write_report(tmp_path, _model_sha())
    page = _about(client, monkeypatch, tmp_path)
    for measured in ("96.07%", "95.10%", "97.14%", "99.16%"):
        assert measured in page
    assert "133" in page and "136" in page  # the app's confusion matrix


def test_about_never_shows_thesis_figures_or_a_comparison(client, monkeypatch, tmp_path):
    """The report on disk contains the thesis comparison block. The About
    page must show none of it."""
    _write_report(tmp_path, _model_sha())
    page = _about(client, monkeypatch, tmp_path)
    # The page legitimately shows the model's 64-character SHA-256, which can
    # contain digit runs like "138" by pure chance — take it out before
    # looking for thesis figures, so this test can't fail on an unlucky hash.
    page = page.replace(_model_sha(), "")
    for figure in THESIS_FIGURES:
        assert figure not in page, figure
    for token in ("MATCH", "DIFF", "thesis", "Thesis", "138"):
        assert token not in page, token


def test_an_unlucky_model_hash_does_not_look_like_a_thesis_figure(client, monkeypatch, tmp_path):
    """Regression: a SHA-256 that happens to contain '138' must not make
    the no-thesis-figures check fail."""
    unlucky = "9138" + "a" * 60
    monkeypatch.setattr(model_card, "file_sha256", lambda path: unlucky)
    _write_report(tmp_path, unlucky)
    page = _about(client, monkeypatch, tmp_path)
    assert unlucky in page  # the hash really is shown
    page_without_hash = page.replace(unlucky, "")
    assert "138" not in page_without_hash
    assert "thesis" not in page_without_hash.lower()


def test_about_withholds_figures_measured_on_a_different_model_file(client, monkeypatch, tmp_path):
    _write_report(tmp_path, "f" * 64)
    page = _about(client, monkeypatch, tmp_path)
    assert "different version of the model file" in page
    assert "96.07%" not in page


def test_about_flags_a_partial_run(client, monkeypatch, tmp_path):
    _write_report(tmp_path, _model_sha(), n=20)
    page = _about(client, monkeypatch, tmp_path)
    assert "partial run of 20 images" in page
    assert "96.07%" in page  # shown, but flagged
