"""
Guards the contract between the page (index.html + app.js) and the server.

There is no browser in the test suite, so these checks catch the classic
front-end breakages statically: a script that looks up an element the
template doesn't have, reads a result field the server never sends, loads
a stylesheet that doesn't exist, or starts building HTML from server text.
"""

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app as app_module  # noqa: E402

APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

# Fields the routes add around explain_image()'s result, and the shapes of
# a rejection (422) and an error reply.
ROUTE_FIELDS = {
    "source",
    "filename",
    "true_label",
    "tilt_warning",
    "tilt_message",
    "face_cropped",
    "face_detector_used_fallback",
}
REJECTION_AND_ERROR_FIELDS = {"rejected", "reason", "message", "error"}


def _ids_in_template():
    return set(re.findall(r'\bid="([^"]+)"', INDEX_HTML))


def test_every_element_the_script_looks_up_exists_in_the_template():
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', APP_JS))
    missing = sorted(wanted - _ids_in_template())
    assert wanted, "no getElementById calls found - has the script changed shape?"
    assert missing == [], f"app.js looks up ids that index.html doesn't have: {missing}"


def test_template_has_no_duplicate_ids():
    ids = re.findall(r'\bid="([^"]+)"', INDEX_HTML)
    assert sorted({i for i in ids if ids.count(i) > 1}) == []


def test_every_result_field_the_script_reads_is_one_the_server_sends(loaded_app, monkeypatch):
    monkeypatch.setattr(app_module, "LIME_NUM_SAMPLES", 20)  # keep the full tier fast
    rng = np.random.default_rng(0)
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((rng.random((120, 120, 3)) * 255).astype("uint8")).save(buf, format="PNG")
    result = app_module.explain_image(
        buf.getvalue(),
        mode="full",
        face_box_224=(10.0, 10.0, 86.0, 86.0),
        landmarks_224={
            "right_eye": (30.0, 35.0),
            "left_eye": (66.0, 35.0),
            "nose": (48.0, 52.0),
            "right_mouth": (36.0, 68.0),
            "left_mouth": (60.0, 68.0),
        },
        blur_eyes=True,
    )
    server_fields = set(result) | ROUTE_FIELDS | REJECTION_AND_ERROR_FIELDS

    used = set(re.findall(r"\bdata\.([A-Za-z_]\w*)", APP_JS))
    unknown = sorted(used - server_fields)
    assert used, "no data.<field> reads found - has the script changed shape?"
    assert unknown == [], f"app.js reads fields the server never sends: {unknown}"

    # the four faithfulness numbers it displays must exist under those exact names
    for key in re.findall(r"f\.((?:GradCAM|ScoreCAM)_\w+AUC)", APP_JS):
        assert key in result["faithfulness"], key


def test_every_new_result_field_is_actually_displayed():
    """The reverse check: the fields the privacy and region work depends on
    must be read by the page, or the work is invisible to the user."""
    used = set(re.findall(r"\bdata\.([A-Za-z_]\w*)", APP_JS))
    must_be_shown = {
        "privacy_caption",
        "eyes_blurred",
        "off_face_warning",
        "region_figure_image",
        "tilt_warning",
        "tilt_message",
        "face_cropped",
        "face_detector_used_fallback",
        "faithfulness",
        "region_summary",
        "rejected",
    }
    assert sorted(must_be_shown - used) == []


def test_stylesheets_and_scripts_the_page_loads_exist():
    for name in re.findall(r"filename='([^']+)'", INDEX_HTML):
        assert (ROOT / "static" / name).exists(), f"index.html loads static/{name}, which doesn't exist"


def test_script_never_builds_html_from_server_text():
    """innerHTML is only allowed for two things whose content is fixed in
    the script itself. Anything the server sends goes through textContent,
    .src or createElement, so a hostile message can't inject markup."""
    allowed = ("loadingSteps.innerHTML = labels.map", 'historyStrip.innerHTML = ""')
    offenders = [
        line.strip()
        for line in APP_JS.splitlines()
        if "innerHTML" in line and not any(a in line for a in allowed)
    ]
    assert offenders == [], f"unexpected innerHTML use: {offenders}"


def test_rendered_page_includes_the_new_blocks(client):
    page = client.get("/").get_data(as_text=True)
    for element_id in (
        "rejectedState",
        "privacyCaption",
        "offFaceWarning",
        "regionBlock",
        "faithfulnessBlock",
    ):
        assert f'id="{element_id}"' in page
    assert "results-extra.css" in page
    assert 'id="blurEyesToggle"' in page


def test_new_blocks_start_hidden_so_an_empty_page_shows_none_of_them(client):
    page = client.get("/").get_data(as_text=True)
    for element_id in (
        "rejectedState",
        "privacyCaption",
        "offFaceWarning",
        "regionBlock",
        "faithfulnessBlock",
    ):
        tag = re.search(rf'<[^>]*id="{element_id}"[^>]*>', page).group(0)
        assert "hidden" in tag, f"{element_id} would be visible before any analysis"


def test_low_confidence_note_does_not_sound_like_clinical_advice():
    note = re.search(r'<p id="lowConfNote"[^>]*>(.*?)</p>', INDEX_HTML, re.S).group(1).lower()
    for phrase in ("clinician", "follow-up", "scan", "diagnos"):
        assert phrase not in note, phrase
