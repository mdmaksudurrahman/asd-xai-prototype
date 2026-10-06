# -*- coding: utf-8 -*-
"""
User-facing wording that must be identical on every page.

Kept in one dependency-free module so the web app, the About page and the
standalone lab-validation report all say exactly the same thing, and so a
test can check that no page describes this prototype as a screening aid.
"""

# Shown at the top of every page (Task 6). It must also print, so it is
# deliberately NOT marked "no-print" in the templates.
RESEARCH_BANNER = (
    "Research prototype – not a diagnostic or screening tool. "
    "Do not use it to make decisions about any child."
)

# Shown under each result.
RESULT_DISCLAIMER = (
    "Research prototype output — not a diagnosis and not a screening tool. "
    "Any concern about a child's development should be discussed with a qualified clinician."
)

# Phrases that must never describe this prototype. Checked by tests/test_about_page.py
# against every rendered page.
FORBIDDEN_PHRASES = ("screening aid", "ASD Screening")
