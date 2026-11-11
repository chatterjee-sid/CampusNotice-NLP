import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extractor import analyze_notice
from ui_helpers import (best_sentence, cgpa_requirement, display_title, evidence_html, passes_cgpa,
                        upcoming_deadlines)

NOTICE = ("Subject: Speaker session\nDate: 01-10-2026, 10:00\n\n"
          "Students with CGPA >= 7.0 may apply. Submit the form by 12/10/2026. Registration fee is Rs. 500.")


def test_evidence_html_marks_match_and_escapes():
    text = "Pay <fee> by 12/10/2026 now."
    item = {"start": 0, "end": len(text), "match_start": text.index("12/10"), "match_end": text.index("12/10") + 10}
    out = evidence_html(item, text)
    assert "<mark>12/10/2026</mark>" in out
    assert "&lt;fee&gt;" in out and "<fee>" not in out


def test_evidence_html_without_match_span_is_plain():
    text = "Some sentence."
    assert evidence_html({"start": 0, "end": len(text)}, text) == "Some sentence."


def test_evidence_html_collapses_whitespace():
    text = "line one\n   line two"
    assert evidence_html({"start": 0, "end": len(text)}, text) == "line one line two"


def test_cgpa_requirement_and_filter():
    res = analyze_notice(NOTICE)
    assert cgpa_requirement(res) == 7.0
    assert passes_cgpa(res, 7.5) and passes_cgpa(res, 7.0)
    assert not passes_cgpa(res, 6.9)
    assert passes_cgpa(res, None)


def test_notice_without_cgpa_rule_always_passes():
    res = analyze_notice("The library will remain closed on 20/11/2026.")
    assert cgpa_requirement(res) is None
    assert passes_cgpa(res, 5.0)


def test_upcoming_deadlines_hides_past_and_sorts():
    res = {"a.txt": analyze_notice("Last date to apply is 12/10/2026."),
           "b.txt": analyze_notice("Last date to apply is 20/10/2026."),
           "c.txt": analyze_notice("Last date to apply is 01/09/2026.")}
    up, n_past, unknown = upcoming_deadlines(res, date(2026, 10, 10))
    assert [r["doc"] for r in up] == ["a.txt", "b.txt"]
    assert [r["days_left"] for r in up] == [2, 10]
    assert n_past == 1 and unknown == []


def test_deadline_today_is_upcoming_with_zero_days_left():
    res = {"a.txt": analyze_notice("Last date to apply is 12/10/2026.")}
    up, n_past, _ = upcoming_deadlines(res, date(2026, 10, 12))
    assert up[0]["days_left"] == 0 and n_past == 0


def test_yearless_deadline_goes_to_unknown_year():
    fake = {"a.txt": {"fields": {"deadline_dates": [{"value": "????-10-12", "status": "needs_review",
                                                      "confidence": 0.5}]}}}
    up, n_past, unknown = upcoming_deadlines(fake, date(2026, 10, 1))
    assert up == [] and n_past == 0 and len(unknown) == 1


def test_best_sentence_picks_overlapping_sentence():
    s = best_sentence("The sky is blue. Registration fee is Rs. 500. Bring a pen.", "registration fee")
    assert s.startswith("Registration fee")
    assert best_sentence("The sky is blue.", "zebra") == ""


def test_display_title():
    assert display_title("mail_20260804-Blood donation camp-4065.txt") == "20260804-Blood donation camp"
    assert display_title("Mess Change Notice A.Y. 2026-27.txt") == "Mess Change Notice A.Y. 2026-27"
