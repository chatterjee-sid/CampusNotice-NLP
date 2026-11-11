"""
ui_helpers.py

Pure functions behind the Streamlit app (no Streamlit import), so they can be unit-tested.
"""
import html
import re
from datetime import date

from preprocess import preprocess, split_sentences

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def evidence_html(item: dict, text: str) -> str:
    """The evidence sentence(s) of one extracted item, HTML-escaped, with the matched span wrapped in <mark>.

    `text` must be the same raw text the item was extracted from (offsets refer to it).
    """
    start, end = item["start"], item["end"]
    ev = text[start:end]
    m0, m1 = item.get("match_start"), item.get("match_end")
    if m0 is None or m1 is None or not (start <= m0 < m1 <= end):
        return _tidy(html.escape(ev))
    return _tidy(html.escape(text[start:m0]) + "<mark>" + html.escape(text[m0:m1]) + "</mark>" + html.escape(text[m1:end]))


def _tidy(s: str) -> str:
    """Collapse runs of whitespace so mail line-wrapping does not look ragged."""
    return re.sub(r"\s+", " ", s).strip()


def cgpa_requirement(result: dict) -> float | None:
    """The lowest CGPA threshold ('>=') found in a notice's eligibility, or None if it states none.

    The lowest is used because a notice can list several programmes with different thresholds;
    the filter should not hide a notice a student may still be eligible for.
    """
    vals = []
    for item in result["fields"]["eligibility"]:
        c = item.get("condition", {})
        if c.get("type") == "cgpa" and c.get("op") == ">=":
            try:
                vals.append(float(c["value"]))
            except (TypeError, ValueError):
                pass
    return min(vals) if vals else None


def passes_cgpa(result: dict, cgpa: float | None) -> bool:
    """True if no CGPA is given, the notice has no CGPA rule, or the student's CGPA meets the (lowest) rule."""
    if cgpa is None:
        return True
    req = cgpa_requirement(result)
    return req is None or cgpa >= req


def upcoming_deadlines(results: dict[str, dict], today: date) -> tuple[list[dict], int, list[dict]]:
    """(upcoming, n_past, unknown_year) from {doc_id: analyze_notice result}.

    upcoming     deadlines on or after `today`, soonest first, each with days_left, status, evidence
    n_past       how many deadlines are already over (hidden from the dashboard)
    unknown_year dates written without a year (stored as '????-MM-DD'): cannot be compared with today
    """
    upcoming, unknown, n_past = [], [], 0
    for doc, res in results.items():
        for item in res["fields"]["deadline_dates"]:
            v = item["value"]
            row = {"doc": doc, "date": v, "status": item["status"], "confidence": item["confidence"], "item": item}
            if not _ISO.match(v):
                unknown.append(row)
                continue
            d = date.fromisoformat(v)
            if d < today:
                n_past += 1
            else:
                row["days_left"] = (d - today).days
                upcoming.append(row)
    upcoming.sort(key=lambda r: (r["days_left"], r["doc"]))
    return upcoming, n_past, unknown


def best_sentence(text: str, query: str) -> str:
    """The sentence of `text` sharing the most query tokens (for a search-result snippet); '' if none shares any."""
    q = set(preprocess(query))
    best, best_n = "", 0
    for s in split_sentences(text):
        n = len(q & set(preprocess(s["text"])))
        if n > best_n:
            best, best_n = s["text"], n
    return _tidy(best)


def display_title(doc_id: str) -> str:
    """File name -> readable title (drops .txt and the numeric mail suffix such as '-4065')."""
    t = doc_id.removesuffix(".txt")
    t = re.sub(r"-\d{3,4}$", "", t)
    return t.removeprefix("mail_")
