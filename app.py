"""
app.py

CampusNotice AI: Streamlit interface.   Run from campusnotice/:  streamlit run app.py

Pages
  Analyse a notice   upload a PDF / mail (.txt, .eml) or pick a notice from the corpus -> category + extracted fields,
                     each with its evidence sentence highlighted and an OK / Needs-review badge
  Search             TF-IDF (or count / boolean) search over the corpus, with category and CGPA filters
  Deadlines          upcoming deadlines across all notices, soonest first (past ones hidden)
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from classifier import NaiveBayesClassifier          # noqa: E402
from email_reader import email_to_text, scrub_pii    # noqa: E402
from extractor import analyze_notice                 # noqa: E402
from index import METHODS, build_index               # noqa: E402
from pdf_reader import extract_pdf                   # noqa: E402
from preprocess import preprocess                    # noqa: E402
from ui_helpers import (best_sentence, cgpa_requirement, display_title, evidence_html,   # noqa: E402
                        passes_cgpa, upcoming_deadlines)

DATA = ROOT / "data"
TEXTS = DATA / "texts"
FIELD_TITLES = {"deadline_dates": "Deadlines", "event_dates": "Event dates", "fees": "Fees (Rs.)",
                "eligibility": "Eligibility", "required_documents": "Required documents",
                "action_items": "Action items"}
BADGE = {"ok": "✅ OK", "needs_review": "⚠️ Needs review"}


# ------------------------------ cached resources ------------------------------
@st.cache_resource(show_spinner="Training the classifier ...")
def get_classifier() -> NaiveBayesClassifier:
    """Naive Bayes trained on the train + dev notices only (the test split is never used here)."""
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    names = split["train"] + split["dev"]
    docs = [preprocess((TEXTS / n).read_text(encoding="utf-8"), remove_stop=True) for n in names]
    return NaiveBayesClassifier(alpha=0.5).fit(docs, [labels[n]["category"] for n in names])


@st.cache_resource(show_spinner="Indexing notices ...")
def get_index():
    return build_index()


@st.cache_data(show_spinner="Analysing all notices ...")
def get_corpus() -> dict:
    """{doc_id: {"text": raw text, "result": analyze_notice(...), "label": draft category}} for every notice."""
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    out = {}
    for p in sorted(TEXTS.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        out[p.name] = {"text": text, "result": analyze_notice(text), "label": labels.get(p.name, {}).get("category", "?")}
    return out


# ------------------------------ input handling ------------------------------
def read_upload(upload) -> tuple[str, str]:
    """(text, note) from an uploaded PDF / .txt / .eml. Empty text means nothing could be read."""
    suffix = Path(upload.name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(upload.getvalue())
        if suffix == ".pdf":
            text, used_ocr = extract_pdf(path, ocr=True)
            return text, "Text was read by OCR (scanned pages): digits and spaces may be misread." if used_ocr else ""
        # mail: names / roll numbers / e-mails / phones are masked before anything is shown
        return scrub_pii(email_to_text(path)), "Personal details in mails are masked."


# ------------------------------ rendering ------------------------------
def show_result(text: str, result: dict, source_note: str = ""):
    clf_note = "trained on 33 draft-labelled notices; treat as a suggestion"
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", result["category"] or "-", help=clf_note)
    c2.metric("Confidence", f"{result['category_proba']:.0%}" if result["category_proba"] is not None else "-")
    c3.metric("Items needing review", f"{result['n_needs_review']} of {result['n_items']}")
    if result["category_status"] == "needs_review":
        st.warning("The classifier is not sure about the category: please check it.")
    if result["subject"]:
        st.caption(f"Subject: {result['subject']}" + (f"  |  mail sent {result['sent_date']}" if result["sent_date"] else ""))
    if source_note:
        st.info(source_note)

    fields = result["fields"]
    if result["n_items"] == 0:
        st.info("No dates, fees, eligibility rules, documents or actions were found in this text.")
    rows = []
    for f, title in FIELD_TITLES.items():
        for it in fields[f]:
            value = it["value"] if f != "action_items" else ", ".join(it["verbs"])
            rows.append({"Field": title, "Value": str(value), "Confidence": round(it["confidence"], 2),
                         "Status": BADGE[it["status"]]})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Evidence")
    st.caption("Every value comes with the sentence it was read from; the exact match is highlighted.")
    for f, title in FIELD_TITLES.items():
        items = fields[f]
        if not items:
            continue
        with st.expander(f"{title} ({len(items)})", expanded=f in ("deadline_dates", "fees")):
            for it in items:
                head = it["value"] if f != "action_items" else ", ".join(it["verbs"])
                extra = " (conflicting deadlines)" if it.get("conflict") else ""
                st.markdown(f"**{head}**  ·  {BADGE[it['status']]}  ·  confidence {it['confidence']:.2f}{extra}")
                st.markdown(f"<div style='border-left:3px solid #999;padding:2px 10px;margin-bottom:10px'>"
                            f"{evidence_html(it, text)}</div>", unsafe_allow_html=True)
    with st.expander("Full text used for analysis"):
        st.text(text)


def page_analyse():
    st.header("Analyse a notice")
    corpus = get_corpus()
    mode = st.radio("Input", ["Upload a file", "Paste text", "Pick from the corpus"], horizontal=True)
    text, note = "", ""
    if mode == "Upload a file":
        up = st.file_uploader("PDF notice or mail (.pdf, .txt, .eml)", type=["pdf", "txt", "eml"])
        if up is not None:
            failed = False
            with st.spinner("Reading the file (scanned PDFs use OCR and can take a while) ..."):
                try:
                    text, note = read_upload(up)
                except Exception as e:      # corrupt file etc.: show a message, never a traceback
                    failed = True
                    st.error(f"Could not read this file: {e}")
            if not failed and not text.strip():
                st.warning("No text could be read from this file (empty or unreadable scan).")
    elif mode == "Paste text":
        text = scrub_pii(st.text_area("Notice text", height=220))
        note = "Personal details are masked."
    else:
        name = st.selectbox("Notice", list(corpus), format_func=display_title)
        text = corpus[name]["text"]
        note = f"Draft label category: {corpus[name]['label']} (labels are not yet user-verified)"
    if text.strip():
        show_result(text, analyze_notice(text, classifier=get_classifier()), note)


def page_search():
    st.header("Search notices")
    idx, corpus = get_index(), get_corpus()
    query = st.text_input("Query", placeholder="e.g. hostel fee payment, internship stipend, mid semester timetable")
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", list(METHODS), index=0,
                          help="tfidf = cosine on TF-IDF weights (best in our evaluation); count = sum of term counts; "
                               "boolean = all query words must appear")
    cats = c2.multiselect("Category (draft labels)", sorted({c["label"] for c in corpus.values()}))
    use_cgpa = c3.checkbox("I have a CGPA filter")
    cgpa = c3.number_input("My CGPA", 0.0, 10.0, 7.5, 0.1) if use_cgpa else None
    if use_cgpa:
        st.caption("Notices that require a higher CGPA are hidden. A notice with several thresholds is kept if you meet "
                   "the lowest one; notices with no CGPA rule are always kept.")
    if not query.strip():
        st.info("Type a query to search all notices.")
        return
    allowed = {d for d, c in corpus.items()
               if (not cats or c["label"] in cats) and passes_cgpa(c["result"], cgpa)}
    hits = idx.search(query, k=10, method=method, allowed=allowed)
    if not hits:
        st.warning("No notice matches. Try fewer or different words" + (", or relax the filters." if len(allowed) < len(corpus) else "."))
        return
    st.caption(f"{len(hits)} result(s); query words are processed exactly like the notices were.")
    for rank, (doc, score) in enumerate(hits, 1):
        c = corpus[doc]
        req = cgpa_requirement(c["result"])
        st.markdown(f"**{rank}. {display_title(doc)}**  ·  {c['label']}  ·  score {score:.3f}"
                    + (f"  ·  CGPA ≥ {req:g}" if req else ""))
        snippet = best_sentence(c["text"], query)
        if snippet:
            st.caption(snippet[:300])


def page_deadlines():
    st.header("Deadlines")
    corpus = get_corpus()
    today = st.date_input("Today", value=date.today())
    up, n_past, unknown = upcoming_deadlines({d: c["result"] for d, c in corpus.items()}, today)
    st.caption(f"{len(up)} upcoming; {n_past} past deadline(s) hidden. Dates come from the extractor, so check the "
               "⚠️ ones against the notice.")
    if not up:
        st.info("No upcoming deadlines found in the corpus for this date.")
    else:
        df = pd.DataFrame([{"Days left": r["days_left"], "Deadline": r["date"], "Notice": display_title(r["doc"]),
                            "Status": BADGE[r["status"]]} for r in up])
        st.dataframe(df, width="stretch", hide_index=True)
        pick = st.selectbox("Show evidence for", range(len(up)),
                            format_func=lambda i: f"{up[i]['date']}  ·  {display_title(up[i]['doc'])}")
        st.markdown(f"<div style='border-left:3px solid #999;padding:2px 10px'>"
                    f"{evidence_html(up[pick]['item'], corpus[up[pick]['doc']]['text'])}</div>", unsafe_allow_html=True)
    if unknown:
        with st.expander(f"{len(unknown)} deadline(s) written without a year (cannot be placed on the calendar)"):
            for r in unknown:
                st.write(f"{r['date']}  ·  {display_title(r['doc'])}")


def main():
    st.set_page_config(page_title="CampusNotice AI", layout="wide")
    st.title("CampusNotice AI")
    st.caption("College notices -> category, deadlines, fees, eligibility, documents and actions, each with evidence. "
               "Rule-based and Naive Bayes; no LLM.")
    page = st.sidebar.radio("Page", ["Analyse a notice", "Search", "Deadlines"])
    st.sidebar.caption(f"Corpus: {len(list(TEXTS.glob('*.txt')))} notices (labels are drafts, not verified).")
    {"Analyse a notice": page_analyse, "Search": page_search, "Deadlines": page_deadlines}[page]()


if __name__ == "__main__":
    main()
