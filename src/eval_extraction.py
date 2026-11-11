"""
eval_extraction.py

Scores the extractors against data/labels.json, per field, on the DEVELOPMENT side of the split only.
There is deliberately no way to load the test split here.

  python src/eval_extraction.py dev         # the 6 dev notices
  python src/eval_extraction.py train       # the 27 train notices (rules were written looking at these: optimistic)
  python src/eval_extraction.py devset      # train + dev together (33 notices)
  add --errors to list every false positive / false negative per notice

How values are compared
  deadline_dates, event_dates   exact ISO dates          fees   exact integer rupees
  eligibility                   canonical conditions ('cgpa>=7', 'backlogs=0', 'batch=2027', ...) found by the same
                                parser in the label text and in the notice; label strings with no parseable
                                condition (e.g. 'from the GSB') are not scored
  required_documents            gazetteer names ('form', 'resume', ...) found in label text and in predictions
  action_items                  the set of action verbs ('submit', 'attend', ...) used in label text vs predictions
Notices with "extraction_eval": false (calendars, committee lists, branch tables) are skipped.
"""
import argparse
import json
from pathlib import Path

from extractor import analyze_notice, action_verbs, canonical_conditions, canonical_documents
from metrics import evaluate_extraction, review_coverage

DATA = Path(__file__).resolve().parent.parent / "data"
FIELDS = ["deadline_dates", "event_dates", "fees", "eligibility", "required_documents", "action_items"]


def truth_for(label: dict) -> dict:
    cond, docs, verbs = set(), set(), set()
    for s in label["eligibility"]:
        cond |= canonical_conditions(s)
    for s in label["required_documents"]:
        docs |= canonical_documents(s)
    for s in label["action_items"]:
        verbs |= action_verbs(s)
    return {"deadline_dates": list(label["deadline_dates"]), "event_dates": list(label["event_dates"]),
            "fees": list(label["fees"]), "eligibility": sorted(cond), "required_documents": sorted(docs),
            "action_items": sorted(verbs)}


def predict(text: str):
    """(values per field, items with status per field) for one notice."""
    result = analyze_notice(text)
    values, items = {}, {}
    for f in FIELDS:
        if f == "action_items":
            verbs = sorted({v for i in result["fields"][f] for v in i["verbs"]})
            values[f] = verbs
            flagged = {v: "needs_review" if all(i["status"] == "needs_review" for i in result["fields"][f] if v in i["verbs"])
                       else "ok" for v in verbs}
            items[f] = [{"value": v, "status": flagged[v]} for v in verbs]
        else:
            values[f] = [i["value"] for i in result["fields"][f]]
            items[f] = [{"value": i["value"], "status": i["status"]} for i in result["fields"][f]]
    return values, items


def evaluate(doc_names: list[str], show_errors: bool = False):
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    docs = [d for d in doc_names if labels[d]["extraction_eval"]]
    preds, items, truths = [], [], []
    for d in docs:
        v, it = predict((DATA / "texts" / d).read_text(encoding="utf-8"))
        preds.append(v)
        items.append(it)
        truths.append(truth_for(labels[d]))
    res = evaluate_extraction(preds, truths, fields=FIELDS)
    cov = review_coverage(items, truths)
    # the action-verb field is a crude proxy, so also report "needs review" coverage without it
    precise = [{f: v for f, v in it.items() if f != "action_items"} for it in items]
    cov["precise"] = review_coverage(precise, truths)
    if show_errors:
        for d, p, t in zip(docs, preds, truths):
            for f in FIELDS:
                fp, fn = sorted(set(map(str, p[f])) - set(map(str, t[f]))), sorted(set(map(str, t[f])) - set(map(str, p[f])))
                if fp or fn:
                    print(f"  {d[:58]:<58} {f:<19} FP={fp}  FN={fn}")
    return docs, res, cov


def fmt(x):
    return "  n/a" if x is None else f"{x:5.2f}"


def report(title: str, docs, res, cov):
    print(f"\n=== {title}: {len(docs)} notices ===")
    print(f"{'field':<20}{'TP':>4}{'FP':>4}{'FN':>4}  {'prec':>5} {'rec':>5} {'F1':>5}")
    for f in FIELDS:
        v = res["per_field"][f]
        print(f"{f:<20}{v['tp']:>4}{v['fp']:>4}{v['fn']:>4}  {fmt(v['precision'])} {fmt(v['recall'])} {fmt(v['f1'])}")
    m = res["micro"]
    print(f"{'micro (all fields)':<20}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}  {fmt(m['precision'])} {fmt(m['recall'])} {fmt(m['f1'])}")
    print(f"macro F1 over fields: {fmt(res['macro_f1']).strip()}")
    for name, c in (("all fields", cov), ("dates/fees/eligibility/documents", cov["precise"])):
        cv = "n/a" if c["coverage"] is None else f"{c['coverage']:.0%}"
        fa = "n/a" if c["false_alarm_rate"] is None else f"{c['false_alarm_rate']:.0%}"
        print(f"needs-review ({name}): {c['wrong']} wrong predictions, {c['flagged_wrong']} flagged (coverage {cv}), "
              f"{c['silent_errors']} silent; {c['flagged_correct']} of {c['correct']} correct ones flagged "
              f"(false-alarm rate {fa})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", choices=["dev", "train", "devset"])
    ap.add_argument("--errors", action="store_true")
    a = ap.parse_args()
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    names = {"dev": split["dev"], "train": split["train"], "devset": split["train"] + split["dev"]}[a.subset]
    docs, res, cov = evaluate(sorted(names), show_errors=a.errors)
    report(a.subset, docs, res, cov)
