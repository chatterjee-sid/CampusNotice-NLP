"""
final_eval.py

THE ONE-TIME TEST-SPLIT EVALUATION. Settings are frozen (see FROZEN) and nothing is tuned afterwards.

  python src/final_eval.py              # runs once, writes data/final_results.json; refuses if that file exists
  python src/final_eval.py --charts     # redraws the charts from the saved json (does not touch the test split)

What it scores on the 6 test notices
  classification   Naive Bayes trained on train + dev (33 notices), compared with always guessing the biggest class
  extraction       per-field P/R/F1 and "needs review" coverage, same scoring as eval_extraction.py
Search is not trained, so it is reported over the whole corpus (eval_search.py); it is copied into the json for the charts.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from classifier import NaiveBayesClassifier
from eval_extraction import FIELDS, evaluate as evaluate_extraction_docs
from eval_search import run as run_search
from metrics import evaluate_classification, format_classification_report, format_confusion
from preprocess import preprocess

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "final_results.json"
FIG = ROOT / "results"
FROZEN = {"alpha": 0.5, "remove_stop": True, "stem_words": False, "review_threshold": 0.6}


def run_once() -> dict:
    if RESULTS.exists():
        sys.exit(f"{RESULTS.name} already exists: the test split was already used. "
                 "Re-running would turn it into a tuning set, so this script refuses.")
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    train, test = sorted(split["train"] + split["dev"]), sorted(split["test"])

    def tokens(d):
        return preprocess((DATA / "texts" / d).read_text(encoding="utf-8"),
                          remove_stop=FROZEN["remove_stop"], stem_words=FROZEN["stem_words"])

    nb = NaiveBayesClassifier(alpha=FROZEN["alpha"]).fit([tokens(d) for d in train], [labels[d]["category"] for d in train])
    y_true = [labels[d]["category"] for d in test]
    y_pred = [nb.predict(tokens(d)) for d in test]
    majority = Counter(labels[d]["category"] for d in train).most_common(1)[0][0]
    cls = evaluate_classification(y_true, y_pred, labels=sorted({labels[d]["category"] for d in labels}))
    cls["majority_class"] = majority
    cls["majority_accuracy"] = sum(t == majority for t in y_true) / len(y_true)
    cls["per_notice"] = [{"notice": d, "true": t, "pred": p} for d, t, p in zip(test, y_true, y_pred)]

    docs, res, cov = evaluate_extraction_docs(test)
    devdocs, dres, dcov = evaluate_extraction_docs(sorted(train))
    _, _, rankings_unused, search = run_search()
    out = {
        "frozen_settings": FROZEN,
        "test_notices": test, "n_test": len(test), "n_test_extraction_scored": len(docs),
        "train_for_classifier": len(train),
        "classification": cls,
        "extraction_test": {"per_field": res["per_field"], "micro": res["micro"], "macro_f1": res["macro_f1"],
                            "review": {k: v for k, v in cov.items() if k != "precise"}, "review_precise": cov["precise"]},
        "extraction_devset": {"per_field": dres["per_field"], "micro": dres["micro"], "macro_f1": dres["macro_f1"],
                              "n": len(devdocs)},
        "search": {m: {k: v for k, v in r.items() if k != "per_query"} for m, r in search.items()},
    }
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def print_report(r: dict):
    c = r["classification"]
    print(f"\n=== CLASSIFICATION on the {r['n_test']} test notices (trained on {r['train_for_classifier']}) ===")
    print(format_classification_report(c))
    print(format_confusion(c))
    print(f"always-'{c['majority_class']}' baseline accuracy = {c['majority_accuracy']:.3f}")
    for n in c["per_notice"]:
        flag = "ok   " if n["true"] == n["pred"] else "WRONG"
        print(f"  {flag} true={n['true']:<10} pred={n['pred']:<10} {n['notice'][:60]}")
    e = r["extraction_test"]
    print(f"\n=== EXTRACTION on the test notices ({r['n_test_extraction_scored']} scored) ===")
    print(f"{'field':<20}{'TP':>4}{'FP':>4}{'FN':>4}  {'prec':>5} {'rec':>5} {'F1':>5}  | devset F1")
    f = lambda x: "  n/a" if x is None else f"{x:5.2f}"
    for fld in FIELDS:
        v, d = e["per_field"][fld], r["extraction_devset"]["per_field"][fld]
        print(f"{fld:<20}{v['tp']:>4}{v['fp']:>4}{v['fn']:>4}  {f(v['precision'])} {f(v['recall'])} {f(v['f1'])}  | {f(d['f1'])}")
    m = e["micro"]
    print(f"{'micro':<20}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}  {f(m['precision'])} {f(m['recall'])} {f(m['f1'])}")
    p = e["review_precise"]
    print(f"needs-review (dates/fees/eligibility/documents): {p['wrong']} wrong, {p['flagged_wrong']} flagged, "
          f"{p['flagged_correct']} of {p['correct']} correct flagged")
    print("\n=== SEARCH (whole corpus, 15 queries) ===")
    for mth, v in r["search"].items():
        print(f"  {mth:<8} P@3 {v['p@3']:.3f}  R@3 {v['recall@3']:.3f}  MRR {v['mrr']:.3f}")


def draw_charts(r: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    ink, muted, grid = "#1f2328", "#6b7280", "#e5e7eb"
    accent, soft, neutral = "#2a6fb0", "#a9c6e4", "#b8bdc5"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": grid, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink, "figure.dpi": 150})

    def tidy(ax):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.xaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    # 1. search methods
    fig, ax = plt.subplots(figsize=(6.4, 3))
    names = ["boolean", "count", "tfidf"]
    vals = [r["search"][m]["mrr"] for m in names]
    labels = ["Boolean AND", "Raw count", "TF-IDF (ours)"]
    bars = ax.barh(labels[::-1], vals[::-1], color=[accent, soft, neutral], height=0.55)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_title("Search quality: MRR over 15 queries", loc="left", fontsize=11, color=ink)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "search_mrr.png")
    plt.close(fig)

    # 2. extraction per field: devset vs test
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fields = FIELDS
    import numpy as np
    y = np.arange(len(fields))
    dev = [r["extraction_devset"]["per_field"][f]["f1"] or 0 for f in fields]
    tst = [r["extraction_test"]["per_field"][f]["f1"] or 0 for f in fields]
    h = 0.36
    ax.barh(y - h / 2, dev, height=h, color=soft, label="train+dev (rules written here)")
    ax.barh(y + h / 2, tst, height=h, color=accent, label="test (6 notices)")
    for yy, v in zip(y - h / 2, dev):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8, color=muted)
    for yy, v in zip(y + h / 2, tst):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [f.replace("_", " ") for f in fields])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_title("Extraction F1 per field", loc="left", fontsize=11, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(FIG / "extraction_f1.png")
    plt.close(fig)

    # 3. confusion matrix, single hue
    c = r["classification"]
    labs = c["labels"]
    m = np.array([[c["confusion"][t][p] for p in labs] for t in labs])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("one", ["#ffffff", accent]), vmin=0)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else ink)
    ax.set_xticks(range(len(labs)), labs)
    ax.set_yticks(range(len(labs)), labs)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Category, test notices (accuracy {c['accuracy']:.2f}, n={r['n_test']})", loc="left", fontsize=10, color=ink)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_test.png")
    plt.close(fig)
    print(f"charts written to {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", action="store_true", help="redraw charts from the saved json only")
    a = ap.parse_args()
    if a.charts:
        draw_charts(json.loads(RESULTS.read_text(encoding="utf-8")))
    else:
        res = run_once()
        print_report(res)
        draw_charts(res)
