"""
eval_classifier.py

Classifier evaluation on the DEV side of the data. There is deliberately no way to load the test split here;
the test set is touched once, by the final evaluation script.

  python src/eval_classifier.py dev    # train on split['train'], evaluate on split['dev'] (6 docs: very noisy)
  python src/eval_classifier.py cv     # leave-one-unit-out on train+dev (near-duplicate groups held out together)
  options: --stop/--no-stop  --stem/--no-stem  --alpha 0.5  --shape-weight N
"""
import argparse
import json
from pathlib import Path

from classifier import NaiveBayesClassifier
from metrics import evaluate_classification, format_classification_report, format_confusion
from preprocess import preprocess

DATA = Path(__file__).resolve().parent.parent / "data"


def load(settings: dict):
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))

    def tokens(doc):
        text = (DATA / "texts" / doc).read_text(encoding="utf-8")
        return preprocess(text, remove_stop=settings["stop"], stem_words=settings["stem"],
                          shape_weight=settings.get("shape", 0))

    return labels, split, tokens


def run_dev(settings: dict):
    labels, split, tokens = load(settings)
    nb = NaiveBayesClassifier(alpha=settings["alpha"]).fit([tokens(d) for d in split["train"]],
                                                           [labels[d]["category"] for d in split["train"]])
    y_true = [labels[d]["category"] for d in split["dev"]]
    y_pred = [nb.predict(tokens(d)) for d in split["dev"]]
    wrong = [(d, labels[d]["category"], p) for d, p in zip(split["dev"], y_pred) if labels[d]["category"] != p]
    return y_true, y_pred, wrong, f"train on {len(split['train'])} docs, evaluate on dev ({len(split['dev'])} docs)"


def run_cv(settings: dict):
    labels, split, tokens = load(settings)
    docs = sorted(split["train"] + split["dev"])          # never split['test']
    toks = {d: tokens(d) for d in docs}
    unit = {d: labels[d]["group"] or d for d in docs}
    y_true, y_pred, wrong = [], [], []
    for u in sorted(set(unit.values())):
        held = [d for d in docs if unit[d] == u]
        rest = [d for d in docs if unit[d] != u]
        nb = NaiveBayesClassifier(alpha=settings["alpha"]).fit([toks[d] for d in rest],
                                                               [labels[d]["category"] for d in rest])
        for d in held:
            p = nb.predict(toks[d])
            y_true.append(labels[d]["category"])
            y_pred.append(p)
            if p != labels[d]["category"]:
                wrong.append((d, labels[d]["category"], p))
    return y_true, y_pred, wrong, f"leave-one-unit-out over {len(docs)} docs in {len(set(unit.values()))} units"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "cv"])
    ap.add_argument("--stop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stem", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--shape-weight", type=int, default=0, help="repeat the date/number shape features this many times")
    a = ap.parse_args()
    settings = {"stop": a.stop, "stem": a.stem, "alpha": a.alpha, "shape": a.shape_weight}
    y_true, y_pred, wrong, desc = (run_dev if a.mode == "dev" else run_cv)(settings)
    result = evaluate_classification(y_true, y_pred, labels=["exam", "general", "hostel", "placement"])
    print(f"mode={a.mode}  settings={settings}\n{desc}\n")
    print(format_classification_report(result))
    print("\nconfusion matrix\n" + format_confusion(result))
    print(f"\nmisclassified ({len(wrong)}):")
    for d, t, p in wrong:
        print(f"  true={t:<9} pred={p:<9} {d[:75]}")
