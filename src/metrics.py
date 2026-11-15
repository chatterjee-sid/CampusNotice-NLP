"""
metrics.py

Evaluation metrics, all written from scratch:
  * search:          precision@k, recall@k, reciprocal rank, MRR
  * classification:  accuracy, per-class precision / recall / F1, macro averages, confusion matrix
  * extraction:      per-field precision / recall / F1 from true/false positives and false negatives,
                     plus how many wrong extractions were caught by "needs review"

Conventions
  * Classification: a class with no predictions has precision 0.0 (same convention as scikit-learn's default).
  * Extraction / search: an undefined ratio (0/0) is returned as None, never silently as 0 or 1, and
    macro averages skip None entries.
"""


# =============================== search ===============================
def precision_at_k(results: list[str], relevant_docs: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant. If fewer than k results exist the denominator is still k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for d in results[:k] if d in relevant_docs) / k


def recall_at_k(results: list[str], relevant_docs: set[str], k: int) -> float:
    """Fraction of all relevant documents that appear in the top-k results (0.0 if there are none)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_docs:
        return 0.0
    return sum(1 for d in results[:k] if d in relevant_docs) / len(relevant_docs)


def reciprocal_rank(results: list[str], relevant_docs: set[str]) -> float:
    """1 / rank of the first relevant result (1 = first place), or 0.0 if none is retrieved."""
    for rank, d in enumerate(results, start=1):
        if d in relevant_docs:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    """MRR over (ranked_results, relevant_docs) pairs, one per query."""
    if not queries_results:
        return 0.0
    return sum(reciprocal_rank(res, rel) for res, rel in queries_results) / len(queries_results)


def evaluate_search(rankings: dict[str, list[str]], relevant: dict[str, set[str]], k: int = 3) -> dict:
    """P@k and MRR over many queries. Queries without any relevant document are skipped (and counted)."""
    used = [q for q in rankings if relevant.get(q)]
    pairs = [(rankings[q], relevant[q]) for q in used]
    n = len(used)
    return {
        "k": k,
        "n_queries": n,
        "n_skipped": len(rankings) - n,
        f"p@{k}": sum(precision_at_k(r, rel, k) for r, rel in pairs) / n if n else 0.0,
        f"recall@{k}": sum(recall_at_k(r, rel, k) for r, rel in pairs) / n if n else 0.0,
        "mrr": mean_reciprocal_rank(pairs),
        "per_query": {q: {f"p@{k}": precision_at_k(rankings[q], relevant[q], k),
                          "rr": reciprocal_rank(rankings[q], relevant[q])} for q in used},
    }


# ============================ classification ===========================
def evaluate_classification(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    """Accuracy, per-class P/R/F1 (+support), macro averages and the confusion matrix.

    Returns {'labels', 'accuracy', 'per_class': {c: {precision, recall, f1, support}},
             'macro_precision', 'macro_recall', 'macro_f1', 'confusion': {true: {pred: count}}}.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("no predictions to evaluate")
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))

    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1

    per_class = {}
    for c in labels:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in labels if t != c)
        fn = sum(confusion[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    n = len(labels)
    return {
        "labels": labels,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
        "per_class": per_class,
        "macro_precision": sum(v["precision"] for v in per_class.values()) / n,
        "macro_recall": sum(v["recall"] for v in per_class.values()) / n,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / n,
        "confusion": confusion,
    }


def format_classification_report(result: dict) -> str:
    """Plain-text table of the output of evaluate_classification."""
    lines = [f"{'class':<12}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}"]
    for c in result["labels"]:
        v = result["per_class"][c]
        lines.append(f"{c:<12}{v['precision']:>10.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['support']:>9d}")
    lines.append(f"{'macro avg':<12}{result['macro_precision']:>10.3f}{result['macro_recall']:>8.3f}{result['macro_f1']:>8.3f}")
    lines.append(f"accuracy = {result['accuracy']:.3f}")
    return "\n".join(lines)


def format_confusion(result: dict) -> str:
    """Confusion matrix as text (rows = true class, columns = predicted class)."""
    labels = result["labels"]
    w = max(8, *(len(c) + 1 for c in labels))
    lines = ["true \\ pred".ljust(14) + "".join(c.rjust(w) for c in labels)]
    for t in labels:
        lines.append(t.ljust(14) + "".join(str(result["confusion"][t][p]).rjust(w) for p in labels))
    return "\n".join(lines)


# ============================== extraction ==============================
def _norm(value) -> str:
    """Values are compared as case- and whitespace-insensitive strings (dates ISO, fees as integers)."""
    return " ".join(str(value).lower().split())


def _ratio(num: int, den: int):
    return num / den if den else None


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def evaluate_extraction(pred_extractions: list[dict], true_extractions: list[dict],
                        fields: list[str] | None = None) -> dict:
    """Per-field precision / recall / F1.

    Both inputs have one dict per notice (same order): {field: [values]}.
    Per notice and field the predicted and true values are compared as sets:
        true positive  = predicted and labeled        false positive = predicted, not labeled
        false negative = labeled, not predicted
    Returns {'per_field': {field: {tp, fp, fn, precision, recall, f1}}, 'micro': {...}, 'macro_f1': float|None}.
    """
    if len(pred_extractions) != len(true_extractions):
        raise ValueError("need one prediction dict and one label dict per notice")
    if fields is None:
        fields = sorted({f for d in (*pred_extractions, *true_extractions) for f in d})

    per_field = {}
    for f in fields:
        tp = fp = fn = 0
        for pred, true in zip(pred_extractions, true_extractions):
            p = {_norm(v) for v in pred.get(f, [])}
            t = {_norm(v) for v in true.get(f, [])}
            tp += len(p & t)
            fp += len(p - t)
            fn += len(t - p)
        per_field[f] = _prf(tp, fp, fn)

    micro = _prf(sum(v["tp"] for v in per_field.values()),
                 sum(v["fp"] for v in per_field.values()),
                 sum(v["fn"] for v in per_field.values()))
    f1s = [v["f1"] for v in per_field.values() if v["f1"] is not None]
    return {"per_field": per_field, "micro": micro, "macro_f1": sum(f1s) / len(f1s) if f1s else None}


def review_coverage(pred_items: list[dict], true_extractions: list[dict]) -> dict:
    """How well 'needs review' catches wrong extractions.

    pred_items: one dict per notice, {field: [{'value': ..., 'status': 'ok' | 'needs_review'}]}.
    A predicted value is wrong if it is not among that notice's labeled values for the field.
    coverage = flagged wrong / all wrong (higher is better); false_alarm_rate = flagged correct / all correct.
    """
    if len(pred_items) != len(true_extractions):
        raise ValueError("need one prediction dict and one label dict per notice")
    right_ok = right_flag = wrong_ok = wrong_flag = 0
    for pred, true in zip(pred_items, true_extractions):
        for f, items in pred.items():
            truth = {_norm(v) for v in true.get(f, [])}
            for it in items:
                correct = _norm(it["value"]) in truth
                flagged = it.get("status") == "needs_review"
                if correct:
                    right_flag += flagged
                    right_ok += not flagged
                else:
                    wrong_flag += flagged
                    wrong_ok += not flagged
    wrong, correct = wrong_flag + wrong_ok, right_flag + right_ok
    return {
        "correct": correct, "wrong": wrong,
        "flagged_wrong": wrong_flag, "silent_errors": wrong_ok, "flagged_correct": right_flag,
        "coverage": _ratio(wrong_flag, wrong),
        "false_alarm_rate": _ratio(right_flag, correct),
    }
