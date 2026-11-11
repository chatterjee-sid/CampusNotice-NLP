"""Tests for metrics.py. Expected numbers were worked out by hand (shown in each docstring)."""
import pytest

from metrics import (evaluate_classification, evaluate_extraction, evaluate_search, format_classification_report,
                     format_confusion, mean_reciprocal_rank, precision_at_k, recall_at_k, review_coverage,
                     reciprocal_rank)

# ------------------------------ search ------------------------------
RESULTS = ["d1", "d2", "d3", "d4"]
RELEVANT = {"d2", "d4"}


def test_precision_at_k():
    assert precision_at_k(RESULTS, RELEVANT, 1) == 0.0
    assert precision_at_k(RESULTS, RELEVANT, 2) == 0.5
    assert precision_at_k(RESULTS, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RESULTS, RELEVANT, 4) == 0.5


def test_precision_at_k_with_fewer_results_than_k():
    assert precision_at_k(["d2"], RELEVANT, 3) == pytest.approx(1 / 3)     # denominator stays k
    assert precision_at_k([], RELEVANT, 3) == 0.0


def test_recall_at_k():
    assert recall_at_k(RESULTS, RELEVANT, 2) == 0.5
    assert recall_at_k(RESULTS, RELEVANT, 4) == 1.0
    assert recall_at_k(RESULTS, set(), 3) == 0.0


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k(RESULTS, RELEVANT, 0)


def test_reciprocal_rank_and_mrr():
    """query 1: first relevant at rank 2 -> 1/2.  query 2: at rank 3 -> 1/3.  query 3: none -> 0."""
    assert reciprocal_rank(RESULTS, RELEVANT) == 0.5
    assert reciprocal_rank(["x", "y", "d2"], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0
    assert mean_reciprocal_rank([(RESULTS, RELEVANT), (["x", "y", "d2"], RELEVANT)]) == pytest.approx((0.5 + 1 / 3) / 2)
    assert mean_reciprocal_rank([]) == 0.0


def test_evaluate_search_skips_queries_without_relevant_docs():
    out = evaluate_search({"q1": RESULTS, "q2": ["d2", "d9"], "q3": ["d1"]},
                          {"q1": RELEVANT, "q2": {"d2"}, "q3": set()}, k=2)
    assert out["n_queries"] == 2 and out["n_skipped"] == 1
    assert out["p@2"] == pytest.approx((0.5 + 0.5) / 2)
    assert out["mrr"] == pytest.approx((0.5 + 1.0) / 2)
    assert out["per_query"]["q2"]["rr"] == 1.0


# --------------------------- classification ---------------------------
TRUE = ["a", "a", "a", "b", "b", "c"]
PRED = ["a", "a", "b", "b", "c", "c"]


def test_classification_hand_calculation():
    """confusion: a->[a2,b1]  b->[b1,c1]  c->[c1]
    a: P=2/2=1.0   R=2/3      F1=0.8
    b: P=1/2=0.5   R=1/2=0.5  F1=0.5
    c: P=1/2=0.5   R=1/1=1.0  F1=2*.5*1/(1.5)=0.6667
    macro F1 = (0.8 + 0.5 + 0.6667)/3 = 0.6556      accuracy = 4/6
    """
    r = evaluate_classification(TRUE, PRED)
    assert r["labels"] == ["a", "b", "c"]
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["per_class"]["a"] == pytest.approx({"precision": 1.0, "recall": 2 / 3, "f1": 0.8, "support": 3})
    assert r["per_class"]["b"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2})
    assert r["per_class"]["c"] == pytest.approx({"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1})
    assert r["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert r["confusion"] == {"a": {"a": 2, "b": 1, "c": 0}, "b": {"a": 0, "b": 1, "c": 1}, "c": {"a": 0, "b": 0, "c": 1}}


def test_perfect_and_all_wrong():
    assert evaluate_classification(["a", "b"], ["a", "b"])["macro_f1"] == 1.0
    r = evaluate_classification(["a", "b"], ["b", "a"])
    assert r["accuracy"] == 0.0 and r["macro_f1"] == 0.0


def test_class_never_predicted_has_zero_precision_not_a_crash():
    r = evaluate_classification(["a", "b", "b"], ["a", "a", "a"])
    assert r["per_class"]["b"]["precision"] == 0.0 and r["per_class"]["b"]["recall"] == 0.0
    assert r["per_class"]["b"]["f1"] == 0.0


def test_labels_argument_keeps_empty_classes():
    r = evaluate_classification(["a", "a"], ["a", "a"], labels=["a", "b"])
    assert r["per_class"]["b"]["support"] == 0
    assert r["macro_f1"] == pytest.approx(0.5)            # (1.0 + 0.0) / 2


def test_classification_input_validation():
    with pytest.raises(ValueError):
        evaluate_classification(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        evaluate_classification([], [])


def test_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    ours = evaluate_classification(TRUE, PRED)
    p, r, f, s = sk.precision_recall_fscore_support(TRUE, PRED, labels=["a", "b", "c"], zero_division=0)
    for i, c in enumerate(["a", "b", "c"]):
        assert ours["per_class"][c]["precision"] == pytest.approx(p[i])
        assert ours["per_class"][c]["recall"] == pytest.approx(r[i])
        assert ours["per_class"][c]["f1"] == pytest.approx(f[i])
        assert ours["per_class"][c]["support"] == s[i]
    assert ours["macro_f1"] == pytest.approx(sk.f1_score(TRUE, PRED, average="macro"))
    assert ours["accuracy"] == pytest.approx(sk.accuracy_score(TRUE, PRED))
    cm = sk.confusion_matrix(TRUE, PRED, labels=["a", "b", "c"])
    assert [[ours["confusion"][t][p_] for p_ in "abc"] for t in "abc"] == cm.tolist()


def test_text_formatting():
    r = evaluate_classification(TRUE, PRED)
    report = format_classification_report(r)
    assert "macro avg" in report and "accuracy = 0.667" in report
    cm = format_confusion(r).splitlines()
    assert len(cm) == 4 and cm[1].split() == ["a", "2", "1", "0"]


# ----------------------------- extraction -----------------------------
PRED_X = [{"deadline": ["2026-10-12", "2026-10-14"], "fees": [500]}, {"deadline": [], "fees": []}]
TRUE_X = [{"deadline": ["2026-10-12"], "fees": [500, 50]}, {"deadline": ["2026-11-01"], "fees": []}]


def test_extraction_hand_calculation():
    """deadline: tp1 fp1 fn1 -> P .5 R .5 F1 .5      fees: tp1 fp0 fn1 -> P 1 R .5 F1 2/3
    micro: tp2 fp1 fn2 -> P 2/3 R .5 F1 4/7           macro F1 = (0.5 + 2/3)/2
    """
    r = evaluate_extraction(PRED_X, TRUE_X)
    d, f = r["per_field"]["deadline"], r["per_field"]["fees"]
    assert (d["tp"], d["fp"], d["fn"]) == (1, 1, 1)
    assert (d["precision"], d["recall"], d["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (f["tp"], f["fp"], f["fn"]) == (1, 0, 1)
    assert (f["precision"], f["recall"], f["f1"]) == pytest.approx((1.0, 0.5, 2 / 3))
    assert r["micro"]["precision"] == pytest.approx(2 / 3)
    assert r["micro"]["recall"] == pytest.approx(0.5)
    assert r["micro"]["f1"] == pytest.approx(4 / 7)
    assert r["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_extraction_values_are_normalised():
    r = evaluate_extraction([{"fees": ["500"], "docs": ["Resume "]}], [{"fees": [500], "docs": ["resume"]}])
    assert r["per_field"]["fees"]["f1"] == 1.0 and r["per_field"]["docs"]["f1"] == 1.0


def test_extraction_undefined_ratios_are_none():
    r = evaluate_extraction([{"fees": []}], [{"fees": []}])
    f = r["per_field"]["fees"]
    assert f["precision"] is None and f["recall"] is None and f["f1"] is None
    assert r["macro_f1"] is None
    miss = evaluate_extraction([{"fees": []}], [{"fees": [500]}])["per_field"]["fees"]
    assert miss["precision"] is None and miss["recall"] == 0.0 and miss["f1"] == 0.0


def test_extraction_duplicates_count_once():
    r = evaluate_extraction([{"deadline": ["2026-10-12", "2026-10-12"]}], [{"deadline": ["2026-10-12"]}])
    assert r["per_field"]["deadline"]["fp"] == 0


def test_extraction_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_extraction([{}], [{}, {}])


def test_review_coverage():
    """wrong: 2026-10-14 (flagged) and 999 (silent).   correct: 2026-10-12 (ok) and 500 (flagged = false alarm)."""
    pred = [{"deadline": [{"value": "2026-10-12", "status": "ok"}, {"value": "2026-10-14", "status": "needs_review"}]},
            {"fees": [{"value": 999, "status": "ok"}, {"value": 500, "status": "needs_review"}]}]
    true = [{"deadline": ["2026-10-12"]}, {"fees": [500]}]
    r = review_coverage(pred, true)
    assert (r["correct"], r["wrong"], r["flagged_wrong"], r["silent_errors"], r["flagged_correct"]) == (2, 2, 1, 1, 1)
    assert r["coverage"] == 0.5 and r["false_alarm_rate"] == 0.5


def test_review_coverage_without_errors_is_none():
    r = review_coverage([{"fees": [{"value": 500, "status": "ok"}]}], [{"fees": [500]}])
    assert r["wrong"] == 0 and r["coverage"] is None and r["false_alarm_rate"] == 0.0
