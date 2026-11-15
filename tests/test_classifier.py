"""Tests for classifier.py. The tiny example was also computed by hand:

    spam: "buy now", "buy cheap now"      ham: "meeting today", "project meeting"
    V = {buy, now, cheap, meeting, today, project}, |V| = 6;  priors 1/2 and 1/2
    spam words: buy 2, now 2, cheap 1  (total 5)     ham words: meeting 2, today 1, project 1  (total 4)
    P(buy|spam) = (2+1)/(5+6) = 3/11     P(meeting|spam) = (0+1)/11 = 1/11
    P(buy|ham)  = (0+1)/(4+6) = 1/10     P(meeting|ham)  = (2+1)/10 = 3/10
    new doc "buy meeting":  spam = 1/2 * 3/11 * 1/11 = 3/242     ham = 1/2 * 1/10 * 3/10 = 3/200   -> ham wins
"""
import math

import pytest

from classifier import NaiveBayesClassifier

DOCS = [["buy", "now"], ["buy", "cheap", "now"], ["meeting", "today"], ["project", "meeting"]]
LABELS = ["spam", "spam", "ham", "ham"]


@pytest.fixture
def nb():
    return NaiveBayesClassifier().fit(DOCS, LABELS)


def test_hand_calculation(nb):
    scores = nb.predict_scores(["buy", "meeting"])
    assert scores["spam"] == pytest.approx(math.log(3 / 242), abs=1e-9)
    assert scores["ham"] == pytest.approx(math.log(3 / 200), abs=1e-9)
    assert nb.predict(["buy", "meeting"]) == "ham"


def test_learns_training_data(nb):
    for tokens, label in zip(DOCS, LABELS):
        assert nb.predict(tokens) == label


def test_priors_follow_class_frequency():
    nb = NaiveBayesClassifier().fit([["a"], ["a"], ["b"]], ["x", "x", "y"])
    assert nb.log_prior["x"] == pytest.approx(math.log(2 / 3))
    assert nb.log_prior["y"] == pytest.approx(math.log(1 / 3))


def test_smoothing_avoids_zero_probability(nb):
    # 'cheap' never appears in ham, but its ham likelihood is still positive (log is finite)
    assert math.isfinite(nb.log_likelihood("cheap", "ham"))
    assert math.isfinite(nb.predict_scores(["cheap"])["ham"])


def test_log_probabilities_do_not_underflow(nb):
    scores = nb.predict_scores(["buy"] * 5000)
    assert all(math.isfinite(s) for s in scores.values())
    assert nb.predict(["buy"] * 5000) == "spam"


def test_unknown_words_are_ignored(nb):
    assert nb.predict_scores(["zzz", "buy"]) == nb.predict_scores(["buy"])


def test_empty_doc_falls_back_to_priors():
    nb = NaiveBayesClassifier().fit([["a"], ["a"], ["b"]], ["x", "x", "y"])
    assert nb.predict([]) == "x"                      # most common class


def test_predict_proba_sums_to_one(nb):
    p = nb.predict_proba(["buy", "meeting"])
    assert sum(p.values()) == pytest.approx(1.0)
    assert max(p, key=p.get) == nb.predict(["buy", "meeting"])


def test_valid_label_returned(nb):
    assert nb.predict(["anything", "goes"]) in {"spam", "ham"}


def test_ties_are_deterministic():
    nb = NaiveBayesClassifier().fit([["a"], ["b"]], ["y", "x"])
    assert nb.predict(["a", "b"]) == "x"              # equal scores -> alphabetical


def test_alpha_changes_smoothing():
    a = NaiveBayesClassifier(alpha=1.0).fit(DOCS, LABELS)
    b = NaiveBayesClassifier(alpha=0.5).fit(DOCS, LABELS)
    assert a.log_likelihood("cheap", "ham") != b.log_likelihood("cheap", "ham")
    with pytest.raises(ValueError):
        NaiveBayesClassifier(alpha=0)


def test_input_validation():
    with pytest.raises(ValueError):
        NaiveBayesClassifier().fit([["a"]], ["x", "y"])
    with pytest.raises(RuntimeError):
        NaiveBayesClassifier().predict(["a"])


def test_top_words(nb):
    assert nb.top_words("spam", 1)[0][0] in {"buy", "now", "cheap"}


def test_save_and_load_roundtrip(nb, tmp_path):
    path = tmp_path / "nb.json"
    nb.save(path)
    loaded = NaiveBayesClassifier.load(path)
    assert loaded.predict_scores(["buy", "meeting"]) == pytest.approx(nb.predict_scores(["buy", "meeting"]))
    assert loaded.vocab == nb.vocab
