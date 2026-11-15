"""Tests for index.py. The 3-document example below was also worked out by hand on paper:

    d1 = apple banana apple      d2 = banana cherry      d3 = cherry date        N = 3
    df: apple 1, banana 2, cherry 2, date 1
    idf(apple) = ln 3 = 1.0986   idf(banana) = idf(cherry) = ln 1.5 = 0.4055
    tfidf(apple, d1)  = (1 + ln 2) * ln 3   = 1.8601
    tfidf(banana, d1) = 1 * ln 1.5          = 0.4055
    |d1| = sqrt(1.8601^2 + 0.4055^2)        = 1.9038
    cosine("apple", d1) = 1.8601 / 1.9038   = 0.9770
"""
import math

import pytest

from index import InvertedIndex, build_index


@pytest.fixture
def idx():
    i = InvertedIndex()
    i.add_document("d1", ["apple", "banana", "apple"])
    i.add_document("d2", ["banana", "cherry"])
    i.add_document("d3", ["cherry", "date"])
    return i


def test_postings_and_lengths(idx):
    assert idx.index["apple"] == {"d1": 2}
    assert idx.index["banana"] == {"d1": 1, "d2": 1}
    assert idx.doc_lengths == {"d1": 3, "d2": 2, "d3": 2}
    assert idx.n_docs == 3


def test_duplicate_doc_id_rejected(idx):
    with pytest.raises(ValueError):
        idx.add_document("d1", ["x"])


def test_idf_matches_hand_calculation(idx):
    assert idx.idf("apple") == pytest.approx(1.0986, abs=1e-4)
    assert idx.idf("banana") == pytest.approx(0.4055, abs=1e-4)
    assert idx.idf("missing") == 0.0


def test_tfidf_matches_hand_calculation(idx):
    assert idx.tfidf("apple", "d1") == pytest.approx(1.8601, abs=1e-4)
    assert idx.tfidf("banana", "d1") == pytest.approx(0.4055, abs=1e-4)
    assert idx.tfidf("apple", "d2") == 0.0          # term not in doc


def test_log_tf_dampens_repetition():
    assert InvertedIndex.tf_weight(20) < 20 * InvertedIndex.tf_weight(1)
    assert InvertedIndex.tf_weight(1) == 1.0


def test_boolean_and(idx):
    assert idx.boolean_and(["banana", "cherry"]) == ["d2"]
    assert idx.boolean_and(["banana"]) == ["d1", "d2"]
    assert idx.boolean_and(["banana", "date"]) == []
    assert idx.boolean_and(["unknown"]) == []


def test_cosine_score_matches_hand_calculation(idx):
    results = idx.search("apple")
    assert [d for d, _ in results] == ["d1"]
    assert results[0][1] == pytest.approx(0.9770, abs=1e-3)


def test_ranking_order(idx):
    # d2 contains both query words, d1 and d3 only one each
    ranked = [d for d, _ in idx.search("banana cherry")]
    assert ranked[0] == "d2"
    assert set(ranked) == {"d1", "d2", "d3"}


def test_scores_sorted_descending_and_capped_by_k(idx):
    results = idx.search("banana cherry date", k=2)
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_query_is_preprocessed_like_documents(idx):
    assert idx.search("APPLE") == idx.search("apple")


def test_unknown_query_returns_empty(idx):
    assert idx.search("zebra") == []
    assert idx.search("") == []


def test_methods(idx):
    assert idx.search("banana cherry", method="boolean") == [("d2", 2.0)]
    count = dict(idx.search("apple banana", method="count"))
    assert count == {"d1": 3.0, "d2": 1.0}
    with pytest.raises(ValueError):
        idx.search("apple", method="nope")


def test_boolean_with_unknown_term_is_empty(idx):
    assert idx.search("banana zebra", method="boolean") == []


def test_allowed_filter(idx):
    results = idx.search("banana cherry", allowed={"d1", "d3"})
    assert {d for d, _ in results} == {"d1", "d3"}      # d2 would rank first but is filtered out


def test_term_in_every_doc_has_zero_idf():
    i = InvertedIndex()
    i.add_document("a", ["notice", "exam"])
    i.add_document("b", ["notice", "fee"])
    assert i.idf("notice") == 0.0
    assert i.search("notice") == []                 # no information -> no ranking
    assert [d for d, _ in i.search("notice exam")] == ["a"]


def test_save_and_load_roundtrip(idx, tmp_path):
    path = tmp_path / "index.json"
    idx.save(path)
    loaded = InvertedIndex.load(path)
    assert loaded.index == idx.index
    assert loaded.search("banana cherry") == idx.search("banana cherry")
    assert math.isclose(loaded.tfidf("apple", "d1"), idx.tfidf("apple", "d1"))


def test_build_index_from_texts(tmp_path):
    (tmp_path / "a.txt").write_text("Hostel mess change notice", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Exam timetable for the semester", encoding="utf-8")
    i = build_index(texts_dir=tmp_path)
    assert i.n_docs == 2
    assert i.search("exam timetable")[0][0] == "b.txt"
    assert build_index(["a.txt"], texts_dir=tmp_path).n_docs == 1
