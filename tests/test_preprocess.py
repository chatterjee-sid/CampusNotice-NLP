import pytest

from preprocess import (STOPWORDS, clean_text, count_date_mentions, preprocess, remove_stopwords,
                        shape_features, split_sentences, stem, tokenize)


# ---------- tokenize ----------
@pytest.mark.parametrize("text, expected", [
    ("12/10/2026", ["12/10/2026"]),
    ("12-10-2026", ["12-10-2026"]),
    ("12.10.26", ["12.10.26"]),                 # day.month.yy used in many notices
    ("12.10.2026", ["12.10.2026"]),
    ("7.5", ["7.5"]),
    ("60%", ["60%"]),
    ("7.5%", ["7.5%"]),
    ("10:30", ["10:30"]),
])
def test_dates_numbers_times_stay_whole(text, expected):
    assert tokenize(text) == expected


def test_roadmap_example():
    assert preprocess("Last date: 12/10/2026, CGPA >= 7.5") == ["last", "date", "12/10/2026", "cgpa", "7.5"]


def test_academic_year_is_not_a_date():
    assert tokenize("2026-27") == ["2026", "27"]


def test_money_formats_split_into_word_and_number():
    assert tokenize("rs.500/-") == ["rs", "500"]
    assert tokenize("rs. 1,200") == ["rs", "1", "200"]


def test_underscores_are_separators():
    # mail formatting leaves runs of underscores ("Surat____"); they must not become tokens
    assert tokenize("surat____ ok __ a_b") == ["surat", "ok", "a", "b"]


def test_empty_and_whitespace():
    assert tokenize("") == []
    assert preprocess("   \n\t ") == []


# ---------- clean_text ----------
def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Hostel \n\n  FEES\tnotice ") == "hostel fees notice"


# ---------- stop words ----------
def test_stopwords_removed_but_numbers_kept():
    assert remove_stopwords(["the", "last", "date", "is", "12/10/2026", "and", "7.5"]) == ["last", "date", "12/10/2026", "7.5"]


def test_notice_critical_words_are_not_stopwords():
    for w in ("last", "date", "no", "not", "before", "after", "on", "till"):
        assert w not in STOPWORDS


# ---------- stemming ----------
def test_stem_conflates_register_family():
    assert stem("registered") == stem("registering") == stem("registration") == stem("register")


def test_stem_plurals():
    assert stem("fees") == stem("fee") == "fee"
    assert stem("notices") == stem("notice")
    assert stem("categories") == "category"


def test_stem_leaves_special_tokens_alone():
    assert stem("12/10/2026") == "12/10/2026"
    assert stem("cgpa") == "cgpa"
    assert stem("class") == "class"             # double s is not a plural
    assert stem("fee") == "fee"
    assert stem("ing") == "ing"


def test_stem_doubled_consonant():
    assert stem("running") == stem("run")


# ---------- preprocess options ----------
def test_preprocess_default_is_plain():
    assert preprocess("The Fees are due") == ["the", "fees", "are", "due"]


def test_preprocess_options():
    assert preprocess("The Fees are due", remove_stop=True) == ["fees", "due"]
    assert preprocess("The Fees are due", remove_stop=True, stem_words=True) == ["fee", "due"]


# ---------- sentences ----------
def test_abbreviations_do_not_split():
    s = split_sentences("Dr. Shah will meet the students. Please register on time.")
    assert [x["text"] for x in s] == ["Dr. Shah will meet the students.", "Please register on time."]


def test_rs_and_decimals_do_not_split():
    s = split_sentences("The fee is Rs. 500 and CGPA must be 7.5 or more. Apply soon.")
    assert len(s) == 2
    assert s[0]["text"].endswith("or more.")


def test_numbered_bullets_start_new_sentences():
    s = split_sentences("Documents required:\n1. Resume\n2. Marksheet\n3. ID card")
    assert [x["text"] for x in s] == ["Documents required:", "1. Resume", "2. Marksheet", "3. ID card"]


def test_sentence_offsets_point_into_original_text():
    text = "  First sentence here.  Second one follows.\nThird after newline."
    for s in split_sentences(text):
        assert text[s["start"]:s["end"]] == s["text"]


def test_no_sentences_for_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# ---------- shape features (date / number density) ----------
def test_count_date_mentions_in_several_formats():
    text = ("Exam on 23-Nov-26 and 25-Nov-26; result on 12/10/2026; held on October 12, 2026; "
            "meeting on 5th May 2026.")
    assert count_date_mentions(text) == 5


def test_count_date_mentions_ignores_words_that_only_contain_month_letters():
    assert count_date_mentions("separate marksheet document, decision pending, marching band") == 0
    assert count_date_mentions("") == 0


def test_shape_features_buckets():
    assert shape_features("no dates here", ["a"] * 20) == ["shape_dates_none", "shape_nums_low"]
    assert shape_features("due 12/10/2026", ["12"] * 5 + ["x"] * 5)[1] == "shape_nums_very_high"
    many = " ".join(f"{d}-nov-26" for d in range(10, 20))
    assert shape_features(many, ["x"] * 10)[0] == "shape_dates_many"
    assert shape_features("on 1/1/2026 and 2/2/2026 and 3/3/2026", ["x"])[0] == "shape_dates_several"
    assert shape_features("on 1/1/2026", ["x"])[0] == "shape_dates_few"


def test_preprocess_shape_weight_appends_feature_tokens():
    plain = preprocess("Fee due 12/10/2026")
    assert preprocess("Fee due 12/10/2026", shape_weight=0) == plain          # off by default
    with_shape = preprocess("Fee due 12/10/2026", shape_weight=3)
    assert with_shape[:len(plain)] == plain
    assert len(with_shape) == len(plain) + 6                                  # 2 feature tokens x 3 copies
    assert with_shape.count("shape_dates_few") == 3
