"""Tests for extractor.py: dates, fees, documents, eligibility, actions and analyze_notice."""
import pytest

from classifier import NaiveBayesClassifier
from extractor import (action_verbs, analyze_notice, canonical_conditions, canonical_documents, find_actions,
                       find_date_mentions, find_dates, find_documents, find_eligibility, find_fees, fold,
                       parse_mail_header)


def ymd(text):
    """(year, month, day) of every explicit date mention found in text."""
    return [(m["y"], m["m"], m["d"]) for m in find_date_mentions(fold(text))]


def values(items, field=None):
    return sorted(i["value"] for i in items if field is None or i["field"] == field)


# =============================== date formats ===============================
@pytest.mark.parametrize("text, expected", [
    ("12/10/2026", (2026, 10, 12)),
    ("12-10-2026", (2026, 10, 12)),
    ("12.10.2026", (2026, 10, 12)),
    ("12.10.26", (2026, 10, 12)),                    # two-digit year
    ("03/11/26", (2026, 11, 3)),
    ("2026-10-12", (2026, 10, 12)),                  # ISO
    ("12th October 2026", (2026, 10, 12)),
    ("12 Oct 2026", (2026, 10, 12)),
    ("12 October, 2026", (2026, 10, 12)),
    ("1st November 2026", (2026, 11, 1)),
    ("2nd Nov 2026", (2026, 11, 2)),
    ("October 12, 2026", (2026, 10, 12)),
    ("Oct 12 2026", (2026, 10, 12)),
    ("12-Oct-26", (2026, 10, 12)),
    ("12th Oct. 2026", (2026, 10, 12)),
    ("12October2026", (2026, 10, 12)),               # OCR glues words together
    ("0ctober 22,2026", (2026, 10, 22)),             # OCR reads the O of October as a zero
    ("July15,2026", (2026, 7, 15)),
    ("*6*^*th* *November, 2026.*", (2026, 11, 6)),   # mail markup around day, suffix and month
    ("12 Oct", (None, 10, 12)),                      # no year: resolved later from context
])
def test_date_formats(text, expected):
    assert ymd(text) == [expected]


def test_dates_are_day_first_never_guessed():
    assert ymd("05/06/2026") == [(2026, 6, 5)]       # 5 June, not 6 May
    assert ymd("12/10/2026") == [(2026, 10, 12)]     # 12 October, not 10 December


@pytest.mark.parametrize("text", ["31/02/2026", "32/01/2026", "15/13/2026", "30 Feb 2026", "00/10/2026"])
def test_impossible_dates_are_rejected(text):
    assert ymd(text) == []


def test_month_letters_inside_other_words_are_not_dates():
    assert ymd("the separate marksheet decision in march marching 5 mark 12") == []
    assert ymd("Jan'27-Jun'27 internship") == []     # month-year abbreviations are not day-month dates


def test_academic_year_is_not_a_date():
    assert ymd("Academic Year 2026-27 and 2026-2027") == []


def test_day_month_does_not_cross_a_line_break():
    # "...2026-27<newline>October05,2026" once read the 27 as a day
    assert ymd("D(SW)/35+/2026-27\nOctober05,2026") == [(2026, 10, 5)]


# ============================ ranges and lists ============================
def test_range_with_shared_month_expands_to_every_day():
    assert ymd("Interviews on 16 to 18th November 2026") == [(2026, 11, d) for d in (16, 17, 18)]


def test_list_with_shared_month_expands():
    assert ymd("scheduled for 28th, 29th & 30th October 2026") == [(2026, 10, d) for d in (28, 29, 30)]
    assert ymd("*28^th, 29^th & 30^th October 2026*") == [(2026, 10, d) for d in (28, 29, 30)]


def test_dash_range_expands():
    assert ymd("5-8 Oct.2026") == [(2026, 10, d) for d in (5, 6, 7, 8)]


def test_full_date_range_keeps_both_ends():
    assert ymd("from 07-11-2026 to 14-11-2026") == [(2026, 11, 7), (2026, 11, 14)]


# ============================ typing: deadline / event / issued ============================
@pytest.mark.parametrize("text, field", [
    ("Last date for registration is 12/10/2026.", "deadline_dates"),
    ("The form must be submitted on or before 19/10/2026 without fail.", "deadline_dates"),
    ("Application Deadline: 15 July 2026, 7:00 PM", "deadline_dates"),
    ("Please apply by 06-11-2026 at the office.", "deadline_dates"),
    ("Registration is open till 30th June, 2026.", "deadline_dates"),
    ("The exam will be held on 12/10/2026.", "event_dates"),
    ("Orientation: Monday, July 27, 2026 at 10:00 AM", "event_dates"),
    ("Classes will commence from 27 July 2026.", "event_dates"),
    ("The Institute remains closed on 31/07/2026 due to rain.", "event_dates"),
])
def test_date_typing(text, field):
    items = find_dates(text)
    assert len(items) == 1 and items[0]["field"] == field


def test_issue_date_label_is_a_notice_date_not_an_event():
    items = find_dates("No. Hostel Section/96/2026-27\nDate:21-09-2026\nNOTICE: mess change from September 24, 2026")
    assert values(items, "notice_dates") == ["2026-09-21"]
    assert values(items, "event_dates") == ["2026-09-24"]


def test_dated_reference_is_an_issue_date():
    items = find_dates("Ref:-No.HostelSection/165/2023-24dated:-01/5/2023 With reference to the above, students may stay.")
    assert items == [] or all(i["field"] == "notice_dates" for i in items)


def test_bare_date_under_a_reference_number_is_the_issue_date():
    items = find_dates("No.: DoCSE/MidSem./Odd/119S/2026-2027\n08-09-2026\nMid-Sem Exam Timetable\n21-Sep-26 Database")
    assert values(items, "notice_dates") == ["2026-09-08"]
    assert "2026-09-21" in values(items, "event_dates")


def test_validity_condition_is_not_a_date_of_the_notice():
    assert find_dates("Income certificate issued on or after 1st April 2026 is required.") == []


def test_old_reference_dates_are_ignored():
    text = "Last date is 12/10/2026. Exemption as per the letter dated June 24, 2016. Held on 14/10/2026."
    assert {i["value"] for i in find_dates(text)} == {"2026-10-12", "2026-10-14"}


def test_range_end_shares_the_type_of_its_start():
    items = find_dates("Fee payment will be held from 07-11-2026 to 14-11-2026 in the office.")
    assert {i["field"] for i in items} == {"event_dates"} and len(items) == 2


def test_date_without_cue_is_flagged_not_guessed():
    items = find_dates("Rooms 12/10/2026 only.")
    assert items[0]["status"] == "needs_review" and items[0]["type"] == "unknown"


def test_year_missing_everywhere_is_flagged():
    items = find_dates("The test is on 11th Nov.")
    assert items[0]["value"] == "????-11-11" and items[0]["status"] == "needs_review"


# ================================ mails ================================
MAIL = ("Subject: Test\nDate: 02-11-2026, 15:53\n\n"
        "Please share their acceptance by tomorrow 10.00 am.\n\n"
        "On Thu, Oct 29, 2026 at 4:57 PM, [NAME] wrote:\n"
        "Interested students can apply through the MIS portal till Monday 10.00 am\n")


def test_mail_header_is_parsed():
    h = parse_mail_header(MAIL)
    assert h["subject"] == "Test" and h["sent_date"].isoformat() == "2026-11-02"
    assert parse_mail_header("Plain PDF text, no header")["sent_date"] is None


def test_relative_dates_use_the_send_date_of_their_own_message():
    items = find_dates(MAIL)
    assert values(items, "deadline_dates") == ["2026-11-02", "2026-11-03"]      # tomorrow (of 2 Nov); Monday after 29 Oct
    assert values(items, "notice_dates") == [] and values(items, "event_dates") == []


def test_send_dates_are_never_events():
    items = find_dates("Subject: Closure\nDate: 31-07-2026, 13:18\n\nThe Institute will remain closed today, *July 31, 2026, from 2:00 PM onwards*.")
    assert values(items) == ["2026-07-31"] and items[0]["field"] == "event_dates"
    assert "Date: 31-07-2026" not in items[0]["evidence"]


def test_forwarded_header_date_is_a_send_date_not_an_event():
    text = ("Subject: Fwd: X\nDate: 05-11-2026, 10:36\n\nFYI\n\n---------- Forwarded message ---------\n"
            "From: *CDC*\nDate: Thu, 5 Nov 2026, 10:21\nSubject: Re: X\n\nPls attend the test scheduled for *5^th November 2026*.")
    items = find_dates(text)
    assert values(items) == ["2026-11-05"]
    assert items[0]["raw"].startswith("5")            # found in the body sentence, not in the header lines


def test_date_label_inside_a_mail_body_is_an_event():
    text = "Subject: Camp\nDate: 04-08-2026, 14:25\n\nYou are invited to:\n*Date*: 15th August 2026, Saturday\n*Venue*: Staff Club"
    assert [(i["field"], i["value"]) for i in find_dates(text)] == [("event_dates", "2026-08-15")]


def test_year_is_assumed_from_the_message_and_flagged():
    text = "Subject: S\nDate: 05-11-2026, 13:32\n\nOnline Assessment: 11th Nov (Virtual)"
    item = find_dates(text)[0]
    assert item["value"] == "2026-11-11" and item["year_assumed"] and item["status"] == "needs_review"


def test_weekday_ranges_and_habits_are_not_dates():
    base = "Subject: S\nDate: 30-07-2026, 11:18\n\n"
    assert find_dates(base + "Coaching is available (monday to \nsaturday) for the following games.") == []
    assert find_dates(base + "The library is open every Monday.") == []
    assert find_dates(base + "Classes on Fridays are shorter.") == []


def test_weekday_next_to_an_explicit_date_is_not_counted_twice():
    items = find_dates("Subject: S\nDate: 01-10-2026, 10:00\n\nThe election is on Friday, 9 October 2026.")
    assert values(items) == ["2026-10-09"]


def test_today_and_day_after_tomorrow():
    base = "Subject: S\nDate: 10-11-2026, 09:00\n\n"
    assert values(find_dates(base + "Submit it by today.")) == ["2026-11-10"]
    assert values(find_dates(base + "The test is on the day after tomorrow.")) == ["2026-11-12"]


def test_relative_words_are_ignored_without_a_send_date():
    assert find_dates("Please submit it by tomorrow.") == []


# ================================== evidence ==================================
def test_evidence_is_the_sentence_with_correct_offsets():
    text = "Dear students. The last date for registration is 12/10/2026. Contact the office for help."
    item = find_dates(text)[0]
    assert item["evidence"] == "The last date for registration is 12/10/2026."
    assert text[item["start"]:item["end"]] == item["evidence"]
    assert item["start"] <= item["match_start"] < item["match_end"] <= item["end"]
    assert text[item["match_start"]:item["match_end"]] == "12/10/2026"


# ================================== fees ==================================
@pytest.mark.parametrize("text, amount", [
    ("The fee is Rs. 500.", 500),
    ("Pay Rs 500/- at the office.", 500),
    ("INR 1,200 per semester", 1200),
    ("Charges: ₹2500", 2500),
    ("Pay 500 rupees by Monday", 500),
    ("Rs.1,25,000 (Indian grouping)", 125000),
    ("Rs. 1500/-per student", 1500),
    ("to be printed on Rs.50/- Stamp Paper", 50),
    ("Rs. 100/- per day as a fine", 100),
])
def test_money_formats(text, amount):
    assert values(find_fees(text)) == [amount]


@pytest.mark.parametrize("text", [
    "Room 500 is closed.",                                   # a plain number is not money
    "CTC: INR 21 LPA for the role",
    "*Internship Stipend* INR 1,00,000 per month",
    "family income is less than Rs. 5.0 lakh",
    "INR 75,000 Joining Bonus",
    "Package of Rs. 12 lakh per annum",
    "Call 9876543210 for help",
])
def test_not_fees(text):
    assert find_fees(text) == []


def test_amounts_after_an_excluded_one_in_a_stipend_list_are_excluded_too():
    assert find_fees("Internship Stipend (per month): B.Tech - INR 50000 and MCA - INR 30000") == []


def test_fee_kind():
    assert find_fees("A late fee of Rs. 100 per day is charged.")[0]["kind"] == "fine"
    assert find_fees("The tuition fee is Rs. 5000.")[0]["kind"] == "fee"
    assert find_fees("The deposit of Rs. 2000 is refundable and will be refunded.")[0]["kind"] == "refund"


# ================================ documents ================================
def test_documents_with_a_trigger_in_the_sentence():
    assert values(find_documents("Candidates should bring their resume, marksheet and ID card.")) == \
        ["id card", "marksheet", "resume"]


def test_bullet_list_after_a_documents_heading():
    text = "Documents required:\n1. Aadhaar card\n2. Fee receipt\n3. Passport size photograph"
    assert values(find_documents(text)) == ["aadhaar", "fee receipt", "photograph"]


def test_gazetteer_words_without_a_trigger_are_not_documents():
    assert find_documents("The speaker published a resume of the talk on the website.") == []


def test_canonical_documents_normalises_label_text():
    assert canonical_documents("income certificate issued on or after 1 April 2026") == {"income certificate"}
    assert canonical_documents("bank passbook copy or cancelled cheque") == {"bank details"}
    assert canonical_documents("ITR form") == {"income tax return", "form"}
    assert canonical_documents("something unlisted") == {"something unlisted"}


# ================================ eligibility ================================
@pytest.mark.parametrize("text, expected", [
    ("Minimum CGPA of 7.5 is required.", {"cgpa>=7.5"}),
    ("CGPA 7 and above", {"cgpa>=7"}),
    ("CGPA >= 7.0 (B.Tech, M.Tech)", {"cgpa>=7"}),
    ("a minimum 7.0 CGPA throughout", {"cgpa>=7"}),
    ("CGPA below 6", {"cgpa<6"}),
    ("Minimum 60% in 10th and 12th", {"percent>=60"}),
    ("Minimum 80% equivalent CGPA in 10th, 12th & B.Tech", {"percent>=80"}),
    ("No active backlogs", {"backlogs=0"}),
    ("students without any backlog", {"backlogs=0"}),
    ("CGPA of less than 6.5", {"cgpa<6.5"}),
    ("2027 pass-out", {"batch=2027"}),
    ("Eligible Batch: 2027 Pass-out", {"batch=2027"}),
    ("The Incoming MBA Batch of 2026-2028", {"batch=2026-2028"}),
    ("A cumulative academic gap of up to 2 years is permitted", {"gap<=2"}),
    ("no year gaps", {"gap=0"}),
    ("22% increase in applications", set()),                  # a percentage with no eligibility context
    ("score of 9.5 points", set()),
])
def test_eligibility_conditions(text, expected):
    assert canonical_conditions(text) == expected


def test_eligibility_items_carry_structure_and_evidence():
    text = "Interested students. B.Tech students with minimum CGPA of 7.0 may apply."
    item = find_eligibility(text)[0]
    assert item["value"] == "cgpa>=7" and item["condition"] == {"type": "cgpa", "op": ">=", "value": 7.0}
    assert "CGPA of 7.0" in item["evidence"] and item["status"] == "ok"


# ================================== actions ==================================
def test_actions_need_a_verb_and_a_modal_or_imperative():
    assert [a["verbs"] for a in find_actions("Interested students should register on the portal.")] == [["register"]]
    assert [a["verbs"] for a in find_actions("Submit the form by 12/10/2026.")] == [["submit"]]
    assert [a["verbs"] for a in find_actions("Dear CM. pls inform the students.")] == [["inform"]]
    assert find_actions("The exam was conducted last week.") == []
    assert find_actions("Students register every year.") == []          # verb but no instruction


def test_action_confidence_rises_with_portal_or_deadline():
    plain = find_actions("Students should register soon.")[0]["confidence"]
    rich = find_actions("Students should register on the MIS portal by 12/10/2026.")[0]["confidence"]
    assert rich > plain


def test_action_verbs_helper_matches_forms():
    assert action_verbs("registered and paid; will attend") == {"register", "pay", "attend"}


# ============================== analyze_notice ==============================
NOTICE = ("Notice\nThe hostel residents who want to stay during the Diwali break (07-11-2026 to 14-11-2026) "
          "should submit the filled form (Annexure-I) latest by 06-11-2026, upto 6:00 PM. "
          "A fine of Rs. 100 per day applies after that. Students need a minimum CGPA of 6.0.")


def test_analyze_notice_returns_every_field_with_evidence_and_status():
    r = analyze_notice(NOTICE)
    f = r["fields"]
    assert values(f["deadline_dates"]) == ["2026-11-06"]
    assert values(f["event_dates"]) == ["2026-11-07", "2026-11-14"]
    assert values(f["fees"]) == [100]
    assert values(f["eligibility"]) == ["cgpa>=6"]
    assert "form" in values(f["required_documents"])
    assert any("submit" in a["verbs"] for a in f["action_items"])
    for items in f.values():
        for i in items:
            assert i["evidence"] and i["status"] in {"ok", "needs_review"} and 0 <= i["confidence"] <= 1
    assert r["n_items"] == sum(len(v) for v in f.values())
    assert r["category"] is None                       # no classifier given


def test_two_different_deadlines_are_flagged_for_review():
    r = analyze_notice("Apply by 12/10/2026. The last date for fee payment is 19/10/2026.")
    deadlines = r["fields"]["deadline_dates"]
    assert len(deadlines) == 2 and all(d["status"] == "needs_review" and d["conflict"] for d in deadlines)


def test_one_deadline_is_not_flagged():
    d = analyze_notice("The last date for registration is 12/10/2026.")["fields"]["deadline_dates"][0]
    assert d["status"] == "ok" and not d.get("conflict")


def test_analyze_notice_with_a_classifier_and_a_mail():
    nb = NaiveBayesClassifier(alpha=0.5).fit([["hostel", "mess", "warden"], ["exam", "timetable", "marks"]],
                                              ["hostel", "exam"])
    r = analyze_notice("Subject: Mess change\nDate: 21-09-2026, 10:00\n\nHostel mess change facility by the warden. "
                       "Apply by tomorrow.", classifier=nb)
    assert r["category"] == "hostel" and r["category_status"] == "ok"
    assert r["sent_date"] == "2026-09-21" and r["subject"] == "Mess change"
    assert values(r["fields"]["deadline_dates"]) == ["2026-09-22"]


def test_empty_text_does_not_crash():
    r = analyze_notice("")
    assert r["n_items"] == 0 and r["notice_dates"] == []


# ---- hard cases found on Day 6 ----
def test_annual_fee_is_a_fee_but_annual_income_is_not():
    assert [f["value"] for f in find_fees("The annual fee is Rs. 150000.")] == [150000]
    assert find_fees("Annual family income should be below Rs. 800000.") == []


def test_cgpa_and_above_phrasing():
    assert canonical_conditions("Students with 6.5 and above CGPA may apply.") == {"cgpa>=6.5"}
    assert canonical_conditions("Students with 7 or above CGPA may apply.") == {"cgpa>=7"}


def test_extended_deadline_sentence_is_flagged_for_review():
    res = analyze_notice("The deadline is extended from 10/11/2026 to 15/11/2026.")
    items = res["fields"]["deadline_dates"] + res["fields"]["event_dates"]
    assert len(items) == 2
    assert all(i["status"] == "needs_review" for i in items)


def test_plain_deadline_is_not_flagged():
    res = analyze_notice("Last date to apply is 15/11/2026.")
    assert res["fields"]["deadline_dates"][0]["status"] == "ok"
