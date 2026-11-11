# Known limitations

What CampusNotice AI does not do well, and what we saw when we tried. Numbers are from `src/eval_*.py`; the test split was used once on 11 Nov 2026 (`src/final_eval.py`, `data/final_results.json`).

## 1. Data and evaluation
- **Tiny corpus:** 39 notices (about 33 independent ones after grouping near-duplicates); `hostel` has 4 examples. One notice is about 3 accuracy points.
- **One annotator.** The answer key (`data/labels.json`) was reviewed by the user for categories, dates and fees (11 Nov); eligibility, documents and action text were not checked line by line. Five placement labels were completed *after* looking at extractor errors, so their scores are optimistic.
- **Test result (6 notices, run once):** category 6/6 (baseline 0.50; with n=6 trust the cross-validation 0.79 instead); extraction micro-F1 0.69, deadlines F1 **0.44** (dev set 1.00: recall 0.33), events 0.79, fees 1.00, eligibility 1.00 (2-3 items each), documents undefined (no true items), actions 0.53. Needs-review flagged 6 of 8 wrong values and 9 of 18 correct ones.
- **Extraction scores on dev are optimistic**, as the test gap shows. Rules were written while reading the train and dev notices, and the test notices were read during labeling. Dev-set results: deadlines/events/fees/eligibility F1 0.98–1.00, documents 0.88, actions 0.45.
- **Search judgments are drafts** written by one person who knew the corpus (15 queries). TF-IDF P@3 0.667 (ceiling 0.733), MRR 0.967; differences of a few points are noise.
- **Classifier:** CV accuracy 0.788, macro-F1 0.830 on 33 notices. `exam` is the weak class (calendars and timetables look like other notices to a word-count model).
- **Synthetic data:** none in the corpus. The hard cases below are made-up sentences used only for probing; they are not part of any score.

## 2. Input problems
- **Scanned PDFs** need OCR (20 of 21 here). OCR drops spaces ("Lastdateofsubmission"), misreads digits (`Rs.80oo00`) and mixes scripts. Search misses OCR-glued notices; glued dates are found but flagged.
- **Tables** (calendars, timetables, committee lists, branch tables) have no sentence structure; extraction is not scored on them.
- **Language:** English only. Hindi-only text yields at most a date, flagged for review.
- **Personal data:** names, roll numbers, e-mails and phones are masked on a best-effort basis. Staff names in signature blocks can remain.

## 3. Extraction: what we tried and what happened
These are made-up probes, run through `analyze_notice` on 11 Nov. "Fixed" means a bug found by this exercise and fixed with a test.

| # | Input (short) | Result | Verdict |
|---|---|---|---|
| H01 | "submit by tomorrow 5 pm" with mail header dated 10-11-2026 | deadline 2026-11-11 | works |
| H02 | same, no header | nothing found for the date; **no warning** | limitation: relative word with no reference date is silently dropped |
| H03 | two different deadlines | both kept, both marked needs-review | works (by design) |
| H04 | 31/02/2026 | nothing found, no warning | invalid dates are rejected silently |
| H05 | 11/20/2026 (month first) | nothing found, no warning | we only read day-first; this is a choice, not a bug |
| H06 | "Lastdateofsubmission:15thNovember2026" (OCR-glued) | date found as an *event*, needs-review; fee found | partial: wrong type, but flagged |
| H07 | "Annual fee is Rs. 1,50,000" | was missed (the word "annual" was on the salary/income exclusion list) | **fixed** |
| H07b | "deposit is Rs 2.5 lakh" | not extracted | limitation: amounts in lakh are excluded on purpose (they are usually salary) |
| H08 | "6.5 and above CGPA" | was missed | **fixed** |
| H09 | "active backlogs are not eligible" | `backlogs=0` | works |
| H10 | "deadline is **extended from** 10/11 **to** 15/11" | both dates become *events*; was status OK, now both are needs-review (sentences with extended / postponed / rescheduled / preponed / deferred are capped below the threshold) | **partly fixed**: still wrong type and the old date is not retired, but a human is now asked. Cost: 2 correct dev-set dates in real extension notices are flagged too |
| H11 | "Register by Nov 18 2026" | deadline 2026-11-18 | works |
| H12 | Hindi sentence with 15/11/2026 | date found, needs-review | works (flagged) |
| H13 | empty text | nothing, no crash | works |
| H14 | text with nothing to extract | nothing | works |
| H15 | "exams from 16/11/2026 to 21/11/2026" | event dates on both ends | works (but a range is stored as its end dates, not every day, when written as two full dates) |
| H16 | "by 18 November" (no year) | `????-11-18`, needs-review | works (flagged; not placed on the dashboard calendar) |
| H17 | "at least 60% marks" | `percent>=60` | works |
| H18 | "Registration fee: 500 INR. Late fine 50 per day." | nothing | limitation: an amount is only a fee if it carries Rs / INR + number, `/-`, or "rupees"; "500 INR" (number before the unit) and bare "50 per day" are missed |

**The worst remaining finding is the silent misses (H02, H04, H05, H18), plus H10's wrong type:** when the extractor finds nothing, the app shows nothing; it does not say "I may have missed something". "Needs review" only covers values we *did* extract.

## 4. Extraction: structural limits
- **Action items** are crude: every instruction sentence in a long mail thread is reported. Scored by comparing action verbs, F1 0.45.
- **Documents** come from a gazetteer; unlisted names are missed and "Google Form" can be mistaken for a required form.
- **Eligibility** covers CGPA, percent, backlogs, batch and study gap. Branch, programme, semester and category rules are **not** extracted.
- **Mail threads:** old quoted messages are extracted too (a design choice), so a thread can show many dates; the conflict rule then marks multiple deadlines for review.
- **"Needs review" catches few wrong values:** on the dev set it flagged 2 of 5 wrong extractions (40%) while 39% of the correct ones were also flagged.
- **Deadline vs event typing** is by nearby keywords and can be wrong when the wording is unusual.

## 5. App
- Checked with Streamlit's test harness and an HTTP check; visual layout was not reviewed in a browser before this note.
- Large scanned PDFs are slow (OCR runs in the request).
- The CGPA filter only knows the 4 of 39 notices that state a CGPA rule; it keeps a notice if you meet its *lowest* threshold.
- Search runs over the bundled corpus only; uploaded notices are analysed but not added to the index.
