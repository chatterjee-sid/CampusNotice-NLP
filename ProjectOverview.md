# CampusNotice-NLP – Project Overview

---

## What is this project?

CampusNotice-NLP is a tool that reads college notices (PDFs or emails) and automatically
pulls out the information students care about most: deadlines, fees, eligibility rules,
required documents, and action items. It also classifies each notice into a broad category
(exam, placement, hostel, or general) so you can filter them quickly. A simple web UI lets
you upload a notice, search the full corpus, or browse all upcoming deadlines on a calendar
view. The whole pipeline is built from scratch — no large language models or external APIs
are used, only regular expressions, a hand-crafted Naive Bayes classifier, and TF-IDF
search, all exactly as laid out in the roadmap.

---

## How the pieces fit together

```
Raw notice
  │
  ├─ PDF  ──► pdf_reader.py  ──► plain text
  └─ Mail ──► email_reader.py ──► plain text (PII scrubbed)
                     │
                     ▼
              preprocess.py   (clean, tokenize, optional stop-word removal / stemming)
                     │
            ┌────────┴─────────────────────────────────────────┐
            ▼                                                   ▼
     classifier.py                                       extractor.py
  (Naive Bayes → category)                (regex rules → dates / fees / eligibility /
                                           documents / actions, each with evidence)
            │                                                   │
            └────────────────┬──────────────────────────────────┘
                             ▼
                       evidence.py      (maps every extracted value back to its
                                         source sentence and assigns a confidence score)
                             │
                             ▼
                         index.py       (TF-IDF inverted index for Search)
                             │
                             ▼
                          app.py        (Streamlit web UI: Analyse / Search / Deadlines)

Offline preparation (run once):
  build_corpus.py  →  data/texts/
  make_split.py    →  data/split.json
  eval_*.py / final_eval.py  →  evaluation reports + charts
```

---

## File-by-file explanation

---

### `requirements.txt`

Lists the four external libraries the project uses: **PyMuPDF** (read PDFs), **RapidOCR**
(OCR scanned pages), **Streamlit** (the web UI), **pandas / numpy / matplotlib** (data
handling and charts), **pytest** (tests), and **scikit-learn** (used only inside the test
that cross-checks our custom metrics against sklearn's reference numbers). Every other
piece of the system — the classifier, the search engine, the extractor, the metrics — is
written from scratch as the roadmap required.

---

### `src/preprocess.py`

The first step every notice goes through before anything else. It does four things:

1. **Cleaning** – converts the text to lowercase and collapses extra whitespace.
2. **Tokenisation** – splits text into *tokens* (words or numbers). For example,
   `"Last date: 12/10/2026"` → `["last", "date", "12/10/2026"]`. The tokeniser uses a
   regex that keeps dates, times, percentages, and decimals in one piece instead of
   splitting them at the slash or dot.
3. **Stop-word removal** – drops common words like "the", "is", "and" that carry no
   topic information. Words containing digits (dates, CGPAs) are always kept.
4. **Stemming** – strips common suffixes so "registered", "registering", and
   "registration" all become "register". This is a custom two-pass stripper, not the
   standard Porter stemmer.
5. **Shape features** – counts how many dates and numbers are in the text and returns
   two special tokens like `shape_dates_many` and `shape_nums_very_high`. The classifier
   uses these to tell timetables (lots of numbers) from general notices.
6. **Sentence splitting** – the `split_sentences` function breaks text at `.`, `?`, `!`,
   or bullet markers, while carefully skipping abbreviations like "Dr.", "Rs.", "M.Sc.".
   Every sentence comes with its start and end character position so the extractor can
   point back to exactly the right sentence.

This file is imported by almost everything else, so keep its interface stable.

---

### `src/email_reader.py`

Converts `.eml` or plain-text mail files into the same format as a PDF notice so the rest
of the pipeline does not need to know the difference. For `.eml` files it uses Python's
built-in `email` library to extract the subject and body. For `.txt` files it strips
header lines (From:, To:, CC:, etc.) and keeps only Subject and Date.

The `scrub_pii` function then removes personal details: student names, roll numbers,
email addresses, phone numbers, and "Regards / Name" sign-offs. The goal is that nothing
visible in the UI can identify individual students. It also replaces quoted-reply headers
("On Thu … wrote:") with an anonymised version while keeping the timestamp — the
extractor needs that timestamp to resolve relative words like "tomorrow" and "Monday".

---

### `src/pdf_reader.py`

Reads PDF files and returns plain text. It uses **PyMuPDF** to extract the text layer
page by page. If a page has no text layer (a scanned image), it falls back to **RapidOCR**
to perform optical character recognition at 200 DPI. The function returns a flag saying
whether OCR was used, so the UI can warn the user that digits and spaces may be misread.
`process_all_pdfs()` is a helper used by `build_corpus.py` to bulk-convert all PDFs in
`data/raw_pdfs/` and save them as `.txt` files in `data/texts/`.

---

### `src/build_corpus.py`

A one-time setup script that gathers all raw notices into the right folder layout. It
copies PDFs from several source folders into `data/raw_pdfs/` (skipping byte-identical
duplicates), then converts every PDF and mail file to a cleaned `.txt` file in
`data/texts/`. Near-duplicate mail threads (defined in `DUP_GROUPS`) are deduplicated by
keeping only the longest copy. The roadmap called for a "corpus builder"; this is exactly
that. Run it once before anything else; re-running it is safe.

---

### `src/make_split.py`

Creates `data/split.json`, which divides the labelled notices into **train / dev / test**
at a 60 / 20 / 20 ratio. The split is *stratified* (each category keeps roughly the same
proportion in every part) and *group-aware* (near-duplicate notices that share a `group`
field in `labels.json` are always kept on the same side, so a model never trains on one
copy of a notice and tests on another). Once `split.json` exists the script refuses to
overwrite it — the split must never change after evaluation has begun, to avoid "peeking"
at the test set.

---

### `src/evidence.py`

A short but critical helper used by the extractor. It has two functions:

- `map_evidence(span, sentences)` — given the character position of an extracted value,
  finds which sentence from `preprocess.split_sentences` contains it and returns that
  sentence's text and offsets. This is how every extracted item gets its "evidence
  sentence".
- `compute_confidence(extraction)` — assigns a confidence score between 0 and 1 based on
  how close the extracted value was to a strong keyword (e.g., "deadline", "last date"),
  whether two conflicting types were found in the same sentence, and whether an evidence
  sentence exists at all. Scores below 0.6 trigger a `needs_review` badge in the UI.

---

### `src/extractor.py`

The largest and most complex file. It is a pure rule-based and regex information
extractor; there is no machine learning here. Given the plain text of one notice it
returns every:

- **Deadline / event / issue date** — five regex patterns cover numeric formats
  (12/10/2026), ordinal formats (12th October 2026), ISO format, month-name-first, and
  sequential ranges ("16 to 18th November"). A `classify_date` function then looks at
  the words around each date (e.g., "last date", "deadline", "scheduled", "held on") to
  decide whether the date is a deadline, an event, or the notice's own issue date. For
  mails it also resolves relative words like "tomorrow" and "Monday" using the send
  timestamp from the email header.
- **Fees** — matches currency patterns (Rs., INR, ₹, rupees) and explicitly excludes
  salaries, stipends, and CTC amounts that look similar.
- **Eligibility conditions** — extracts CGPA thresholds (e.g., `cgpa>=7.5`), percentage
  requirements, backlog rules, graduation batch, and academic gap limits, all in a
  normalised canonical form so they can be compared with hand-written labels.
- **Required documents** — a fixed *gazetteer* (lookup list) of 20 common document
  names (resume, marksheet, Aadhaar, etc.) matched against sentences that contain a
  trigger word like "submit", "bring", or "attach".
- **Action items** — sentences that contain a known action verb ("register", "submit",
  "pay", …) together with a modal word ("should", "must", "please", "kindly") or start
  with an imperative are tagged as things the student must do.

The top-level function `analyze_notice(text, classifier=None)` calls all five extractors
and also runs the classifier if one is provided. It detects conflicting deadlines (two
different dates both typed as deadlines) and flags them all for human review.

---

### `src/classifier.py`

A complete **Multinomial Naive Bayes** classifier written from scratch. Naive Bayes is a
simple probability model: it looks at how often each word appears in each category's
training documents and uses those counts to guess which category a new document belongs
to. For example, a notice with many occurrences of "mess", "hostel", and "warden" will
score high for the "hostel" category.

Key details:
- **Laplace (add-alpha) smoothing** prevents the probability of an unseen word from
  becoming zero, which would ruin the whole product. Alpha is set to 0.5 (found by
  cross-validation).
- Scores are computed as **sums of logarithms** to avoid underflow (multiplying hundreds
  of tiny probabilities together would round down to zero in a computer).
- `predict_proba` converts log-scores to real probabilities using a softmax. The
  confidence shown in the UI is this probability for the winning category.
- `save` and `load` let the model be stored as a JSON file (although in practice the app
  retrains it at startup from the training split).
- `top_words` shows which words most distinguish one category from the others, useful for
  debugging.

---

### `src/index.py`

Implements a TF-IDF search engine from scratch.

**TF-IDF** (Term Frequency – Inverse Document Frequency) is a way to score how important
a word is to a document. A word that appears often in one notice but rarely across the
corpus scores high for that notice — this is more informative than raw word counts.

The `InvertedIndex` class stores, for each word, which documents contain it and how many
times (this is the "posting list"). At query time it only scores documents that share at
least one word with the query — the index makes this fast. Three search methods are
supported: TF-IDF cosine similarity (best), raw count sum, and boolean AND (all query
words must appear). The `build_index` function reads all notices from `data/texts/` and
indexes them. The app calls this once at startup and caches the result.

---

### `src/storage.py`

A tiny utility that saves and loads analysis results as a JSON file
(`data/results.json`). Each entry maps a document ID to the dict returned by
`analyze_notice`. This was intended as a cache layer. In the current app, `analyze_notice`
is called live and results are not persisted between sessions, so `storage.py` exists but
is not imported by `app.py` — it could be used to speed up future cold starts.

---

### `src/ui_helpers.py`

Pure Python functions that support the Streamlit UI but do not import Streamlit, so they
can be tested easily. Key functions:

- `evidence_html` — given an extracted item and the original text, returns an HTML
  snippet that highlights the exact matched span with a `<mark>` tag.
- `cgpa_requirement` / `passes_cgpa` — read the eligibility fields of a notice and check
  whether a student's CGPA meets the requirement. Used by the Search page to filter
  notices.
- `upcoming_deadlines` — scans all notices and returns deadlines that are on or after
  today, sorted soonest-first. Deadlines without a year (stored as `????-MM-DD`) go to a
  separate "unknown year" list.
- `best_sentence` — finds the sentence in a notice that shares the most words with the
  search query. Used for search-result snippets.
- `display_title` — cleans up a filename like `mail_20260804-Blood donation camp-4065.txt`
  into a readable title for the UI.

---

### `src/eval_classifier.py`

A command-line script for evaluating the Naive Bayes classifier during development. It
supports two modes: `dev` (train on the train split, evaluate on dev) and `cv`
(leave-one-unit-out cross-validation over train + dev). It never touches the test split.
Run it to compare configurations: `python src/eval_classifier.py cv --alpha 0.5 --stop`.
It prints a classification report (precision / recall / F1 per category, plus a confusion
matrix) and lists every misclassified notice.

---

### `src/eval_extraction.py`

A command-line script for evaluating the information extractor. It compares the
extractor's output against the hand-written labels in `data/labels.json`, field by field,
and reports precision / recall / F1. Eligibility and document values are normalised before
comparison (the same parser is applied to both the label text and the prediction, so
wording differences do not count as errors). It also reports how many wrong extractions
are caught by the `needs_review` flag versus how many slip through silently. It operates
only on the train and dev subsets; the test split is off-limits.

---

### `src/eval_search.py`

A command-line script for evaluating the search engine. It reads hand-written relevance
judgements from `data/search_queries.json` and measures **P@3** (how many of the top 3
results are relevant), **R@3**, and **MRR** (mean reciprocal rank — on average, how high
up is the first relevant result?). Unlike classification and extraction, search has no
training phase, so it is evaluated over the whole corpus rather than a split. All three
methods (TF-IDF, count, boolean) are compared side by side.

---

### `src/final_eval.py`

The one-time, locked-down final evaluation on the **test split**. It runs once, saves the
results to `data/final_results.json`, and then refuses to run again (to prevent the test
set from becoming a tuning set). It trains the classifier on train + dev, evaluates on
test, and reports classification accuracy, extraction F1 per field, and search MRR.
It also draws three charts (search MRR bar chart, extraction F1 per field, confusion
matrix) and saves them to the `results/` folder. The settings used (alpha = 0.5, stop
words on, no stemming) are frozen in the `FROZEN` constant and must not be changed
after the first run.

> **Deviates from roadmap:** The roadmap described three separate evaluation phases with
> growing test exposure. The code implements a strict one-time gate: `final_eval.py`
> exits immediately if the results file already exists, which is a stronger guard than
> the roadmap specified.

---

### `src/metrics.py`

All evaluation metrics, written from scratch (the roadmap required this). Three groups:

- **Search**: `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`,
  `evaluate_search`.
- **Classification**: `evaluate_classification` returns accuracy, per-class precision /
  recall / F1, macro averages, and a confusion matrix. The test `test_matches_scikit_learn`
  cross-checks these numbers against sklearn to prove correctness.
- **Extraction**: `evaluate_extraction` compares predicted and labelled field values as
  sets, reporting TP / FP / FN and P / R / F1 per field. `review_coverage` measures how
  well the `needs_review` flag catches wrong predictions versus generating false alarms on
  correct ones.

Undefined ratios (0 / 0) are returned as `None`, not silently as 0 or 1.

---

### `app.py`

The entry point for the web UI. Run it with `streamlit run app.py` from the
`CampusNotice-NLP/` directory. It has three pages:

- **Analyse a notice** — upload a PDF or mail file, paste text, or pick from the corpus.
  The extractor and classifier run on the input and display the results: category,
  confidence, a table of all extracted fields, and collapsible "evidence" sections that
  show the exact sentence each value came from with the matched span highlighted.
- **Search** — a text box, with optional category and CGPA filters. TF-IDF cosine search
  is the default; count and boolean are also available. For each result, the best matching
  sentence is shown as a snippet.
- **Deadlines** — shows all upcoming deadlines across the whole corpus, soonest first,
  with the evidence sentence for each. Past deadlines are hidden.

The classifier and index are loaded with `@st.cache_resource` so they are trained/built
only once per server session rather than on every page interaction.

---

### `tests/conftest.py`

A single pytest setup file that adds `src/` to Python's module search path. This means
test files can write `from classifier import ...` instead of a longer path, and there is
no need to install the project as a package.

---

### `tests/test_preprocess.py`

Unit tests for `preprocess.py`. Verifies that dates, times, and percentages are kept
whole during tokenisation; that the custom stemmer handles plurals, doubled consonants,
and special tokens (dates, short words) correctly; that sentence splitting avoids breaking
on abbreviations; and that the shape-feature buckets fire at the right thresholds.

---

### `tests/test_classifier.py`

Unit tests for `classifier.py`. Includes a hand-computed example (worked out in the
docstring with actual numbers) that proves `predict_scores` matches the formula. Also
verifies Laplace smoothing, underflow prevention, deterministic tie-breaking, and
save/load round-trip correctness.

---

### `tests/test_extractor.py`

The largest test file. Covers every extractor:

- **Dates**: all written formats, day-first parsing, impossible dates, range expansion,
  relative words ("tomorrow", "Monday"), mail header handling, forwarded-thread send dates,
  issue-date vs event-date vs deadline classification.
- **Fees**: all currency formats and negative cases (salaries, stipends, plain numbers).
- **Documents**: trigger-word gating, bullet lists after a heading, gazetteer
  normalisation.
- **Eligibility**: all condition types (CGPA, percentage, backlogs, batch, gap).
- **Action items**: verb + modal requirement.
- **`analyze_notice`**: integration test; also tests conflicting-deadline flagging.

---

### `tests/test_index.py`

Unit tests for `index.py`. Includes hand-calculated TF-IDF values (worked out in the
docstring) and verifies they match the code. Also tests the three search methods, the
`allowed` document filter, the case-insensitivity of search, and save/load round-trips.

---

### `tests/test_metrics.py`

Unit tests for `metrics.py`. Each test includes the hand calculation in its docstring so
you can check the expected number yourself. Covers all three metric groups, the
scikit-learn cross-check, edge cases (no predictions, undefined ratios, duplicate values),
and text-report formatting.

---

### `tests/test_ui_helpers.py`

Unit tests for `ui_helpers.py`. Covers HTML escaping and `<mark>` insertion in
`evidence_html`, CGPA filtering logic, the upcoming-deadlines calendar view (past
deadlines hidden, year-less dates separated), `best_sentence` snippet selection, and
`display_title` formatting.

---

## How to run it

**Install dependencies** (from the `CampusNotice-NLP/` directory):

```bash
pip install -r requirements.txt
```

**Prepare the corpus** (one-time, requires raw notices in `data/`):

```bash
python src/build_corpus.py   # converts PDFs + mails → data/texts/
python src/make_split.py     # creates data/split.json (run only once)
```

**Launch the web UI:**

```bash
streamlit run app.py
```

**Run the tests:**

```bash
pytest
```

**Evaluate during development** (never touches the test split):

```bash
python src/eval_classifier.py cv          # cross-validated classifier performance
python src/eval_extraction.py devset      # extractor on train + dev
python src/eval_search.py                 # search comparison (all 3 methods)
```

**Final evaluation** (run only once, locks results to `data/final_results.json`):

```bash
python src/final_eval.py          # writes results + draws charts in results/
python src/final_eval.py --charts # redraws charts from saved json (safe to re-run)
```
