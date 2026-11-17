# CampusNotice AI

An explainable NLP system for college notices (SVNIT Surat placement-cell and institute notices). It reads a PDF (scanned pages go through OCR) or a mail, and then:

1. **classifies** it (`placement`, `exam`, `hostel`, `general`) with a Naive Bayes model written from scratch,
2. **extracts** deadlines, event dates, fees, eligibility rules, required documents and action items with rules and regexes,
3. shows, for every extracted value, the **evidence sentence** it came from and a **confidence**; values below 0.6 are marked **Needs review**,
4. lets you **search** all notices (TF-IDF over an inverted index we built) with category and CGPA filters, and lists upcoming **deadlines**.

Course project for CS459 (NLP), solo. **No LLM APIs.** The NLP core (tokenizer, inverted index, TF-IDF, Naive Bayes, date/fee/eligibility extraction, metrics) is our own code; scikit-learn is used only to cross-check it.

## Results (honest summary)
Dataset: 39 notices (21 PDFs, 18 mails), 4 categories, one annotator. Fixed split train 27 / dev 6 / test 6 (near-duplicates stay together). The test split was used **once**.

| What | Result |
|---|---|
| Category, cross-validation on 33 notices | accuracy 0.79, macro-F1 0.83 (always-`general` baseline 0.45) |
| Category, 6 test notices | 6 of 6 (baseline 0.50): a small-sample result; trust the cross-validation figure |
| Search, 15 queries (P@3 / MRR) | boolean 0.22 / 0.63, raw count 0.51 / 0.83, **TF-IDF 0.67 / 0.97** (P@3 ceiling 0.73) |
| Extraction on test (micro P / R / F1) | 0.61 / 0.79 / 0.69; deadlines F1 only **0.44** (dev set 1.00 was optimistic); fees and eligibility 1.00 on 2 and 3 items |
| Needs-review on test | flagged 6 of 8 wrong values, and 9 of 18 correct ones |

Charts: `results/`. Raw numbers: `data/final_results.json`.

## Setup
Python 3.12.
```
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
From this folder (`campusnotice/`):
```
streamlit run app.py              # the app: Analyse a notice | Search | Deadlines
python -m pytest tests/           # 205 tests
python src/eval_classifier.py dev|cv       # classifier reports (never the test split)
python src/eval_extraction.py dev|train|devset [--errors]
python src/eval_search.py [--detail]       # 15-query comparison of the 3 search methods
python src/final_eval.py --charts          # redraw charts from data/final_results.json
```
`python src/final_eval.py` (without `--charts`) is the one-time test evaluation; it refuses to run again once `data/final_results.json` exists.

### Using the app
- **Analyse a notice:** upload a PDF / `.txt` / `.eml`, paste text, or pick a corpus notice. Mails and pasted text are PII-masked first. Each value shows confidence, an OK / Needs-review badge, and its evidence sentence with the exact match highlighted.
- **Search:** pick a method (TF-IDF is the default and the best in our evaluation), filter by category, optionally by your CGPA.
- **Deadlines:** upcoming deadlines across all notices, soonest first; change "Today" to explore.

## Data and privacy
The notices contain real students' names, roll numbers, e-mails and phone numbers. These are masked (`[STUDENT]`, `[ROLL]`, `[EMAIL]`, `[PHONE]`, `[NAME]`) before anything is stored in `data/texts/`. The original mail folders (`data/cdc_mails/`, `data/other_mails/`, `data/newfound/`), `data/raw_pdfs/` and `data/texts/` are **not in the repository** (see `.gitignore`).
Without `data/texts/` the app and the evaluation scripts cannot run; `python src/build_corpus.py` rebuilds it from the source folders (about 5 minutes because of OCR). `tests/` use their own small inputs and run without the corpus.

## Layout
```
app.py                  Streamlit app
src/
  pdf_reader.py         PDF -> text, OCR fallback per page (PyMuPDF + RapidOCR)
  email_reader.py       mail -> text, personal-data masking
  preprocess.py         tokenizer (keeps 12.10.26, 7.5%, 10:30), stop words, stemmer, sentence splitter
  index.py              inverted index, TF-IDF cosine / raw count / boolean search
  classifier.py         multinomial Naive Bayes (Laplace smoothing, log probabilities)
  extractor.py          dates, fees, eligibility, documents, actions, analyze_notice
  evidence.py           evidence sentence mapping + confidence (threshold 0.6)
  metrics.py            P@k, MRR, accuracy/P/R/F1, confusion matrix, extraction metrics
  eval_*.py, final_eval.py   evaluation scripts
  ui_helpers.py         app logic without Streamlit (tested)
tests/                  205 tests
data/                   labels.json, split.json, search_queries.json, ANNOTATION_RULES.md, final_results.json
results/                charts from the final evaluation
KNOWN_LIMITATIONS.md    what does not work
```
Fuller documentation (design decisions, formulas, glossary, troubleshooting) is in `HOW_IT_WORKS.md` in the parent folder.
