# CampusNotice AI

An explainable NLP system for campus notice analysis, classification, information extraction, and search.

## Folder Structure

```
campusnotice/
├── data/
│   ├── raw_pdfs/          # Collected raw PDF notices
│   ├── texts/             # Extracted plain text files
│   └── labels.json        # Manual annotations & ground truth
├── src/
│   ├── pdf_reader.py      # PyMuPDF PDF text extraction module
│   ├── preprocess.py      # Cleaning, tokenization & sentence splitting module
│   ├── index.py           # Inverted index, TF-IDF, and search engine module
│   ├── classifier.py      # Naive Bayes text classification module
│   ├── extractor.py       # Rule-based & regex extraction (dates, fees, eligibility, docs, actions)
│   ├── evidence.py        # Sentence mapping and confidence scoring module
│   ├── metrics.py         # Evaluation metrics (Precision, Recall, F1, MRR, etc.)
│   └── storage.py         # Persistence storage handler (JSON / SQLite)
├── tests/                 # Unit and integration tests
├── app.py                 # Streamlit web application
└── README.md
```

## Setup & Running

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Dependencies**:
   ```bash
   pip install pymupdf streamlit pytest pandas matplotlib scikit-learn
   ```

3. **Launch Web App**:
   ```bash
   streamlit run app.py
   ```
