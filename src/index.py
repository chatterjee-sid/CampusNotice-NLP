"""
index.py

Inverted index construction, TF-IDF calculation, and search module.

Formulas (all natural log):
    idf(t)        = ln(N / df(t))           0 if the term is unseen
    tf_weight     = 1 + ln(tf)              0 if tf = 0
    tfidf(t, d)   = tf_weight(t, d) * idf(t)
    cosine(q, d)  = dot(q, d) / (|q| * |d|)  with q weighted the same way as a document
"""
import json
import math
from pathlib import Path

from preprocess import preprocess

DATA = Path(__file__).resolve().parent.parent / "data"

METHODS = ("tfidf", "count", "boolean")


class InvertedIndex:
    """Inverted index for document indexing and ranked TF-IDF search."""

    def __init__(self):
        self.index = {}          # term -> {doc_id: term count}
        self.doc_lengths = {}    # doc_id -> number of tokens
        self.doc_terms = {}      # doc_id -> {term: term count}  (forward index, for vector norms)
        self._norms = None       # cache of document vector lengths

    # ---------- building ----------
    def add_document(self, doc_id: str, tokens: list[str]):
        if doc_id in self.doc_lengths:
            raise ValueError(f"document already indexed: {doc_id}")
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            self.index.setdefault(t, {})[doc_id] = c
        self.doc_terms[doc_id] = counts
        self.doc_lengths[doc_id] = len(tokens)
        self._norms = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_lengths)

    # ---------- weights ----------
    def df(self, term: str) -> int:
        return len(self.index.get(term, {}))

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log(self.n_docs / df) if df else 0.0

    @staticmethod
    def tf_weight(tf: int) -> float:
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    def tfidf(self, term: str, doc_id: str) -> float:
        tf = self.index.get(term, {}).get(doc_id, 0)
        return self.tf_weight(tf) * self.idf(term)

    def doc_norm(self, doc_id: str) -> float:
        if self._norms is None:
            self._norms = {}
        if doc_id not in self._norms:
            self._norms[doc_id] = math.sqrt(sum(self.tfidf(t, doc_id) ** 2 for t in self.doc_terms[doc_id]))
        return self._norms[doc_id]

    # ---------- search ----------
    def boolean_and(self, terms: list[str]) -> list[str]:
        """Documents that contain ALL the query terms (sorted by id)."""
        terms = list(dict.fromkeys(terms))
        if not terms or any(t not in self.index for t in terms):
            return []
        docs = set(self.index[terms[0]])
        for t in terms[1:]:
            docs &= set(self.index[t])
        return sorted(docs)

    def search_tokens(self, terms: list[str], k: int = 10, method: str = "tfidf",
                      allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Rank documents for already-preprocessed query terms.

        method: 'tfidf' (cosine), 'count' (sum of raw term counts), 'boolean' (AND, ties by count).
        allowed: optional set of doc_ids to restrict results to (used for category filters).
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        q_counts = {}
        for t in terms:
            if t in self.index:
                q_counts[t] = q_counts.get(t, 0) + 1
        if not q_counts:
            return []

        # only documents containing at least one query term are scored (this is what the index is for)
        candidates = set()
        for t in q_counts:
            candidates |= set(self.index[t])
        if method == "boolean":
            if len(q_counts) < len(set(terms)):          # some query term is unknown -> AND fails
                return []
            candidates = set(self.boolean_and(list(q_counts)))
        if allowed is not None:
            candidates &= allowed

        scores = {}
        if method == "tfidf":
            q_w = {t: self.tf_weight(c) * self.idf(t) for t, c in q_counts.items()}
            q_norm = math.sqrt(sum(w * w for w in q_w.values()))
            if q_norm == 0:
                return []
            for d in candidates:
                dot = sum(q_w[t] * self.tfidf(t, d) for t in q_w)
                d_norm = self.doc_norm(d)
                scores[d] = dot / (q_norm * d_norm) if d_norm else 0.0
        else:  # count / boolean
            for d in candidates:
                scores[d] = float(sum(self.index[t].get(d, 0) for t in q_counts))

        # a zero score means the document matched only zero-information words: not a real hit
        ranked = sorted(((d, s) for d, s in scores.items() if s > 0), key=lambda x: (-x[1], x[0]))
        return ranked[:k]

    def search(self, query: str, k: int = 10, method: str = "tfidf",
               allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """Search with a raw query string. The query goes through the SAME preprocess() as documents."""
        return self.search_tokens(preprocess(query), k=k, method=method, allowed=allowed)

    # ---------- persistence ----------
    def save(self, path: str | Path):
        payload = {"index": self.index, "doc_lengths": self.doc_lengths, "doc_terms": self.doc_terms}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls()
        idx.index = payload["index"]
        idx.doc_lengths = payload["doc_lengths"]
        idx.doc_terms = payload["doc_terms"]
        return idx


def build_index(doc_ids: list[str] | None = None, texts_dir: str | Path | None = None) -> InvertedIndex:
    """Index the notices in data/texts (all of them, or only `doc_ids`, given as .txt file names)."""
    texts_dir = Path(texts_dir) if texts_dir else DATA / "texts"
    names = doc_ids if doc_ids is not None else sorted(p.name for p in texts_dir.glob("*.txt"))
    idx = InvertedIndex()
    for name in names:
        text = (texts_dir / name).read_text(encoding="utf-8")
        idx.add_document(name, preprocess(text))
    return idx
