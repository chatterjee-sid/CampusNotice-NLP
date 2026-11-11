"""
classifier.py

Multinomial Naive Bayes text classifier built from scratch.

    P(c | words)  is proportional to  P(c) * prod_i P(w_i | c)
    P(c)          = docs_in_c / N                               (prior)
    P(w | c)      = (count(w, c) + alpha) / (total_c + alpha*|V|)   (Laplace / add-alpha smoothing)

Scores are kept as sums of logs (a product of many tiny probabilities underflows to 0).
Words never seen in training are ignored at prediction time.
"""
import json
import math
from pathlib import Path


class NaiveBayesClassifier:
    """Multinomial Naive Bayes text classifier."""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.word_counts: dict[str, dict[str, int]] = {}   # class -> word -> count
        self.total_words: dict[str, int] = {}              # class -> total tokens
        self.vocab: set[str] = set()

    def fit(self, docs: list[list[str]], labels: list[str]):
        """Trains the Naive Bayes model (training = counting)."""
        if len(docs) != len(labels) or not docs:
            raise ValueError("docs and labels must be non-empty and the same length")
        self.classes = sorted(set(labels))
        n = len(docs)
        doc_counts = {c: 0 for c in self.classes}
        self.word_counts = {c: {} for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}
        self.vocab = set()
        for tokens, c in zip(docs, labels):
            doc_counts[c] += 1
            wc = self.word_counts[c]
            for w in tokens:
                wc[w] = wc.get(w, 0) + 1
                self.vocab.add(w)
            self.total_words[c] += len(tokens)
        self.log_prior = {c: math.log(doc_counts[c] / n) for c in self.classes}
        return self

    def log_likelihood(self, word: str, c: str) -> float:
        v = len(self.vocab)
        return math.log((self.word_counts[c].get(word, 0) + self.alpha) / (self.total_words[c] + self.alpha * v))

    def predict_scores(self, tokens: list[str]) -> dict[str, float]:
        """Log-score (log prior + sum of log likelihoods) for every class. Higher = more likely."""
        if not self.classes:
            raise RuntimeError("call fit() first")
        known = [w for w in tokens if w in self.vocab]
        return {c: self.log_prior[c] + sum(self.log_likelihood(w, c) for w in known) for c in self.classes}

    def predict(self, tokens: list[str]) -> str:
        """Predicts the category for a tokenized document (ties go to the alphabetically first class)."""
        scores = self.predict_scores(tokens)
        return max(self.classes, key=lambda c: (scores[c], -self.classes.index(c)))

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        """Scores turned into probabilities that sum to 1 (softmax over the log scores)."""
        scores = self.predict_scores(tokens)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: e / total for c, e in exp.items()}

    def top_words(self, c: str, n: int = 10) -> list[tuple[str, float]]:
        """Words that most favour class c over the other classes (log-odds), for explaining the model."""
        others = [o for o in self.classes if o != c]
        rows = []
        for w in self.vocab:
            if self.word_counts[c].get(w, 0) == 0:
                continue
            rest = sum(self.log_likelihood(w, o) for o in others) / max(len(others), 1)
            rows.append((w, self.log_likelihood(w, c) - rest))
        return sorted(rows, key=lambda x: -x[1])[:n]

    # ---------- persistence ----------
    def save(self, path: str | Path):
        payload = {"alpha": self.alpha, "classes": self.classes, "log_prior": self.log_prior,
                   "word_counts": self.word_counts, "total_words": self.total_words}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesClassifier":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        nb = cls(alpha=p["alpha"])
        nb.classes, nb.log_prior = p["classes"], p["log_prior"]
        nb.word_counts, nb.total_words = p["word_counts"], p["total_words"]
        nb.vocab = {w for wc in nb.word_counts.values() for w in wc}
        return nb
