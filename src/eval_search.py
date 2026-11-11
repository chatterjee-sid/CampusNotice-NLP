"""
eval_search.py

Compares the three search methods (boolean AND, raw count, TF-IDF cosine) on the queries in data/search_queries.json.
Search has no training, so the whole corpus (39 notices) is indexed and the train/dev/test split does not apply.

  python src/eval_search.py            # summary table
  python src/eval_search.py --detail   # per-query top-3 for every method, relevant hits marked with *
"""
import argparse
import json
from pathlib import Path

from index import METHODS, build_index
from metrics import evaluate_search

DATA = Path(__file__).resolve().parent.parent / "data"
ORDER = ("boolean", "count", "tfidf")


def load_queries(names: list[str]) -> list[tuple[str, set[str]]]:
    spec = json.loads((DATA / "search_queries.json").read_text(encoding="utf-8"))
    out = []
    for item in spec["queries"]:
        rel = set()
        for prefix in item["relevant"]:
            found = [n for n in names if n.startswith(prefix)]
            if len(found) != 1:
                raise ValueError(f"{prefix!r} matches {len(found)} notices for query {item['q']!r}")
            rel.add(found[0])
        out.append((item["q"], rel))
    return out


def run(k: int = 3):
    idx = build_index()
    queries = load_queries(sorted(idx.doc_lengths))
    relevant = {q: rel for q, rel in queries}
    results, rankings = {}, {}
    for m in ORDER:
        rankings[m] = {q: [d for d, _ in idx.search(q, k=idx.n_docs, method=m)] for q, _ in queries}
        results[m] = evaluate_search(rankings[m], relevant, k=k)
    return queries, relevant, rankings, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    queries, relevant, rankings, results = run(a.k)
    k = a.k
    print(f"{len(queries)} queries over the whole corpus; relevance judgments are drafts (not yet user-verified)")
    print(f"\n{'method':<9}{'P@'+str(k):>7}{'R@'+str(k):>7}{'MRR':>7}")
    for m in ORDER:
        r = results[m]
        print(f"{m:<9}{r[f'p@{k}']:>7.3f}{r[f'recall@{k}']:>7.3f}{r['mrr']:>7.3f}")
    ceiling = sum(min(len(rel), k) / k for rel in relevant.values()) / len(relevant)
    print(f"(P@{k} ceiling with these judgments = {ceiling:.3f}, since some queries have fewer than {k} relevant notices)")

    print(f"\n{'query':<46}" + "".join(f"{m + ' rr':>12}" for m in ORDER))
    for q, _ in queries:
        print(f"{q:<46}" + "".join(f"{results[m]['per_query'][q]['rr']:>12.2f}" for m in ORDER))

    if a.detail:
        for q, rel in queries:
            print(f"\n## {q}   ({len(rel)} relevant)")
            for m in ORDER:
                top = rankings[m][q][:k]
                shown = ", ".join(("*" if d in rel else " ") + d[:34] for d in top) or "(no results)"
                print(f"  {m:<8}{shown}")
            missed = [d[:34] for d in rel if d not in rankings["tfidf"][q][:k]]
            if missed:
                print(f"  tfidf top-{k} misses: {missed}")


if __name__ == "__main__":
    main()
