"""
make_split.py

Creates data/split.json: a fixed, stratified 60/20/20 train/dev/test split of the labeled notices.
Near-duplicate documents (same `group` in labels.json) are treated as ONE unit so they never
end up on both sides of a split (leakage). Refuses to overwrite an existing split: it never changes.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42
DEV_FRAC, TEST_FRAC = 0.20, 0.20


def make_split(labels: dict, seed: int = SEED) -> dict:
    # 1. units = groups (kept together) or single documents
    units = defaultdict(list)
    for doc, rec in sorted(labels.items()):
        units[rec.get("group") or doc].append(doc)

    # 2. stratify units by category
    by_cat = defaultdict(list)
    for key, docs in sorted(units.items()):
        cats = {labels[d]["category"] for d in docs}
        assert len(cats) == 1, f"group {key} mixes categories {cats}"
        by_cat[cats.pop()].append(docs)

    rng = random.Random(seed)
    split = {"train": [], "dev": [], "test": []}
    for cat, cat_units in sorted(by_cat.items()):
        rng.shuffle(cat_units)
        n = len(cat_units)
        n_test = max(1, round(TEST_FRAC * n)) if n >= 3 else 0
        n_dev = max(1, round(DEV_FRAC * n)) if n >= 3 else 0
        for i, docs in enumerate(cat_units):
            part = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
            split[part].extend(docs)

    return {part: sorted(docs) for part, docs in split.items()}


if __name__ == "__main__":
    out = DATA / "split.json"
    if out.exists():
        raise SystemExit(f"{out} already exists. The split must never change; delete it by hand only if you really mean to.")
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    split = make_split(labels)
    payload = {"seed": SEED, "fractions": "60/20/20 by unit, stratified by category, groups kept together", **split}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for part in ("train", "dev", "test"):
        cats = defaultdict(int)
        for d in split[part]:
            cats[labels[d]["category"]] += 1
        print(f"{part:5s} {len(split[part]):2d} docs  {dict(sorted(cats.items()))}")
