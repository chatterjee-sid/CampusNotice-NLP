"""
storage.py

Data storage module for persisting and loading processed notice data (JSON).
"""
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "results.json"


def load_analysis_results(path: str | Path = None) -> dict:
    """Returns {doc_id: result} (empty dict if nothing saved yet or file is corrupt)."""
    path = Path(path) if path else RESULTS_PATH
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_analysis_result(doc_id: str, result: dict, path: str | Path = None):
    """Adds/overwrites one notice's analysis result in the JSON store."""
    path = Path(path) if path else RESULTS_PATH
    data = load_analysis_results(path)
    data[doc_id] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
