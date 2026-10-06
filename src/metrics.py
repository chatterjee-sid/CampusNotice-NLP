"""
metrics.py

Evaluation metrics for search (P@k, MRR) and classification/extraction (Precision, Recall, F1).
"""

def precision_at_k(results: list[str], relevant_docs: set[str], k: int) -> float:
    pass

def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    pass

def evaluate_classification(y_true: list[str], y_pred: list[str]) -> dict:
    pass

def evaluate_extraction(pred_extractions: list[dict], true_extractions: list[dict]) -> dict:
    pass
