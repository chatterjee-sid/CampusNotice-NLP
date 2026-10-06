"""
evidence.py

Maps extracted information back to source sentences and assigns confidence scores.
"""

def map_evidence(match_span: tuple[int, int], sentences: list[dict]) -> dict:
    """Finds the sentence corresponding to character offsets."""
    pass

def compute_confidence(extraction: dict) -> tuple[float, str]:
    """Calculates confidence score and status ('ok' vs 'needs_review')."""
    pass
