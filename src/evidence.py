"""
evidence.py

Maps extracted information back to source sentences and assigns confidence scores.
"""

REVIEW_THRESHOLD = 0.6


def map_evidence(match_span: tuple[int, int], sentences: list[dict]) -> dict:
    """Finds the sentence (from split_sentences) that contains the character span.

    Returns {'text', 'start', 'end'} of the sentence, or an empty dict if no
    sentence contains the span's start offset.
    """
    m_start, m_end = match_span
    for s in sentences:
        if s['start'] <= m_start < s['end'] or (s['start'] <= m_start and m_end <= s['end']):
            return s
    return {}


def compute_confidence(extraction: dict) -> tuple[float, str]:
    """Calculates confidence score and status ('ok' vs 'needs_review').

    Reads optional hints from the extraction dict:
      - 'keyword_distance': words between value and a strong keyword (None = no keyword)
      - 'conflict': True if competing keyword types / conflicting values were seen
      - 'confidence': a pre-computed base score (used if present and no hints given)
    """
    if extraction.get('conflict'):
        score = 0.3
    elif 'keyword_distance' in extraction:
        d = extraction['keyword_distance']
        if d is None:
            score = 0.4
        elif d <= 3:
            score = 0.9
        elif d <= 8:
            score = 0.75
        else:
            score = 0.55
    else:
        score = float(extraction.get('confidence', 0.5))

    if not extraction.get('evidence'):
        score = min(score, 0.4)

    status = 'ok' if score >= REVIEW_THRESHOLD else 'needs_review'
    return score, status
