"""
extractor.py

Rule-based and regex information extractors for dates, deadlines, fees,
eligibility, required documents, and action items.
"""

def find_dates(text: str) -> list[dict]:
    """Finds dates and normalizes them."""
    pass

def find_fees(text: str) -> list[dict]:
    """Extracts money amounts and context."""
    pass

def find_documents(text: str) -> list[dict]:
    """Extracts required documents matching gazetteer/patterns."""
    pass

def find_eligibility(text: str) -> list[dict]:
    """Extracts structured eligibility conditions."""
    pass

def find_actions(text: str) -> list[dict]:
    """Extracts action items from text."""
    pass
