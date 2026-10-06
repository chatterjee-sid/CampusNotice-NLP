"""
index.py

Inverted index construction, TF-IDF calculation, and search module.
"""

class InvertedIndex:
    """Inverted index for document indexing and ranked TF-IDF search."""
    def __init__(self):
        self.index = {}
        self.doc_lengths = {}

    def add_document(self, doc_id: str, tokens: list[str]):
        pass

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        pass
