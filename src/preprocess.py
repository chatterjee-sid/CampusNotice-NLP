"""
preprocess.py

Cleans text, performs tokenization, and handles sentence segmentation.
"""
import re
from sys import argv

def clean_text(text: str) -> str:
    """Cleans and normalizes input text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> list[str]:
    """Tokenizes text into words/tokens while preserving dates and decimals."""
    # \d{1,2}[/-]\d{1,2}[/-]\d{2,4} matches dates like 12/31/2020 or 12-31-20
    # \d+\.\d+ matches decimal numbers like 3.14
    # \w+ matches words
    pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d+\.\d+|\w+'
    return re.findall(pattern, text)

def preprocess(text: str) -> list[str]:
    """Runs the full text through cleaning and tokenization."""
    cleaned_text = clean_text(text)
    return tokenize(cleaned_text)

def split_sentences(text: str) -> list[dict]:
    """Splits text into sentences based on punctuation."""
    # (?<!...) negative look behinds, don't split on periods that are part of abbreviations or numbers
    # [.?!]\s+(?=[A-Z]) matches sentence-ending punctuation followed by whitespace and a capital letter
    # |\n(?=\s*(?:[*-]|\d+\.|\([a-z]\))) for newline followed by a bullet, nuumber or a letter in brackets
    boundary_regex = re.compile(
        r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bProf)(?<!\bRs)(?<!\bNo)(?<!\bSt)(?<!\bvs)(?<!\bInc)(?<!\bLtd)(?<!\bCo)(?<!\bJr)(?<!\bSr)(?<!\bPh\.D)(?<!\bM\.D)(?<!\bB\.Sc)(?<!\bM\.Sc)(?<!\bD\.C)(?<!\bR\.N)(?<!\bP\.O)(?<!\bU\.S)(?<!\bU\.K)(?<!\bE\.U)(?<!\bA\.I)(?<!\bB\.C)(?<!\bA\.D)[.?!]\s+(?=[A-Z])|\n(?=\s*(?:[*-]|\d+\.|\([a-z]\)))'  # Negative lookbehind for common abbreviations
    )
    sentences = []
    start = 0
    for match in boundary_regex.finditer(text):
        end = match.start() + 1 if text[match.start()] != '\n' else match.start()
        raw_sentence = text[start:end]
        sentence = raw_sentence.strip()
        if sentence:
            true_start = start + len(raw_sentence) - len(raw_sentence.lstrip())
            sentences.append({
                'text': sentence,
                'start': true_start,
                'end': true_start + len(sentence)
            })

        start = match.end()

    raw_final = text[start:]
    final_text = raw_final.strip()
    if final_text:
        true_start = start + len(raw_final) - len(raw_final.lstrip())
        sentences.append({
            'text': final_text,
            'start': true_start,
            'end': true_start + len(final_text)
        })

    return sentences