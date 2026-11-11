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

# Alternatives are tried in order, so the specific patterns win over the generic \w+.
_TOKEN_RE = re.compile(
    r'(?<!\d)\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)'   # dates: 12/10/2026, 12-10-26, 12.10.26
    r'|\d{1,2}:\d{2}'                                  # times: 10:30
    r'|\d+(?:\.\d+)?%'                                 # percentages: 60%, 7.5%
    r'|\d+\.\d+'                                       # decimals: 7.5
    r'|[^\W_]+'                                        # everything else: words, integers ('_' is a separator)
)


def tokenize(text: str) -> list[str]:
    """Tokenizes text into words/tokens while keeping dates, times, percentages and decimals whole."""
    return _TOKEN_RE.findall(text)


# Small hand-written English stop list (common words that carry little topic information).
# Deliberately NOT included: words that matter for notices, e.g. 'last', 'date', 'no', 'not', 'before', 'after', 'on'.
STOPWORDS = frozenset("""
a an the and or but if of to in at by for with from as is are was were be been being am do does did have has had
this that these those it its i we you he she they them their our your his her will shall would should can could may
might must so than then there here also any all each such into about over under up out off again further both more
most other some only own same too very s t just which who whom what when where why how
""".split())


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drops stop words. Numbers, dates and other tokens containing digits are always kept."""
    return [t for t in tokens if t not in STOPWORDS or any(ch.isdigit() for ch in t)]


# Longest suffix first. One suffix is stripped per pass; a stem must keep at least 3 letters.
_SUFFIXES = ("ations", "ration", "ation", "ings", "ing", "edly", "ers", "er", "ed", "ies", "ly", "s")


def stem(word: str) -> str:
    """Very light suffix stripper (our own; not Porter). Tokens with digits or <= 3 letters are untouched."""
    if len(word) <= 3 or any(ch.isdigit() for ch in word):
        return word
    for _ in range(2):                                   # two passes: 'registered' -> 'register' -> 'regist'
        for suf in _SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                if suf == "ies":
                    word = word[:-3] + "y"
                elif suf == "s" and word.endswith(("ss", "us", "is")):
                    continue                              # class, status, analysis keep their s
                else:
                    word = word[:-len(suf)]
                    if suf in ("ing", "ed", "edly") and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "ls":
                        word = word[:-1]                  # 'running' -> 'runn' -> 'run'
                break
        else:
            break
    return word


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem(t) for t in tokens]


# ---------- document "shape" features: how date- and number-heavy is a notice? ----------
# Calendars and timetables are mostly dates and numbers, so plain word counts cannot tell them from other notices.
_MONTH = (r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|'
          r'sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b')
_DATE_MENTION = re.compile(
    r'(?<!\d)\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}(?!\d)'                                   # 12/10/2026, 12.10.26
    r'|(?<!\d)\d{1,2}(?:st|nd|rd|th)?[\s.-]*' + _MONTH + r'(?:[\s,.-]*\d{2,4})?'      # 23-nov-26, 12th october 2026
    r'|' + _MONTH + r'\.?\s*\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?'                   # october 12, 2026
)
_NUMBER = re.compile(r'\d+(?:\.\d+)?%?')


def count_date_mentions(text: str) -> int:
    """Number of dates mentioned in the text, in numeric or day-month-name formats (on lowercased text)."""
    return len(_DATE_MENTION.findall(clean_text(text)))


def shape_features(text: str, tokens: list[str]) -> list[str]:
    """Two coarse feature tokens describing the document: how many dates it mentions and how numeric it is.

    Buckets were fixed from the spread of the training notices (calendars/timetables: many dates, >= 30% numbers).
    """
    n_dates = count_date_mentions(text)
    dates = "none" if n_dates == 0 else "few" if n_dates <= 2 else "several" if n_dates <= 5 else "many"
    ratio = sum(bool(_NUMBER.fullmatch(t)) for t in tokens) / max(len(tokens), 1)
    nums = "low" if ratio < 0.05 else "mid" if ratio < 0.15 else "high" if ratio < 0.30 else "very_high"
    return [f"shape_dates_{dates}", f"shape_nums_{nums}"]


def preprocess(text: str, remove_stop: bool = False, stem_words: bool = False, shape_weight: int = 0) -> list[str]:
    """Cleans and tokenizes text. Stop-word removal, stemming and shape features are optional (off by default).

    shape_weight: if > 0, the two shape feature tokens are appended `shape_weight` times each. Repetition is how a
    single feature gets a chance to matter next to hundreds of word counts in a bag-of-words model.
    Documents and queries must always go through the same settings (shape features are for the classifier only).
    """
    tokens = tokenize(clean_text(text))
    if remove_stop:
        tokens = remove_stopwords(tokens)
    if stem_words:
        tokens = stem_tokens(tokens)
    if shape_weight > 0:
        tokens = tokens + shape_features(text, tokens) * shape_weight
    return tokens

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
        if re.fullmatch(r'\s*(\d{1,3}|[A-Za-z]|[ivxIVX]+)[.)]\s*', raw_sentence):
            continue                      # a bare list marker like "1." or "a)" is not a sentence on its own
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