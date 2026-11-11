"""
extractor.py

Rule-based and regex information extractors for dates (deadline / event / issue date), fees, eligibility,
required documents and action items. Written from scratch: regular expressions plus keyword-proximity rules.

Every extracted value is a dict with
    field        which label field it belongs to ('deadline_dates', 'event_dates', 'notice_dates', 'fees',
                 'eligibility', 'required_documents', 'action_items')
    value        normalised value (ISO date string, integer rupees, canonical condition / document / verbs)
    evidence     the sentence the value came from, with start/end = that sentence's character offsets
    match_start/match_end   where the value itself sits in the text
    confidence   0..1 and  status  'ok' | 'needs_review'  (see evidence.py; threshold 0.6)

Rules that came from the project owner's review of the labels:
  * A mail's own 'Date:' header is its SEND time. It is never an event; it is only used to resolve relative words
    ("today", "tomorrow", "Monday"). Each quoted older message in a thread has its own send time.
  * Ranges and lists expand to every day ("16 to 18th November", "28th, 29th & 30th October 2026").
  * Dates inside older quoted messages are extracted too.
  * Dates are day-first (DD/MM/YYYY): a library never guesses.
"""
import re
from collections import Counter
from datetime import date, timedelta

from evidence import compute_confidence, map_evidence
from preprocess import split_sentences

# =====================================================================================================
# Text helpers
# =====================================================================================================
_MARKUP = "*^_`"


def fold(text: str) -> str:
    """Lowercase and blank out mail markup (`* ^ _ ``) while keeping every character position unchanged."""
    out = []
    for ch in text:
        if ch in _MARKUP:
            out.append(" ")
        else:
            low = ch.lower()
            out.append(low if len(low) == 1 else ch)
    return "".join(out)


def _overlaps(spans, s, e) -> bool:
    return any(s < te and ts < e for ts, te in spans)


def _words_between(text: str) -> int:
    return len(text.split())


# =====================================================================================================
# Dates: finding explicit date mentions
# =====================================================================================================
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
          "nov": 11, "dec": 12}
# '0' is accepted for 'o' in October because OCR often reads "October" as "0ctober"
_MON = (r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|"
        r"[o0]ct(?:[o0]ber)?|nov(?:ember)?|dec(?:ember)?)(?![a-z])")
_ORD = r"(?:[ \t]*(?:st|nd|rd|th))?"

PAT_ISO = re.compile(r"(?<![\d/.\-])(?P<y>20\d{2})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)")
# day sequences that share one month: "28th, 29th & 30th October 2026", "16 to 18th November", "5-10 Oct 2026"
PAT_SEQ = re.compile(r"(?<!\d)(?P<days>\d{1,2}" + _ORD + r"(?:\s*(?:,|&|and|to|-|–|—)\s*\d{1,2}" + _ORD + r")+)"
                     r"\s*(?:of\s+)?" + _MON + r"\.?(?:[\s,.\-]*(?P<y>\d{4}))?")
# numeric, day first: 12/10/2026, 12-10-2026, 12.10.26
PAT_NUM = re.compile(r"(?<![\d/])(?<!\d\.)(?P<d>\d{1,2})[ \t]?(?P<s>[/.\-])[ \t]?(?P<m>\d{1,2})[ \t]?(?P=s)[ \t]?"
                     r"(?P<y>\d{4}|\d{2})(?!\d)")
# day then month name: 12th October 2026, 23-Nov-26, 30th June, 2026
# (no newline between day and month: "...2026-27<newline>October05,2026" must not read the "27" as a day)
PAT_DM = re.compile(r"(?<![\d'’])(?P<d>\d{1,2})" + _ORD + r"[ \t]*(?:of[ \t]+)?[\-.]?[ \t]*" + _MON +
                    r"\.?(?:(?:[ \t,.\-]*(?P<y4>\d{4}))|(?:-(?P<y2>\d{2})(?![\d:])))?")
# month name then day: October 12, 2026, July15,2026
PAT_MD = re.compile(_MON + r"\.?\s*(?P<d>\d{1,2})(?![\d:])" + _ORD + r"(?:\s*,?\s*(?P<y>\d{4}))?")


def _month_number(txt: str) -> int:
    return MONTHS[txt[:3].replace("0", "o")]


def _valid(y, m, d) -> bool:
    try:
        date(y if y else 2024, m, d)          # 2024 is a leap year: Feb 29 is accepted when the year is unknown
        return True
    except ValueError:
        return False


def _two_digit_year(y: str) -> int:
    return 2000 + int(y) if len(y) == 2 else int(y)


def _expand_days(days: str) -> list[int]:
    """'28th, 29th & 30th' -> [28, 29, 30];  '16 to 18th' -> [16, 17, 18];  '5-10' -> [5..10]."""
    nums = list(re.finditer(r"\d{1,2}", days))
    out = [int(nums[0].group())]
    for prev, cur in zip(nums, nums[1:]):
        sep = days[prev.end():cur.start()]
        a, b = int(prev.group()), int(cur.group())
        if re.search(r"\bto\b|[-–—]", sep) and a < b <= a + 31:
            out.extend(range(a + 1, b + 1))
        else:
            out.append(b)
    seen = []
    for d in out:
        if d not in seen:
            seen.append(d)
    return seen


def find_date_mentions(folded: str) -> list[dict]:
    """All explicit date mentions in (already folded) text, as {start, end, d, m, y|None, form, group}.

    A day sequence ("16 to 18th November") yields one mention per day, all sharing the same span and a group id.
    Impossible dates (31/02/2026) are dropped.
    """
    taken, out, gid = [], [], 0

    def add(s, e, d, m, y, form, group=None):
        if not _valid(y, m, d):
            return False
        out.append({"start": s, "end": e, "d": d, "m": m, "y": y, "form": form, "group": group})
        return True

    for mt in PAT_ISO.finditer(folded):
        if add(mt.start(), mt.end(), int(mt["d"]), int(mt["m"]), int(mt["y"]), "iso"):
            taken.append(mt.span())
    for mt in PAT_SEQ.finditer(folded):
        if _overlaps(taken, *mt.span()):
            continue
        month = _month_number(mt["mon"])
        year = int(mt["y"]) if mt["y"] else None
        days = _expand_days(mt["days"])
        if all(_valid(year, month, d) for d in days):
            gid += 1
            for d in days:
                add(mt.start(), mt.end(), d, month, year, "seq", gid)
            taken.append(mt.span())
    for mt in PAT_NUM.finditer(folded):
        if _overlaps(taken, *mt.span()):
            continue
        d, m, y = int(mt["d"]), int(mt["m"]), _two_digit_year(mt["y"])
        if 1 <= m <= 12 and 2000 <= y <= 2100 and add(mt.start(), mt.end(), d, m, y, "num"):
            taken.append(mt.span())
    for mt in PAT_DM.finditer(folded):
        if _overlaps(taken, *mt.span()):
            continue
        y = mt["y4"] or mt["y2"]
        if add(mt.start(), mt.end(), int(mt["d"]), _month_number(mt["mon"]), _two_digit_year(y) if y else None, "dm"):
            taken.append(mt.span())
    for mt in PAT_MD.finditer(folded):
        if _overlaps(taken, *mt.span()):
            continue
        y = int(mt["y"]) if mt["y"] else None
        if add(mt.start(), mt.end(), int(mt["d"]), _month_number(mt["mon"]), y, "md"):
            taken.append(mt.span())
    return sorted(out, key=lambda m: (m["start"], m["d"]))


# =====================================================================================================
# Mails: header, forwarded headers, quoted replies (each message has its own send time)
# =====================================================================================================
_ATTRIBUTION = re.compile(r"(?is)\bOn (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\n]*?\d{1,2}:\d{2}(?:\s*[AP]M)?"
                          r"[^\n]*(?:\n[^\n]*){0,4}?wrote:")
_HDR_CONTEXT = re.compile(r"^[ \t>*]*(?:from|subject|to|cc|forwarded message|-{3,}\s*forwarded)", re.I)
_DATE_LINE = re.compile(r"^[ \t>*]*date\s*:", re.I)


MAX_EVIDENCE = 260          # characters; longer "sentences" are cut at line breaks (OCR text has few full stops)


def _split_long(sent: dict, text: str, limit: int = MAX_EVIDENCE) -> list[dict]:
    """Cut an over-long sentence at line breaks into chunks of at most ~limit characters.

    OCR output and pasted mails often have no full stops, so a whole letterhead plus the first paragraph would become
    one 'sentence'. A chunk also closes after a line that ends with . : ? or ! (once it has some substance)."""
    if len(sent["text"]) <= limit:
        return [sent]
    out, chunk_start, pos = [], None, sent["start"]
    raw = text[sent["start"]:sent["end"]]

    def close(a, b):
        piece = text[a:b]
        s = a + len(piece) - len(piece.lstrip())
        t = piece.strip()
        if t:
            out.append({"text": t, "start": s, "end": s + len(t)})

    for line in raw.split("\n"):
        line_start, line_end = pos, pos + len(line)
        pos = line_end + 1
        if chunk_start is None:
            chunk_start = line_start
        elif line_end - chunk_start > limit:
            close(chunk_start, line_start)
            chunk_start = line_start
        if re.search(r"[.:?!]\s*$", line) and line_end - chunk_start >= 40:
            close(chunk_start, line_end)
            chunk_start = None
    if chunk_start is not None:
        close(chunk_start, sent["end"])
    return out


def _sentences(text: str) -> list[dict]:
    """Sentences with offsets into `text`; over-long ones are cut at line breaks (see _split_long). For a mail the
    header lines are one 'sentence' each, so that the evidence of a body value never includes 'Subject:' / 'Date:'."""
    end = parse_mail_header(text)["header_end"]
    if not end:
        return [c for s in split_sentences(text) for c in _split_long(s, text)]
    out, pos = [], 0
    for line in text[:end].splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            s = pos + len(line) - len(line.lstrip())
            out.append({"text": stripped, "start": s, "end": s + len(stripped)})
        pos += len(line)
    body = text[end:]
    out += [{**c, "start": c["start"] + end, "end": c["end"] + end}
            for s in split_sentences(body) for c in _split_long(s, body)]
    return out


def _first_full_date(folded_piece: str):
    for m in find_date_mentions(folded_piece):
        if m["y"]:
            return date(m["y"], m["m"], m["d"])
    return None


def parse_mail_header(text: str) -> dict:
    """Header of a mail text produced by email_reader: {'subject', 'sent_date', 'header_end'} (None / 0 for PDFs)."""
    info = {"subject": None, "sent_date": None, "header_end": 0}
    if not text.startswith("Subject:"):
        return info
    end = text.find("\n\n")
    info["header_end"] = len(text) if end == -1 else end
    for line in text[:info["header_end"]].splitlines():
        if line.lower().startswith("subject:"):
            info["subject"] = line.split(":", 1)[1].strip()
        elif _DATE_LINE.match(line):
            info["sent_date"] = _first_full_date(fold(line))
    return info


def _thread_structure(text: str, folded: str):
    """Returns (excluded_spans, markers).

    excluded_spans: header Date lines, forwarded-header Date lines and 'On ... wrote:' lines. Dates there are send
                    times, not events, so they are never extracted.
    markers:        sorted [(offset, send_date)]; the send date in force at an offset is that of the last marker before it.
    """
    excluded, markers = [], []
    pos, recent = 0, []
    header = parse_mail_header(text)
    if header["sent_date"]:
        markers.append((0, header["sent_date"]))
    for line in text.splitlines(keepends=True):
        s, e = pos, pos + len(line)
        stripped = line.strip()
        if _DATE_LINE.match(line):
            in_header = s < header["header_end"]
            if in_header or any(_HDR_CONTEXT.match(r) for r in recent[-3:]):
                excluded.append((s, e))
                d = _first_full_date(folded[s:e])
                if d and not in_header:
                    markers.append((s, d))
        if stripped:
            recent.append(line)
        pos = e
    for m in _ATTRIBUTION.finditer(text):
        excluded.append(m.span())
        d = _first_full_date(folded[m.start():m.end()])
        if d:
            markers.append((m.start(), d))
    return excluded, sorted(markers)


def _reference_date(markers, pos):
    ref = None
    for off, d in markers:
        if off <= pos:
            ref = d
        else:
            break
    return ref


# =====================================================================================================
# Dates: relative words ("today", "tomorrow", "Monday")
# =====================================================================================================
_REL_WORD = re.compile(r"\b(day\s+after\s+tomorrow|tomorrow|today|tonight|yesterday)\b")
_WEEKDAY = re.compile(r"\b(?:(this|next|coming|upcoming)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(s?)\b")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_REL_OFFSET = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}


def _find_relative(folded, mention_spans, excluded, markers):
    """Relative expressions resolved against the send date of the message they sit in."""
    out = []
    near = [(s - 22, e + 22) for s, e in mention_spans]          # an explicit date next to the word wins
    for m in _REL_WORD.finditer(folded):
        if _overlaps(excluded, *m.span()) or _overlaps(near, *m.span()):
            continue
        ref = _reference_date(markers, m.start())
        if ref is None:
            continue
        word = " ".join(m.group(1).split())
        days = 2 if word == "day after tomorrow" else _REL_OFFSET[word]
        out.append((m.start(), m.end(), ref + timedelta(days=days)))
    for m in _WEEKDAY.finditer(folded):
        s, e = m.span()
        if m.group(3) or _overlaps(excluded, s, e) or _overlaps(near, s, e):
            continue
        before, after = folded[max(0, s - 14):s], folded[e:e + 24]
        if re.search(r"\b(?:every|each|all)\s*$", before):                       # "every monday"
            continue
        if _WEEKDAY.search(after.lstrip(" ,&")) and re.match(r"\s*(?:to|-|–|—|and|&|,)\s*[a-z]", after):  # "monday to saturday"
            continue
        if re.search(r"(?:to|-|–|—|and|&|,)\s*$", before) and _WEEKDAY.search(before):  # "... to saturday"
            continue
        ref = _reference_date(markers, s)
        if ref is None:
            continue
        ahead = (_WEEKDAYS.index(m.group(2)) - ref.weekday()) % 7 or 7
        out.append((s, e, ref + timedelta(days=ahead)))
    return out


# =====================================================================================================
# Dates: deadline / event / issue date
# =====================================================================================================
_DL_STRONG = re.compile(r"last\s+date|last\s+day|deadline|due\s+date|\bdue\b|on\s+or\s+before|not\s+later\s+than|"
                        r"no\s+later\s+than|latest\s+by|closing\s+date|\bcloses?\b|apply\s+by|submit\s+by|register\s+by")
_DL_WEAK = re.compile(r"\bby\b|\bbefore\b|\btill\b|\buntil\b|\bup\s*to\b|\bupto\b")
_EV_VERB = re.compile(r"scheduled|will\s+be\s+held|held\s+on|to\s+be\s+held|conducted|commenc\w*|start\w*|begin\w*|"
                      r"\bfrom\b|with\s+effect|effective|resum\w*|taken|available")
_EV_NOUN = re.compile(r"interview|assessment|\btests?\b|exam\w*|orientation|session|lecture|closed|closure|shutdown|"
                      r"election|break|\bcamp\b|webinar|workshop|seminar|round|drive|event|presentation|"
                      r"allotment|timetable|attend|vacation|schedule|mock|webcast")
_NOT_A_DATE_EVENT = re.compile(r"(?:on\s+or\s+after|not\s+earlier\s+than|issued\s+on|valid\s+(?:till|until|up\s*to))\s*$")
_EVENT_CONTEXT = re.compile(r"role|time|venue|mode|schedule|details|interview|assessment|session|between|semester")
_ISSUE_BEFORE = re.compile(r"(?:\bdate|dated)\s*[:\-.]*\s*$")     # no \b before 'dated': OCR glues "2023-24dated:-"


def _nearest(regex, before: str, max_dist: int):
    """Smallest number of words between a cue and the date (None if no cue within max_dist words)."""
    best = None
    for m in regex.finditer(before):
        dist = _words_between(before[m.end():])
        if dist <= max_dist and (best is None or dist < best):
            best = dist
    return best


def classify_date(folded: str, start: int, end: int, sent_start: int, sent_end: int, mail: bool = False) -> dict:
    """Decide deadline / event / issued from the words around a date. Returns {type, distance, conflict}.

    mail=True: the notice is a mail, whose issue date is its header, so a "Date:" label in the BODY is an event date."""
    lo = max(sent_start, start - 140, 0)
    before = folded[lo:start]
    after = folded[end:min(sent_end, end + 80)]
    sentence = folded[sent_start:sent_end]

    if _NOT_A_DATE_EVENT.search(before):                                  # "issued on or after 1st April 2026"
        return {"type": "other", "distance": None, "conflict": False}
    # issue date: "Date: 21-09-2026" (unless the label sits in a schedule: "Role: ... Date: 5 Nov"), or "dated: 01/5/2023"
    if _ISSUE_BEFORE.search(before.rstrip(" ")):
        if "dated" in before[-8:] or (not mail and not _EVENT_CONTEXT.search(folded[max(0, start - 60):start])):
            return {"type": "issued", "distance": 0, "conflict": False}
    elif not mail:                                                        # a bare date on its own line, top of a letter
        line_s = folded.rfind("\n", 0, start) + 1
        line_e = folded.find("\n", end)
        line_e = len(folded) if line_e == -1 else line_e
        alone = not re.sub(r"[\s,.:;()\-]", "", folded[line_s:start] + folded[end:line_e])
        # ...right after a reference number such as "No.: DoCSE/MidSem/2026-27" or "D(SW)/35+/2026-27"
        if alone and start < max(400, 0.2 * len(folded)) and re.search(
                r"\bno\b|\bref\b|/\s*\d{2,4}(?:-\d{2,4})?\s*$", folded[max(0, start - 120):start].rstrip()):
            return {"type": "issued", "distance": 0, "conflict": False}

    dl = _nearest(_DL_STRONG, before, 9)
    weak = _nearest(_DL_WEAK, before, 2)
    if weak is not None and (dl is None or weak < dl):
        dl = weak
    ev = _nearest(_EV_VERB, before, 6)
    ev_noun = _nearest(_EV_NOUN, before, 8)                               # "closed today, July 31", "Election : Oct 22"
    if ev_noun is not None and (ev is None or ev_noun < ev):
        ev = ev_noun
    if re.search(r"\bon\s*$", before):                                    # "... held on 12/10/2026"
        ev = 0 if ev is None else min(ev, 0)
    noun = bool(_EV_NOUN.search(sentence))

    if dl is not None and (ev is None or dl <= ev):
        return {"type": "deadline", "distance": dl, "conflict": ev is not None and ev - dl <= 1}
    if ev is not None:
        return {"type": "event", "distance": ev, "conflict": False}
    if noun:
        return {"type": "event", "distance": 12, "conflict": False}         # noun cue only: weak, will be flagged
    return {"type": "unknown", "distance": None, "conflict": False}


def find_dates(text: str) -> list[dict]:
    """Finds every date in a notice, resolves it to an ISO date and types it as deadline, event or notice (issue) date."""
    folded = fold(text)
    sentences = _sentences(text)
    is_mail = text.startswith("Subject:")
    excluded, markers = _thread_structure(text, folded)
    mentions = [m for m in find_date_mentions(folded) if not _overlaps(excluded, m["start"], m["end"])]

    # year of the message each date sits in; else the most common explicit year of the document
    explicit = Counter(m["y"] for m in mentions if m["y"])
    doc_year = explicit.most_common(1)[0][0] if explicit else None
    items = []
    for m in mentions:
        year, assumed = m["y"], False
        if year is None:
            ref = _reference_date(markers, m["start"])
            year = ref.year if ref else doc_year
            assumed = True
        if year is None:                              # no year anywhere: keep the day and month, flag it
            iso = f"????-{m['m']:02d}-{m['d']:02d}"
        elif doc_year and year <= doc_year - 2:       # old reference ("letter dated June 24, 2016"), not this notice's date
            continue
        elif _valid(year, m["m"], m["d"]):
            iso = date(year, m["m"], m["d"]).isoformat()
        else:
            continue
        items.append({"start": m["start"], "end": m["end"], "iso": iso, "assumed": assumed,
                      "relative": False, "group": m["group"], "form": m["form"]})
    spans = [(m["start"], m["end"]) for m in mentions]
    for s, e, d in _find_relative(folded, spans, excluded, markers):
        items.append({"start": s, "end": e, "iso": d.isoformat(), "assumed": False, "relative": True, "group": None,
                      "form": "rel"})
    items.sort(key=lambda i: (i["start"], i["iso"]))

    # "X to Y" between two single dates: the second one shares the first one's type
    leader_of = {}
    singles = [i for i in items if i["group"] is None]
    for a, b in zip(singles, singles[1:]):
        if re.fullmatch(r"\s*(?:to|till|until|through|-|–|—)\s*", folded[a["end"]:b["start"]]):
            leader_of[id(b)] = a

    results, typed = [], {}
    for it in items:
        lead = leader_of.get(id(it))
        key = ("g", it["group"]) if it["group"] is not None else ("r", id(lead)) if lead else None
        sent = map_evidence((it["start"], it["end"]), sentences)
        s0, e0 = (sent["start"], sent["end"]) if sent else (it["start"], it["end"])
        if lead is not None and id(lead) in typed:
            kind = typed[id(lead)]
        elif key and key in typed:
            kind = typed[key]
        else:
            kind = classify_date(folded, it["start"], it["end"], s0, e0, mail=is_mail)
            typed[id(it)] = kind
            if key:
                typed[key] = kind
        if kind["type"] == "other":                      # a date that is neither a deadline, an event nor an issue date
            continue
        field = {"deadline": "deadline_dates", "event": "event_dates", "issued": "notice_dates",
                 "unknown": "event_dates"}[kind["type"]]
        conf, status = compute_confidence({"keyword_distance": kind["distance"], "conflict": kind["conflict"],
                                           "evidence": sent.get("text", "")})
        if it["assumed"]:
            conf = min(conf, 0.55)
        if it["iso"].startswith("????"):
            conf = min(conf, 0.4)
        if it["relative"]:
            conf = min(conf, 0.8)
        if _CHANGED.search(sent.get("text", "")):      # 'extended from X to Y': which date is the live one? ask a human
            conf = min(conf, 0.55)
        status = "ok" if conf >= 0.6 else "needs_review"
        results.append({
            "field": field, "value": it["iso"], "type": kind["type"],
            "raw": text[it["start"]:it["end"]].strip(), "evidence": sent.get("text", ""), "start": s0, "end": e0,
            "match_start": it["start"], "match_end": it["end"], "confidence": round(conf, 2), "status": status,
            "relative": it["relative"], "year_assumed": it["assumed"], "conflict": kind["conflict"],
        })
    return _dedupe(results)


_CHANGED = re.compile(r"\b(?:extended|postponed|rescheduled|preponed|deferred)\b", re.I)


def _dedupe(items: list[dict]) -> list[dict]:
    """One item per (field, value), keeping the most confident one."""
    best = {}
    for it in items:
        k = (it["field"], it["value"])
        if k not in best or it["confidence"] > best[k]["confidence"]:
            best[k] = it
    return sorted(best.values(), key=lambda i: i["match_start"])


# =====================================================================================================
# Fees
# =====================================================================================================
_FEE = re.compile(r"(?:(?<![a-z])(?:rs|inr)\b\.?|₹|\brupees?\b)\s*[:\-]?\s*(?P<a>\d[\d,]*(?:\.\d+)?)(?:\s*/-|(?![\w]))"
                  r"|(?<![\w,.])(?P<b>\d[\d,]*)(?:\s*/-|\s*rupees?\b)")
_NOT_FEE_BEFORE = re.compile(r"ctc|stipend|salary|package|income|compensation|bonus|annual\s+(?:family\s+)?(?:income|salary|package|ctc)|per\s+annum|lpa")
_NOT_FEE_AFTER = re.compile(r"^\s*(?:lpa|lakhs?|lacs?|crores?|cr\b|per\s+annum|p\.?a\b)|bonus|ctc|stipend|joining|fixed")
_FEE_WORD = re.compile(r"\bfees?\b|fine|charges?|payment|\bpay\b|amount|cost|deposit|penalty|refund|per\s+(?:day|student|semester)")


def find_fees(text: str) -> list[dict]:
    """Rupee amounts a student pays. Salary / CTC / stipend / bonus / income limits are NOT fees."""
    folded = fold(text)
    sentences = _sentences(text)
    results = []
    last_excluded = -10_000
    for m in _FEE.finditer(folded):
        raw = (m["a"] or m["b"]).replace(",", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        before = folded[max(0, m.start() - 170):m.start()]       # a "Stipend (per month):" heading can sit ~140 chars back
        after = folded[m.end():m.end() + 30]
        # an amount that follows an excluded one closely belongs to the same list ("B.Tech - INR 50000  MCA - INR 30000")
        if (_NOT_FEE_BEFORE.search(before) or _NOT_FEE_AFTER.search(after) or m.start() - last_excluded < 100):
            last_excluded = m.end()
            continue
        sent = map_evidence(m.span(), sentences)
        low = sent.get("text", "").lower()
        kind = "fine" if re.search(r"fine|penalt|late", low) else "refund" if "refund" in low else "fee"
        conf, status = compute_confidence({"confidence": 0.9 if _FEE_WORD.search(low) else 0.7,
                                           "evidence": sent.get("text", "")})
        results.append({
            "field": "fees", "value": int(amount) if amount == int(amount) else amount, "kind": kind,
            "raw": text[m.start():m.end()].strip(), "evidence": sent.get("text", ""),
            "start": sent.get("start", m.start()), "end": sent.get("end", m.end()),
            "match_start": m.start(), "match_end": m.end(), "confidence": conf, "status": status,
        })
    return _dedupe_by_value(results)


def _dedupe_by_value(items: list[dict]) -> list[dict]:
    best = {}
    for it in items:
        if it["value"] not in best or it["confidence"] > best[it["value"]]["confidence"]:
            best[it["value"]] = it
    return sorted(best.values(), key=lambda i: i["match_start"])


# =====================================================================================================
# Required documents (gazetteer = our own list of known document names)
# =====================================================================================================
DOCUMENT_GAZETTEER = {
    "resume": r"\bresumes?\b|\bcv\b|curriculum\s+vitae",
    "marksheet": r"mark\s*sheets?|marks\s*card|grade\s*sheets?|transcripts?",
    "id card": r"\bid\s*card|identity\s*card",
    "aadhaar": r"aadh?aa?r",
    "fee receipt": r"fee\s*receipts?|\breceipts?\b",
    "bonafide certificate": r"bona\s*fide",
    "photograph": r"passport[\s\-]*size\s*photo|\bphotographs?\b(?!\s+of\s+the)",
    "noc": r"\bnoc\b|no\s+objection\s+certificate",
    "income certificate": r"income\s*certificate",
    "caste certificate": r"caste\s*certificate",
    "affidavit": r"affidavit",
    "income tax return": r"\bitr\b|income\s*tax\s*return",
    "bank details": r"pass\s*book|cancell?ed\s*cheque",
    "form": r"\bforms?\b|annexure",
    "letter": r"(?:internship|permission|offer|recommendation|undertaking)\s*letters?",
    "project report": r"project\s*report|final\s*report",
    "abstract": r"abstract",                 # no \b: OCR glues words ("copyofabstract")
    "proof": r"\bproofs?\b",
    "hall ticket": r"hall\s*ticket|admit\s*card",
    "printout": r"print[\s\-]*outs?",
}
_DOC_COMPILED = {name: re.compile(rx) for name, rx in DOCUMENT_GAZETTEER.items()}
_DOC_TRIGGER = re.compile(r"document|submit|bring|carry|attach|enclos|along\s+with|required|produce|upload|"
                          r"\bfill|following|should\s+have|must\s+have|annex|with\s+(?:their|your)\b")


def canonical_documents(text: str) -> set[str]:
    """Gazetteer names found in free text. Used for BOTH predictions and labels so they can be compared fairly."""
    low = fold(text)
    found = {name for name, rx in _DOC_COMPILED.items() if rx.search(low)}
    return found if found else ({" ".join(low.split())} if low.strip() else set())


def find_documents(text: str) -> list[dict]:
    """Document names from the gazetteer, in sentences that have a trigger word (submit, bring, attach, ...)
    in the sentence itself or shortly before it (bullet lists after 'documents required:')."""
    folded = fold(text)
    sentences = _sentences(text)
    results = []
    for sent in sentences:
        own = folded[sent["start"]:sent["end"]]
        context = folded[max(0, sent["start"] - 300):sent["start"]]
        own_trigger, ctx_trigger = bool(_DOC_TRIGGER.search(own)), bool(_DOC_TRIGGER.search(context))
        if not (own_trigger or ctx_trigger):
            continue
        for name, rx in _DOC_COMPILED.items():
            m = rx.search(own)
            if m:
                conf, status = compute_confidence({"confidence": 0.9 if own_trigger else 0.65,
                                                   "evidence": sent["text"]})
                results.append({
                    "field": "required_documents", "value": name, "raw": text[sent["start"] + m.start():sent["start"] + m.end()],
                    "evidence": sent["text"], "start": sent["start"], "end": sent["end"],
                    "match_start": sent["start"] + m.start(), "match_end": sent["start"] + m.end(),
                    "confidence": conf, "status": status,
                })
    return _dedupe_by_value(results)


# =====================================================================================================
# Eligibility conditions
# =====================================================================================================
_OP_LOW = re.compile(r"below|less\s+than|under|\bmax(?:imum)?\b|not\s+more\s+than|at\s+most|<")
_OP_NUM = r"(?P<v>\d{1,2}(?:\.\d+)?)"
_CGPA_CONN = (r"(?:(?:of|is|should\s+be|must\s+be|at\s+least|minimum|min\.?|above|below|under|over|less\s+than|"
              r"more\s+than|greater\s+than|not\s+less\s+than|not\s+more\s+than|>=|=>|>|<|=|≥|:)\s*)*")
_CGPA_A = re.compile(r"(?:cgpa|c\.g\.p\.a\.?|\bgpa)\s*" + _CGPA_CONN + _OP_NUM + r"(?![\d.]*\s*%)")
_CGPA_B = re.compile(_OP_NUM + r"(?![\d.]*\s*%)\s*(?:\+\s*|(?:and|or)\s+(?:above|higher|more)\s*)?(?:cgpa|gpa)")
_PERCENT = re.compile(r"(?P<pre>(?:minimum|min\.?|at\s+least|not\s+less\s+than|above|more\s+than|atleast)\s*(?:of\s*)?)?"
                      r"(?P<v>\d{2,3}(?:\.\d+)?)\s*%")
_BACKLOG = re.compile(r"(?:\bno\b|without|zero|\bnil\b|not\s+have\s+any|should\s+not\s+have\s+any)\s+(?:any\s+)?(?:active\s+|current\s+|pending\s+|standing\s+)?(?:backlogs?|arrears?)")
_BATCH = [
    re.compile(r"(?P<y>20\d{2})\s*(?:[-–]\s*(?P<y2>20\d{2})\s*)?pass(?:ing)?[\s\-]*out"),
    re.compile(r"batch\s*(?:of|:)?\s*(?P<y>20\d{2})(?:\s*[-–]\s*(?P<y2>20\d{2}))?"),
    re.compile(r"class\s+of\s+(?P<y>20\d{2})"),
    re.compile(r"(?P<y>20\d{2})\s*[-–]?\s*(?:batch)"),
]
_GAP_UPTO = re.compile(r"gaps?\s*(?:of\s*)?(?:up\s*to|upto|maximum|max\.?(?:\s*of)?)\s*(?P<n>\d+)\s*years?")
_GAP_NONE = re.compile(r"\bno\s+(?:year\s+|academic\s+|education\s+|study\s+)?gaps?")


def _num(v: float) -> str:
    return f"{v:g}"


def parse_conditions(text: str) -> list[dict]:
    """Structured eligibility conditions found in free text: cgpa, percent, backlogs, batch, gap.

    Each is {type, op, value, canonical, start, end}; `canonical` is e.g. 'cgpa>=7', 'backlogs=0', 'batch=2027'.
    """
    low = fold(text)
    found, taken = [], []

    def add(kind, op, value, s, e, op_word=False):
        if _overlaps(taken, s, e):
            return
        taken.append((s, e))
        val = _num(value) if isinstance(value, float) else str(value)
        found.append({"type": kind, "op": op, "value": value, "canonical": f"{kind}{op}{val}", "start": s, "end": e,
                      "op_word": op_word})

    for m in _PERCENT.finditer(low):
        ctx = low[max(0, m.start() - 60):m.end() + 40]
        if m["pre"] or re.search(r"equivalent|aggregate|marks|percent", ctx):
            v = float(m["v"])
            if 30 <= v <= 100:
                add("percent", ">=", v, m.start(), m.end(), bool(m["pre"]))
    for rx in (_CGPA_A, _CGPA_B):
        for m in rx.finditer(low):
            v = float(m["v"])
            if not 0 < v <= 10:
                continue
            around = low[max(0, m.start() - 30):m.start()] + " " + low[m.start():m.end()]
            op = "<" if _OP_LOW.search(around) else ">="
            add("cgpa", op, v, m.start(), m.end(), bool(re.search(r"minimum|min\.|at\s+least|above|not\s+less", around)))
    for m in _BACKLOG.finditer(low):
        add("backlogs", "=", 0, m.start(), m.end(), True)
    for rx in _BATCH:
        for m in rx.finditer(low):
            y2 = m.groupdict().get("y2")
            value = f"{m['y']}-{y2}" if y2 else m["y"]
            add("batch", "=", value, m.start(), m.end(), True)
    for m in _GAP_UPTO.finditer(low):
        add("gap", "<=", int(m["n"]), m.start(), m.end(), True)
    for m in _GAP_NONE.finditer(low):
        add("gap", "=", 0, m.start(), m.end(), True)
    return sorted(found, key=lambda c: c["start"])


def canonical_conditions(text: str) -> set[str]:
    """Canonical condition strings in free text (labels and predictions both go through this)."""
    return {c["canonical"] for c in parse_conditions(text)}


def find_eligibility(text: str) -> list[dict]:
    """Structured eligibility conditions (CGPA, percentage, backlogs, batch, academic gap) with evidence."""
    sentences = _sentences(text)
    results = []
    for c in parse_conditions(text):
        sent = map_evidence((c["start"], c["end"]), sentences)
        conf, status = compute_confidence({"confidence": 0.9 if c["op_word"] else 0.7, "evidence": sent.get("text", "")})
        results.append({
            "field": "eligibility", "value": c["canonical"],
            "condition": {"type": c["type"], "op": c["op"], "value": c["value"]},
            "raw": text[c["start"]:c["end"]].strip(), "evidence": sent.get("text", ""),
            "start": sent.get("start", c["start"]), "end": sent.get("end", c["end"]),
            "match_start": c["start"], "match_end": c["end"], "confidence": conf, "status": status,
        })
    return _dedupe_by_value(results)


# =====================================================================================================
# Action items
# =====================================================================================================
_ACTION_VERBS = {
    "register": r"register(?:ed|ing|s)?", "submit": r"submit(?:ted|ting|s)?", "apply": r"appl(?:y|ied|ies|ying)",
    "pay": r"pay(?:s|ing|ment)?|paid", "bring": r"bring(?:ing|s)?", "carry": r"carry(?:ing)?", "fill": r"fill(?:ed|ing)?",
    "upload": r"upload(?:ed|ing)?", "contact": r"contact(?:ed|ing)?", "attend": r"attend(?:ed|ing)?",
    "check": r"check(?:ed|ing)?", "join": r"join(?:ed|ing)?", "confirm": r"confirm(?:ed|ing)?",
    "inform": r"inform(?:ed|ing)?", "complete": r"complete(?:d)?", "collect": r"collect(?:ed|ing)?",
    "visit": r"visit(?:ed|ing)?", "download": r"download(?:ed|ing)?", "send": r"send(?:ing)?|sent",
    "select": r"select(?:ed|ing)?", "ensure": r"ensure(?:d)?", "install": r"install(?:ed|ing)?",
    "provide": r"provide(?:d)?", "share": r"share(?:d)?", "verify": r"verif(?:y|ied)", "reschedule": r"reschedul(?:e|ed)",
    "adjust": r"adjust(?:ed)?", "report": r"(?:(?<=to )|(?<=should )|(?<=must )|(?<=please ))report",
    "enroll": r"enrol(?:l|led)?", "withdraw": r"withdr(?:aw|awn)",
}
_ACTION_RX = {v: re.compile(r"\b(?:" + rx + r")\b") for v, rx in _ACTION_VERBS.items()}
_MODAL = re.compile(r"\b(?:should|must|shall|need\s+to|needs\s+to|have\s+to|has\s+to|required\s+to|requested\s+to|advised\s+to|"
                    r"instructed\s+to|asked\s+to|are\s+to|is\s+to|kindly|please|pls|expected\s+to|interested|"
                    r"may\s+(?:visit|contact|apply|register|submit))\b")
_IMPERATIVE = re.compile(r"^\W*(?:\d+[.)]\s*)?(?:" + "|".join(_ACTION_VERBS.values()) + r")\b")


def action_verbs(text: str) -> set[str]:
    """Action verbs (canonical names) used in free text. Used for BOTH predictions and labels."""
    low = fold(text)
    return {v for v, rx in _ACTION_RX.items() if rx.search(low)}


def find_actions(text: str) -> list[dict]:
    """Sentences that tell the reader to do something: an action verb plus a modal ('should', 'must', 'requested to',
    'please', ...) or an imperative start ('Submit ...'). Confidence rises if a deadline, portal or link is mentioned."""
    folded = fold(text)
    results = []
    for sent in _sentences(text):
        own = folded[sent["start"]:sent["end"]]
        verbs = sorted(action_verbs(own))
        if not verbs:
            continue
        modal, imperative = bool(_MODAL.search(own)), bool(_IMPERATIVE.match(own.lstrip()))
        if not (modal or imperative):
            continue
        bonus = 0.15 if re.search(r"portal|\bmis\b|http|www\.|\blink\b|\bform\b|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|"
                                  r"\bby\b|before|deadline|last date", own) else 0.0
        conf, status = compute_confidence({"confidence": min(0.9, 0.7 + bonus), "evidence": sent["text"]})
        results.append({
            "field": "action_items", "value": " ".join(sent["text"].split())[:220], "verbs": verbs,
            "evidence": sent["text"], "start": sent["start"], "end": sent["end"],
            "match_start": sent["start"], "match_end": sent["end"], "confidence": conf, "status": status,
        })
    return results


# =====================================================================================================
# One call for everything
# =====================================================================================================
FIELDS = ("deadline_dates", "event_dates", "fees", "eligibility", "required_documents", "action_items")


def analyze_notice(text: str, classifier=None, preprocess_kwargs: dict | None = None) -> dict:
    """Category (if a trained classifier is given) plus every extracted field, each with evidence and status.

    A notice with two or more different deadlines is flagged: each deadline's confidence is capped below the review
    threshold, because we cannot tell which one the reader needs to act on first.
    """
    header = parse_mail_header(text)
    dates = find_dates(text)
    fields = {f: [] for f in FIELDS}
    notice_dates = []
    for d in dates:
        (notice_dates if d["field"] == "notice_dates" else fields[d["field"]]).append(d)
    fields["fees"] = find_fees(text)
    fields["eligibility"] = find_eligibility(text)
    fields["required_documents"] = find_documents(text)
    fields["action_items"] = find_actions(text)

    if len({d["value"] for d in fields["deadline_dates"]}) >= 2:
        for d in fields["deadline_dates"]:
            d["conflict"] = True
            d["confidence"] = min(d["confidence"], 0.55)
            d["status"] = "needs_review"

    category = proba = cat_status = None
    if classifier is not None:
        from preprocess import preprocess
        tokens = preprocess(text, **(preprocess_kwargs or {"remove_stop": True}))
        probs = classifier.predict_proba(tokens)
        category = max(probs, key=probs.get)
        proba = round(probs[category], 3)
        cat_status = "ok" if proba >= 0.6 else "needs_review"

    all_items = [i for f in FIELDS for i in fields[f]]
    return {
        "category": category, "category_proba": proba, "category_status": cat_status,
        "subject": header["subject"], "sent_date": header["sent_date"].isoformat() if header["sent_date"] else None,
        "notice_dates": notice_dates, "fields": fields,
        "n_items": len(all_items), "n_needs_review": sum(i["status"] == "needs_review" for i in all_items),
    }
