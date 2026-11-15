"""
email_reader.py

Reads placement/notice emails (.eml or pasted .txt) and returns plain body text
so they can go through the same pipeline as PDF notices.
"""
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

_HEADER_RE = re.compile(r'^(from|to|cc|bcc|date|sent|subject|reply-to):', re.I)


def email_to_text(path: str | Path) -> str:
    """Returns 'Subject: ...' line plus body for .eml; for .txt strips leading header lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Email file not found: {path}")

    if path.suffix.lower() == ".eml":
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body = body_part.get_content() if body_part else ""
        if body_part is not None and body_part.get_content_type() == "text/html":
            body = re.sub(r'<[^>]+>', ' ', body)
        subject = msg.get("subject", "")
        return f"Subject: {subject}\n{body}".strip()

    # .txt: header block = everything up to the first blank line (headers like CC:
    # wrap onto continuation lines). Keep only Subject and Date; body is the rest.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    kept = []
    while i < len(lines) and lines[i].strip() and _HEADER_RE.match(lines[i].strip()):
        # consume this header and its indented/wrapped continuation lines
        first = lines[i]
        i += 1
        while i < len(lines) and lines[i].strip() and not _HEADER_RE.match(lines[i].strip()):
            i += 1
        if first.lower().startswith(("subject:", "date:")):
            kept.append(first.strip())
    return "\n".join(kept + [""] + lines[i:]).strip()


_ROLL = r'[UuIiPpMm]\d{2}[A-Za-z]{2}\d{3}'


def scrub_pii(text: str) -> str:
    """Removes individuals' details: 'Name - ROLLNO' lines, roll numbers, e-mails, phones."""
    text = re.sub(rf'^[ \t>]*[A-Za-z .\']+?\s*[-–:]\s*{_ROLL}[ \t]*$', '[STUDENT]', text, flags=re.M)
    text = re.sub(rf"[A-Za-z][A-Za-z .']+[(（]\s*{_ROLL}\s*[)）]", '[STUDENT]', text)   # Name(ROLL)
    text = re.sub(rf"[A-Za-z][A-Za-z .']+[(（]\s*\[ROLL\]\s*[)）]", '[STUDENT]', text)  # already-scrubbed
    text = re.sub(rf'\b{_ROLL}\b', '[ROLL]', text)
    text = re.sub(r'<?(mailto|tel):[^>\s]*>?', '', text)
    text = re.sub(r'[\w.+-]+@[\w-]+(\.[\w-]+)+', '[EMAIL]', text)
    text = re.sub(r'(?<!\d)(?:\+?91[- ]?0?|0|1)?[6-9]\d{9}(?!\d)', '[PHONE]', text)
    # Quoted-reply attribution "On Thu, Oct 29, 2026 at 5:09 PM <name> wrote:": the name is masked but the date and
    # time are KEPT, because they are the send time of the quoted message (needed to resolve "tomorrow", "Monday").
    text = re.sub(r"(?is)\bOn ((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\n]*?\d{1,2}:\d{2}(?:\s*[AP]M)?)[^\n]*(?:\n[^\n]*){0,4}?wrote:",
                  r"On \1, [NAME] wrote:", text)
    text = re.sub(r"(?is)\bOn (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\n]*(?:\n[^\n]*){0,4}?wrote:",
                  "On [DATE], [NAME] wrote:", text)       # fallback when no time is present
    text = re.sub(r"(?im)^([ \t>]*(?:thanks\s*(?:&|and)\s*regards|with warm regards|with regards|best regards|regards|best),?[ \t]*\n)"
                  r"[ \t>]*[A-Z][A-Za-z.' ]{2,40}[ \t]*$", r"\1[NAME]", text)
    return _drop_recipient_lists(text)


def _is_recipient_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and bool(re.match(r'^>?\s*[,<]', s) or '[EMAIL]' in s or '[ROLL]@' in s or s.endswith(','))


def _drop_recipient_lists(text: str) -> str:
    """Drops 'To:/Cc:' recipient blocks of forwarded headers, name tables of shortlists,
    and name-only lines next to e-mail placeholders (these hold students' real names)."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        s = lines[i].strip().lstrip('>').strip()
        if re.match(r'(to|cc|bcc):', s, re.I):
            out.append(s.split(':')[0] + ': [RECIPIENTS REMOVED]')
            i += 1
            while i < len(lines) and _is_recipient_line(lines[i]):
                i += 1
            continue
        # shortlist tables: 'FULL NAME' line directly followed by 'SVNIT Surat'
        nxt = lines[i + 1].strip().lstrip('>').strip() if i + 1 < len(lines) else ''
        if nxt == 'SVNIT Surat' and re.fullmatch(r"[A-Za-z][A-Za-z .']+", s):
            out.append('[STUDENT]')
            i += 1
            continue
        # candidate-table rows: 'Name <TAB> [EMAIL] ...' and 'Surname, Firstname____'
        if re.match(r"[A-Za-z][A-Za-z .']+\t.*\[EMAIL\]", s) or re.fullmatch(r"[A-Z][A-Za-z']+,\s+[A-Z][A-Za-z' ]+_*", s):
            out.append('[STUDENT ROW]')
            i += 1
            continue
        if '[EMAIL]' in s and re.search(r'wrote:?\s*$|<\[EMAIL\]', s) and re.match(r"[A-Za-z][A-Za-z .']+ <", s):
            s = re.sub(r"^[A-Za-z][A-Za-z .']+ <", '<', s)   # 'Name <[EMAIL]>' -> '<[EMAIL]>'
        out.append(lines[i] if s == lines[i].strip().lstrip('>').strip() else s)
        i += 1
    return "\n".join(out)
