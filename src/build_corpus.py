"""
build_corpus.py

One-time (re-runnable) script that gathers every collected notice into the roadmap layout:
  data/raw_pdfs/  all PDFs (incl. mail attachments and 'newfound')
  data/texts/     one .txt per notice (PDF text/OCR, or scrubbed mail body)
Source folders are only COPIED from, never deleted. Duplicate mail threads are merged
(longest copy kept); the merge list goes to data/duplicates_merged.txt.
"""
import hashlib
import re
import shutil
from pathlib import Path

from email_reader import email_to_text, scrub_pii
from pdf_reader import extract_pdf

DATA = Path(__file__).resolve().parent.parent / "data"
RAW, TEXTS = DATA / "raw_pdfs", DATA / "texts"

# Near-duplicate mail threads: keep the longest, drop the others.
DUP_GROUPS = [
    ["535", "536"],        # ION Group LDP
    ["539", "540"],        # Capgemini Exceller
    ["4247", "4254"],      # Expert lecture (forward + reply)
    ["4784", "4789"],      # 7th Sem students application (+ reply)
]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def collect_pdfs():
    """Copy PDFs from side folders into raw_pdfs; skip byte-identical files."""
    seen = {_md5(p) for p in RAW.glob("*.pdf")}
    names = {p.name for p in RAW.glob("*.pdf")}
    sources = [DATA / "newfound", *sorted(DATA.glob("other_mails/Attachments*")), DATA / "cdc_mails"]
    for folder in sources:
        for pdf in sorted(folder.rglob("*.pdf")):
            h = _md5(pdf)
            if h in seen:
                print(f"[skip identical] {pdf.name}")
                continue
            name = pdf.name
            if name in names:
                name = f"{pdf.stem}__{folder.name}.pdf"
            shutil.copy2(pdf, RAW / name)
            seen.add(h); names.add(name)
            print(f"[pdf copied] {name}")


def mail_files():
    files = sorted((DATA / "cdc_mails" / "messages").glob("*.txt")) + sorted((DATA / "other_mails").glob("*.txt")) \
        + sorted((DATA / "newfound").glob("*.txt"))
    drop, log = set(), []
    for group in DUP_GROUPS:
        members = [f for f in files if re.search(rf'-({"|".join(group)})\.txt$', f.name)]
        if len(members) > 1:
            keep = max(members, key=lambda f: f.stat().st_size)
            for f in members:
                if f != keep:
                    drop.add(f)
                    log.append(f"dropped: {f.name}\n   kept: {keep.name}")
    (DATA / "duplicates_merged.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    return [f for f in files if f not in drop]


def build_texts():
    TEXTS.mkdir(exist_ok=True)
    for old in TEXTS.glob("*.txt"):
        old.unlink()
    for pdf in sorted(RAW.glob("*.pdf")):
        text, ocr = extract_pdf(pdf)
        # Scholarship notice has a table of named students in the middle: drop just the table.
        if pdf.name.startswith("Top Class SC scholarship"):
            text = re.sub(r"\nInters.*?(?=\nPlease note that this scholarship)",
                          "\n[STUDENT LIST REMOVED]", text, flags=re.S)
        (TEXTS / f"{pdf.stem}.txt").write_text(scrub_pii(text), encoding="utf-8")
        print(f"[pdf text] {pdf.name[:60]} {len(text)} chars{' (OCR)' if ocr else ''}")
    for f in mail_files():
        text = scrub_pii(email_to_text(f))
        (TEXTS / f"mail_{f.stem}.txt").write_text(text, encoding="utf-8")
        print(f"[mail text] {f.name[:60]} {len(text)} chars")


if __name__ == "__main__":
    collect_pdfs()
    build_texts()
    print("Done:", len(list(RAW.glob('*.pdf'))), "PDFs,", len(list(TEXTS.glob('*.txt'))), "texts")
