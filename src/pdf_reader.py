"""
pdf_reader.py

Extracts raw plain text from PDF files using PyMuPDF (fitz) and saves
extracted text files into the specified directory.
"""

from pathlib import Path
import pymupdf


def pdf_to_text(pdf_path: str | Path) -> str:
    """
    Extracts plain text page by page from a given PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted plain text as a string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_content = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text:
                text_content.append(page_text)
        doc.close()
    except Exception as e:
        print(f"[Error] Failed to read PDF '{pdf_path.name}': {e}")
        return ""

    return "\n".join(text_content)


def process_all_pdfs(raw_dir: str | Path = None, output_dir: str | Path = None) -> list[Path]:
    """
    Extracts text from all PDF files in raw_dir and saves them as .txt files in output_dir.

    Args:
        raw_dir: Directory containing raw PDF files. Default is campusnotice/data/raw_pdfs.
        output_dir: Directory to save extracted .txt files. Default is campusnotice/data/texts.

    Returns:
        List of Path objects for the generated .txt files.
    """
    base_dir = Path(__file__).resolve().parent.parent

    if raw_dir is None:
        raw_dir = base_dir / "data" / "raw_pdfs"
    else:
        raw_dir = Path(raw_dir)

    if output_dir is None:
        output_dir = base_dir / "data" / "texts"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.PDF")))
    if not pdf_files:
        print(f"[Info] No PDF files found in '{raw_dir}'. Place PDF notices in this directory.")
        return []

    processed_files = []
    print(f"Processing {len(pdf_files)} PDF(s) from '{raw_dir}'...")

    for pdf_file in pdf_files:
        extracted_text = pdf_to_text(pdf_file)
        txt_filename = pdf_file.stem + ".txt"
        txt_path = output_dir / txt_filename

        if not extracted_text.strip():
            print(f"[Warning] '{pdf_file.name}' yielded empty text (may be scanned image/empty).")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        processed_files.append(txt_path)
        print(f"[Success] Extracted '{pdf_file.name}' -> '{txt_path.name}' ({len(extracted_text)} chars)")

    return processed_files


if __name__ == "__main__":
    process_all_pdfs()
