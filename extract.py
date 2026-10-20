
from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "sample.pdf"


def extract_pdf(path):
    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

    return pages


if __name__ == "__main__":
    pages = extract_pdf(PDF_PATH)

    print(f"Extracted {len(pages)} text pages.")

    if pages:
        print("\n--- First Page Preview ---")
        print(pages[0]["text"][:1000])
    else:
        print("No extractable text found.")
