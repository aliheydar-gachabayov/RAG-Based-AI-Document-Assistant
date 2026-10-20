
from extract import extract_pdf, PDF_PATH


def chunk_pages(pages, chunk_size=800, overlap=150):
    """
    Split page text into overlapping character chunks.
    Preserve original PDF page numbers.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer breaking at a word boundary
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

            content = text[start:end].strip()

            if content:
                chunks.append({
                    "page": page["page"],
                    "text": content
                })

            if end == len(text):
                break

            next_start = max(start + 1, end - overlap)
            start = next_start

    return chunks


if __name__ == "__main__":
    pages = extract_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    print(f"Extracted pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {index}, Page {chunk['page']} ---")
        print(chunk["text"][:500])
