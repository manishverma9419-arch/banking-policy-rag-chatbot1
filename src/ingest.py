
import os
from pypdf import PdfReader

DATA_FOLDER = "data/raw"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


def ingest_documents():
    documents = []

    if not os.path.exists(DATA_FOLDER):
        print("PDF folder not found!")
        return documents

    pdf_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found in data/raw!")
        return documents

    for filename in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, filename)

        print(f"\nProcessing: {filename}")

        pages = extract_text_from_pdf(pdf_path)

        for page in pages:
            documents.append({
                "source": filename,
                "page": page["page"],
                "text": page["text"]
            })

        print(f"Extracted {len(pages)} pages")

    return documents


if __name__ == "__main__":
    print("Starting Banking Policy PDF Ingestion...")

    docs = ingest_documents()

    print("\n========== INGESTION SUMMARY ==========")
    print("Total pages extracted:", len(docs))

    for doc in docs[:3]:
        print("\nSource:", doc["source"])
        print("Page:", doc["page"])
        print("Preview:", doc["text"][:200])

    print("\nPDF ingestion completed!")

