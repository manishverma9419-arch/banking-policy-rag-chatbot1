import os
from pathlib import Path

from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def start_rag_chatbot():
    print("Project folder:", BASE_DIR)
    print("Searching folder:", DATA_DIR)

    # Check that the data directory exists
    if not DATA_DIR.is_dir():
        print("ERROR: data folder does not exist.")
        print("Create this folder:", DATA_DIR)
        return

    # Find files without relying on a glob pattern
    files = [
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt")
    ]

    print("Files found:", len(files))

    if not files:
        print("No PDF or TXT files found.")
        print("All items currently inside data:")
        for item in DATA_DIR.iterdir():
            print(" -", repr(item.name))
        print("Please check the folder contents and extensions.")
        return

    for path in files:
        print("Found:", path.name, "-", path.stat().st_size, "bytes")

    # Read the API key securely from an environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        print('Run: $env:GEMINI_API_KEY = "YOUR_NEW_API_KEY"')
        return

    client = genai.Client(api_key=api_key)
    uploaded_files = []

    try:
        print("\nUploading documents to Gemini...")

        for path in files:
            print("Uploading:", path.name)
            uploaded = client.files.upload(file=str(path))
            uploaded_files.append(uploaded)

        print("\nAll documents uploaded successfully!")
        print("--- Banking Policy RAG Chatbot ---")
        print("Ask a question about the policies.")
        print("Type 'exit' to quit.\n")

        system_instruction = (
            "You are an expert banking policy assistant. "
            "Answer using only the supplied banking policy documents. "
            "Do not invent policies, deadlines, amounts, or legal requirements. "
            "When possible, identify the source document and relevant page. "
            "If the answer is not supported by the documents, say: "
            "'I cannot find the answer in the provided banking policies.'"
        )

        while True:
            user_query = input("You: ").strip()

            if user_query.lower() == "exit":
                break

            if not user_query:
                continue

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=uploaded_files + [user_query],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                )

                print("\nBot:", response.text or "No text response returned.", "\n")

            except Exception as e:
                print("\nQuestion error:", e, "\n")

    except Exception as e:
        print("\nDocument upload/setup error:", e)

    finally:
        # Clean up uploaded temporary files on the Gemini service
        for uploaded in uploaded_files:
            try:
                client.files.delete(name=uploaded.name)
            except Exception as e:
                print("Could not delete uploaded file:", uploaded.name, e)

        client.close()
        print("Chatbot stopped.")


if __name__ == "__main__":
    start_rag_chatbot()