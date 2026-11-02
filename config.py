import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.5")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "text-embedding-3-small"
)

CHROMA_PATH = str(
    (ROOT_DIR / os.getenv("CHROMA_PATH", "chroma_db")).resolve()
)
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME", "banking_policies"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
