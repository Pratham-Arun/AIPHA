import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables relative to this file's directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MEDICAL_DOCS_DIR = PROJECT_ROOT / "Medical Documents"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# ── Environment & MongoDB Settings ──
APP_ENV = os.getenv("APP_ENV", "development")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "health_assistant")

# ── RAG Settings ──
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_TOP_K = 6
RETRIEVER_DISTANCE_THRESHOLD = 1.25

# ── Model Settings ──
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def validate_config():
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please update the .env file "
            "in the HealthAssistant directory."
        )
