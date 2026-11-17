"""All settings in one place. Change behaviour here, not deep in the code."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads keys from .env (never commit that file!)

DATA = Path("data")
(DATA / "pdfs").mkdir(parents=True, exist_ok=True)

# Local embedding model, 384 dims. For Bangla/multilingual books use
# "intfloat/multilingual-e5-small" instead.
#GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MIN_SCORE = 0.10  # cosine similarity now (no reranker). MUST be tuned, see step 4

CHUNK_CHARS = 1200   # ~300 tokens: big enough for context, small enough to be precise
TOP_K = 4            # how many passages we give the LLM
#MIN_SCORE = 0.10     # reranker score below this = "not in the book". TUNE with real questions!

# The LLM fallback chain: tried in order, first one that works wins.
# Groq and Gemini both offer OpenAI-compatible endpoints, so one client class fits all.
PROVIDERS = [p for p in [
    {"name": "groq", "base_url": "https://api.groq.com/openai/v1",
     "key": os.getenv("GROQ_API_KEY"),
     "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")},
    {"name": "groq-backup", "base_url": "https://api.groq.com/openai/v1",
     "key": os.getenv("GROQ_FALLBACK_API_KEY"),
     "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")},
    {"name": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
     "key": os.getenv("GEMINI_API_KEY"),
     "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest")},
] if p["key"]]

RATE_LIMIT_PER_MIN = 20   # per student (IP), protects your free quota
