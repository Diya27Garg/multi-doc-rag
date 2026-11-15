import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5            # normal questions
TOP_K_MULTI = 10     # cross-document comparisons
MIN_SCORE = 0.25     # below this similarity => "not found". Tune later.
HISTORY_TURNS = 6    # past turns used for memory