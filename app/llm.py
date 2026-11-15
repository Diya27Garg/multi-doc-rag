import os
import time
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, GEMINI_MODEL

# Optional: comma-separated backups in .env, e.g. GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite
FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=25_000),  # 25 s per attempt
        )
    return _client


def generate(prompt: str) -> str:
    client = _get_client()
    last_error = None
    for model in [GEMINI_MODEL] + FALLBACKS:
        for attempt in range(2):  # 2 tries per model
            start = time.time()
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                print(f"[LLM] {model} took {time.time() - start:.1f}s", flush=True)
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                print(f"[LLM] {model} attempt {attempt + 1} failed: {str(e)[:100]}", flush=True)
                time.sleep(1.5)
    raise last_error