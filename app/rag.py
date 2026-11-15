"""Core RAG logic: rewrite -> retrieve -> threshold check -> generate with citations."""
import re
from app import llm
from app.config import TOP_K, TOP_K_MULTI, MIN_SCORE, HISTORY_TURNS
from app.prompts import REWRITE_PROMPT, ANSWER_PROMPT
from app.vectorstore import VectorStore

NOT_FOUND = "I could not find this information in the uploaded documents."
COMPARE_HINTS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|differ|both|between|across)\b", re.I
)


def format_history(history: list[dict]) -> str:
    turns = history[-HISTORY_TURNS * 2:]
    return "\n".join(f"{m['role'].title()}: {m['content']}" for m in turns)


def rewrite_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    try:
        out = llm.generate(
            REWRITE_PROMPT.format(history=format_history(history), question=question)
        )
        return out or question
    except Exception:
        return question  # fail safe: fall back to the original question


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks)


def ask(question: str, history: list[dict], store: VectorStore) -> dict:
    standalone = rewrite_question(question, history)

    multi = bool(COMPARE_HINTS.search(standalone)) and len(store.sources) > 1
    chunks = store.search(standalone, k=TOP_K_MULTI if multi else TOP_K)

    if not chunks or chunks[0]["score"] < MIN_SCORE:
        return {"answer": NOT_FOUND, "sources": [], "standalone": standalone, "found": False}

    chunks = [c for c in chunks if c["score"] >= MIN_SCORE]
    answer = llm.generate(
        ANSWER_PROMPT.format(context=build_context(chunks), question=standalone)
    )
    return {
        "answer": answer,
        "sources": chunks,
        "standalone": standalone,
        "found": NOT_FOUND not in answer,
    }