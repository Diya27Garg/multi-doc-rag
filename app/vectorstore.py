"""FAISS vector store with local embeddings. Metadata is kept alongside each vector."""
import faiss
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.dim = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # inner product = cosine on normalized vectors
        self.chunks: list[dict] = []

    @property
    def sources(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        emb = self.model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.index.add(emb)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results