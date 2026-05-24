from __future__ import annotations
import time
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, show_progress: bool = True):
        print(f"  embedder: loading {model_name}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.show_progress = show_progress
        print(f"  embedder: model loaded in {time.time() - t0:.1f}s")

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []

        t0 = time.time()
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.time() - t0
        print(
            f"  embedder: {len(texts)} texts embedded in {elapsed:.1f}s "
            f"({elapsed/len(texts)*1000:.1f}ms per chunk)"
        )
        return [row.tolist() for row in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].tolist()