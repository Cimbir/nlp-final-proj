import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# search(query, k) -> list[(score, text, chunk_id)]


class DenseRetriever:
    """Cosine similarity retrieval"""

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        self.matrix: np.ndarray | None = None  # (N, d_model) float32
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict], model, tokenizer,
              batch_size: int = 64, device: str = "cpu") -> None:
        texts = [c["text"] for c in chunks]
        print(f"Encoding {len(texts)} chunks...")
        embs = model.encode(texts, tokenizer, batch_size=batch_size,
                            max_len=256, device=device)
        self.matrix = embs.astype(np.float32)
        self.chunks = chunks
        print(f"Dense index built: {self.matrix.shape[0]} vectors of dim {self.matrix.shape[1]}")

    def save(self, index_path: str = "indexes/jm_corpus.npy",
             map_path: str = "indexes/jm_chunks_map.json") -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(index_path, self.matrix)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        print(f"Saved dense index to {index_path}, chunk map to {map_path}")

    def load(self, index_path: str = "indexes/jm_corpus.npy",
             map_path: str = "indexes/jm_chunks_map.json") -> None:
        self.matrix = np.load(index_path)
        with open(map_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.d_model = self.matrix.shape[1]
        print(f"Loaded dense index: {self.matrix.shape[0]} vectors of dim {self.d_model}")

    def search(self, query: str, model, tokenizer, k: int = 10,
               device: str = "cpu") -> list[tuple[float, str, str]]:
        """Returns top-k (score, text, chunk_id) tuples"""
        emb = model.encode([query], tokenizer, batch_size=1,
                           max_len=64, device=device).astype(np.float32)
        # (N, d) @ (d, 1) -> (N, 1) -> (N,)
        scores = (self.matrix @ emb.T).squeeze(1)
        top_k = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]["text"], self.chunks[i]["id"])
                for i in top_k]


class BM25Retriever:
    """BM25 keyword retrieval using rank_bm25"""

    def __init__(self):
        self.bm25 = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(chunks)} chunks.")

    def save(self, path: str = "indexes/bm25.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str = "indexes/bm25.pkl") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        print(f"BM25 index loaded: {len(self.chunks)} chunks")

    def search(self, query: str, k: int = 10, **_) -> list[tuple[float, str, str]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]["text"], self.chunks[i]["id"])
                for i in top_idx]


class TFIDFRetriever:
    """TF-IDF cosine similarity retrieval using scikit-learn"""

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True, max_features=50000, sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])
        print(f"TF-IDF index built: {self.tfidf_matrix.shape}")

    def save(self, path: str = "indexes/tfidf.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix,
                 "chunks": self.chunks}, f
            )
        print(f"TF-IDF index saved to {path}")

    def load(self, path: str = "indexes/tfidf.pkl") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        self.chunks = data["chunks"]
        print(f"TF-IDF index loaded: {len(self.chunks)} chunks")

    def search(self, query: str, k: int = 10, **_) -> list[tuple[float, str, str]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunks[i]["text"], self.chunks[i]["id"])
                for i in top_idx]
