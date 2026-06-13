import os
import sys
from pathlib import Path
import streamlit as st
from retrieval import DenseRetriever, BM25Retriever, TFIDFRetriever
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

PDF_PATH = "data/raw/jm.pdf"
PDF_CHUNKS_PATH = "data/processed/chunks.jsonl"
BEST_MODEL_PATH = "checkpoints/TextEncoder-4L-256d-scratch/best_model.pt"
DENSE_INDEX_PATH = "indexes/jm_corpus.npy"
DENSE_MAP_PATH = "indexes/jm_chunks_map.json"
BM25_PATH = "indexes/bm25.pkl"
TFIDF_PATH = "indexes/tfidf.pkl"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def dense_exists() -> bool:
    return Path(DENSE_INDEX_PATH).exists() and Path(DENSE_MAP_PATH).exists()


def bm25_exists() -> bool:
    return Path(BM25_PATH).exists()


def tfidf_exists() -> bool:
    return Path(TFIDF_PATH).exists()


def index_missing() -> bool:
    return not dense_exists() or not bm25_exists() or not tfidf_exists()


def build_chunks():
    from data import chunk_pdfs, load_chunks

    if Path(PDF_CHUNKS_PATH).exists():
        return load_chunks(PDF_CHUNKS_PATH)

    st.write("Chunking PDF file...")
    return chunk_pdfs([PDF_PATH], out_path=PDF_CHUNKS_PATH)


def build_all() -> None:
    chunks = build_chunks()

    if not dense_exists():
        from train import load_model

        st.write("Building DENSE...")
        model, tokenizer = load_model(BEST_MODEL_PATH, device=device)

        dense_ret = DenseRetriever()
        dense_ret.build(chunks, model, tokenizer, device=device)
        dense_ret.save(DENSE_INDEX_PATH, DENSE_MAP_PATH)

    if not bm25_exists():
        st.write("Building BM25...")
        bm25_ret = BM25Retriever()
        bm25_ret.build(chunks)
        bm25_ret.save(BM25_PATH)

    if not tfidf_exists():
        st.write("Building TF-IDF...")
        tfidf_ret = TFIDFRetriever()
        tfidf_ret.build(chunks)
        tfidf_ret.save(TFIDF_PATH)


@st.cache_resource(show_spinner="Loading models and indexes...")
def load_all():
    from train import load_model

    model, tokenizer = load_model(BEST_MODEL_PATH, device=device)

    faiss_ret = DenseRetriever()
    faiss_ret.load(DENSE_INDEX_PATH, DENSE_MAP_PATH)

    bm25_ret = BM25Retriever()
    bm25_ret.load(BM25_PATH)

    tfidf_ret = TFIDFRetriever()
    tfidf_ret.load(TFIDF_PATH)

    return model, tokenizer, faiss_ret, bm25_ret, tfidf_ret, device


def highlight_terms(text: str, query: str) -> str:
    import re

    terms = [t for t in query.lower().split() if len(t) > 2]
    for term in terms:
        text = re.sub(
            rf"(?i)({re.escape(term)})",
            r"**\1**",
            text,
        )
    return text


st.set_page_config(page_title="Neural Search — SLP", layout="wide")
st.title("Neural Search Engine")
st.caption("Search the Jurafsky & Martin *Speech and Language Processing* textbook")

if index_missing():
    st.warning("Some indexes are missing")
    if st.button("Build Indexes"):
        with st.spinner("Building indexes..."):
            try:
                load_all.clear()
                build_all()
                st.rerun()
            except Exception as e:
                st.error(f"Index Build failed: {e}")
    st.stop()

query = st.text_input(
    "Enter your query", placeholder="e.g. What is the Viterbi algorithm?"
)
col1, col2 = st.columns([1, 2])
with col1:
    k = st.slider("Top-K results", min_value=1, max_value=20, value=5)
with col2:
    mode = st.radio(
        "Retrieval method",
        ["Neural (TextEncoder + FAISS)", "BM25", "TF-IDF"],
        horizontal=True,
    )

if query.strip():
    try:
        model, tokenizer, faiss_ret, bm25_ret, tfidf_ret, device = load_all()
    except Exception as e:
        st.error(
            f"Could not load models/indexes: {e}\n\n"
            "Make sure you have trained the model and built the indexes first."
        )
        st.stop()

    with st.spinner("Searching..."):
        if mode.startswith("Neural (TextEncoder"):
            results = faiss_ret.search(query, model, tokenizer, k=k, device=device)
        elif mode == "BM25":
            results = bm25_ret.search(query, k=k)
        else:
            results = tfidf_ret.search(query, k=k)

    st.markdown(f"**{len(results)} results** for: *{query}*")
    st.divider()

    for rank, (score, text, _) in enumerate(results, 1):
        with st.container():
            st.markdown(f"**Rank {rank}** &nbsp; `score: {score:.4f}`")
            highlighted = highlight_terms(text, query)
            st.markdown(highlighted)
            st.divider()
