import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from retrieval import DenseRetriever, BM25Retriever, TFIDFRetriever


@st.cache_resource(show_spinner="Loading models and indexes...")
def load_all():
    import torch
    from train import load_model
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model("checkpoints/TextEncoder-4L-256d-scratch/best_model.pt", device=device)

    faiss_ret = DenseRetriever()
    faiss_ret.load("indexes/jm_corpus.npy", "indexes/jm_chunks_map.json")

    bm25_ret = BM25Retriever()
    bm25_ret.load("indexes/bm25.pkl")

    tfidf_ret = TFIDFRetriever()
    tfidf_ret.load("indexes/tfidf.pkl")

    return model, tokenizer, faiss_ret, bm25_ret, tfidf_ret, device


def highlight_terms(text: str, query: str) -> str:
    """Wrap query terms in a simple markdown highlight."""
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

query = st.text_input("Enter your query", placeholder="e.g. What is the Viterbi algorithm?")
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
