import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import torch

from data import load_jsonl
from retrieval import BM25Retriever, DenseRetriever, TFIDFRetriever
from train import load_model

DATASETS = {
    "J&M NLP Textbook": ("nlp", "data/processed/chunks_nlp.jsonl"),
    "Theory of Computation": ("oov", "data/processed/chunks_oov.jsonl"),
    "WW2 Wikipedia": ("wiki", "data/processed/chunks_wiki.jsonl"),
}

CHECKPOINT = "checkpoints/TextEncoder-4L-256d-scratch/best_model.pt"


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(CHECKPOINT, device=device)
    return model, tokenizer, device


@st.cache_resource(show_spinner="Building indexes for dataset...")
def get_retrievers(dataset_key: str, chunks_path: str):
    chunks = load_jsonl(chunks_path)
    model, tokenizer, device = get_model()

    dense = DenseRetriever()
    dense.build(chunks, model, tokenizer, device=device)

    bm25 = BM25Retriever()
    bm25.build(chunks)

    tfidf = TFIDFRetriever()
    tfidf.build(chunks)

    return dense, bm25, tfidf


def highlight_terms(text: str, query: str) -> str:
    terms = [t for t in query.lower().split() if len(t) > 2]
    for term in terms:
        text = re.sub(rf"(?i)({re.escape(term)})", r"**\1**", text)
    return text


st.set_page_config(page_title="Neural Search Engine", layout="wide")
st.title("Neural Search Engine")

col_ds, col_mode, col_k = st.columns([2, 2, 1])

with col_ds:
    dataset_label = st.selectbox("Dataset", list(DATASETS.keys()))

with col_mode:
    mode = st.radio(
        "Retrieval method",
        ["Neural (Dense)", "BM25", "TF-IDF"],
        horizontal=True,
    )

with col_k:
    k = st.slider("Top-K", min_value=1, max_value=20, value=5)

dataset_key, chunks_path = DATASETS[dataset_label]
st.caption(f"Corpus: `{chunks_path}`")

query = st.text_input(
    "Enter your query", placeholder="e.g. What is the Viterbi algorithm?"
)

if query.strip():
    try:
        model, tokenizer, device = get_model()
        dense_ret, bm25_ret, tfidf_ret = get_retrievers(dataset_key, chunks_path)
    except Exception as e:
        st.error(
            f"Could not load model or build indexes: {e}\n\n"
            "Make sure the model checkpoint exists and the chunk files are present."
        )
        st.stop()

    with st.spinner("Searching..."):
        if mode == "Neural (Dense)":
            results = dense_ret.search(query, model, tokenizer, k=k, device=device)
        elif mode == "BM25":
            results = bm25_ret.search(query, k=k)
        else:
            results = tfidf_ret.search(query, k=k)

    st.markdown(f"**{len(results)} results** for: *{query}*")
    st.divider()

    for rank, (score, text, chunk_id) in enumerate(results, 1):
        with st.container():
            st.markdown(f"**#{rank}** &nbsp; `{chunk_id}` &nbsp; `score: {score:.4f}`")
            st.markdown(highlight_terms(text, query))
            st.divider()
