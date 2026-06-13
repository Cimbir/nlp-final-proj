# Neural Search Engine — NLP Final Project

Neural search over the Jurafsky & Martin *Speech and Language Processing* textbook, using a custom transformer trained from scratch with InfoNCE contrastive learning.

## Setup

```bash
pip install -r requirements.txt
# For GPU training (recommended):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu  # instead of faiss-cpu
```

## Pipeline

### Download training data
```python
from src.data import download_msmarco
download_msmarco(train_size=100_000, val_size=5_000)
```


### Build Train/Validation data from the book
```bash
python -m src.bookdata
```

### Chunk J&M corpus
```python
from src.data import chunk_pdfs
chunk_pdfs(["data/raw/jm_corpus/slp3_ch1.pdf", ...])
```

### Train the model
```bash
python -m src.train
```

### Build indexes
```python
from src.train import load_model
from src.retrieval import FAISSRetriever, BM25Retriever, TFIDFRetriever
from src.data import load_chunks

model, tokenizer = load_model("checkpoints/best_model.pt")
chunks = load_chunks()

faiss_ret = FAISSRetriever()
faiss_ret.build(chunks, model, tokenizer)
faiss_ret.save()

bm25_ret = BM25Retriever()
bm25_ret.build(chunks)
bm25_ret.save()

tfidf_ret = TFIDFRetriever()
tfidf_ret.build(chunks)
tfidf_ret.save()
```

### Evaluate
See `notebook.ipynb` — Section 4 (Results).

### Demo
```bash
streamlit run app.py
```

## Model Architecture

Custom transformer encoder (no pretrained weights):
- Token embedding: up to 30,000 × 256 (vocab size set by training data)
- Sinusoidal positional encoding
- 4× TransformerEncoderLayer (d=256, heads=4, ffn=512, pre-norm)
- Mean pooling + L2 normalization → 256-dim embedding

Loss: InfoNCE with in-batch negatives, temperature=0.05

## Project Structure

```
src/
  model.py      - Custom TextEncoder (from scratch)
  losses.py     - InfoNCE loss
  data.py       - MS MARCO download, J&M chunking, Dataset
  train.py      - Training loop
  retrieval.py  - FAISS + BM25 + TF-IDF retrievers
  metrics.py    - Recall@K, MRR@K, NDCG@K
app.py          - Streamlit demo
notebook.ipynb  - Analysis, training, evaluation
```
