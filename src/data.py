import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _write_jsonl(records: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records : {path}")


def _print_stats(train_path: Path, val_path: Path) -> None:
    for label, path in [("Train", train_path), ("Val", val_path)]:
        with open(path) as f:
            records = [json.loads(l) for l in f]
        q_lens = [len(r["query"].split()) for r in records]
        p_lens = [len(r["pos"].split()) for r in records]
        print(
            f"{label}: {len(records)} triplets | "
            f"avg query {sum(q_lens)/len(q_lens):.1f} words | "
            f"avg passage {sum(p_lens)/len(p_lens):.1f} words"
        )


def download_msmarco(
    train_size: int = 100000,
    val_size: int = 5000,
    out_dir: str = "data/processed",
) -> None:
    """
    Download MS MARCO and save train/val triplets
    Each triplet: {"query": str, "pos": str, "neg": str}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "msmarco_train_triplets.jsonl"
    val_path = out_dir / "msmarco_val_triplets.jsonl"

    if train_path.exists() and val_path.exists():
        print("Data already exists, skipping download.")
        _print_stats(train_path, val_path)
        return

    print("Downloading MS MARCO")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True, trust_remote_code=True)

    triplets: list[dict] = []
    skipped = 0

    for example in tqdm(ds, desc="Processing examples"):
        passages = example["passages"]["passage_text"]
        selected = example["passages"]["is_selected"]
        query = example["query"].strip()

        pos_list = [p for p, s in zip(passages, selected) if s == 1]
        neg_list = [p for p, s in zip(passages, selected) if s == 0]

        if not pos_list or not neg_list or not query:
            skipped += 1
            continue

        for pos in pos_list:
            triplets.append({"query": query, "pos": pos, "neg": neg_list[0]})

        if len(triplets) >= train_size + val_size:
            break

    print(f"Triplet amount : {len(triplets)} ({skipped} skipped. no pos or neg)")

    _write_jsonl(triplets[:train_size], train_path)
    _write_jsonl(triplets[train_size : train_size + val_size], val_path)
    _print_stats(train_path, val_path)


def download_squad(
    train_size: int = 80000,
    val_size: int = 5000,
    out_dir: str = "data/processed",
) -> None:
    """
    Download SQuAD and save train/val triplets
    query = question, pos = context paragraph, neg = BM25-mined hard negative
    (top BM25 result for the question that is not the gold context)
    """
    random.seed(42)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "squad_train_triplets.jsonl"
    val_path   = out_dir / "squad_val_triplets.jsonl"

    if train_path.exists() and val_path.exists():
        print("SQuAD data already exists, skipping.")
        _print_stats(train_path, val_path)
        return

    print("Downloading SQuAD")
    ds = load_dataset("rajpurkar/squad", split="train")

    all_contexts = list({ex["context"] for ex in ds})
    print(f"Building BM25 index over {len(all_contexts):,} unique contexts")
    bm25 = BM25Okapi([c.lower().split() for c in all_contexts])

    triplets: list[dict] = []
    for i, ex in enumerate(ds):
        query = ex["question"].strip()
        pos   = ex["context"].strip()

        scores   = bm25.get_scores(query.lower().split())
        top_idxs = scores.argsort()[::-1]

        neg = None
        for idx in top_idxs[:20]:
            candidate = all_contexts[idx]
            if candidate != pos:
                neg = candidate
                break

        if not neg:
            neg = random.choice([c for c in all_contexts if c != pos])

        if query and pos and neg:
            triplets.append({"query": query, "pos": pos, "neg": neg})

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,} / {len(ds):,} processed")

        if len(triplets) >= train_size + val_size:
            break

    print(f"Triplet amount : {len(triplets)}")
    _write_jsonl(triplets[:train_size], train_path)
    _write_jsonl(triplets[train_size:train_size + val_size], val_path)
    _print_stats(train_path, val_path)


def download_nq(
    train_size: int = 80000,
    val_size: int = 5000,
    out_dir: str = "data/processed",
) -> None:
    """
    Download Natural Questions and save train/val triplets
    query = question, pos = Wikipedia long answer, neg = random other passage
    """
    random.seed(42)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "nq_train_triplets.jsonl"
    val_path   = out_dir / "nq_val_triplets.jsonl"

    if train_path.exists() and val_path.exists():
        print("NQ data already exists, skipping.")
        _print_stats(train_path, val_path)
        return

    def _extract_long_answer(ex: dict) -> str | None:
        annotations = ex.get("annotations")
        if not annotations:
            return None
        long_answers = annotations["long_answer"] if isinstance(annotations, dict) else [a["long_answer"] for a in annotations]
        if not long_answers:
            return None
        la = long_answers[0]
        start = la["start_token"]
        end   = la["end_token"]
        if start < 0:
            return None
        doc_tokens = ex["document"]["tokens"]
        if isinstance(doc_tokens, dict):
            token_texts = doc_tokens["token"]
            is_html     = doc_tokens["is_html"]
            text_tokens = [
                token_texts[i] for i in range(start, min(end, len(token_texts)))
                if not is_html[i]
            ]
        else:
            text_tokens = [
                t["token"] for t in doc_tokens[start:end]
                if not t.get("is_html", False)
            ]
        text = " ".join(text_tokens).strip()
        return text if len(text.split()) >= 20 else None

    print("Downloading Natural Questions")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        streaming=True,
    )

    passages: list[str] = []
    triplets: list[dict] = []
    skipped = 0

    for ex in tqdm(ds, desc="Processing NQ"):
        question = ex["question"]["text"].strip()
        pos = _extract_long_answer(ex)
        if not question or not pos:
            skipped += 1
            continue
        passages.append(pos)
        triplets.append({"query": question, "pos": pos, "neg": None})
        if len(triplets) >= train_size + val_size:
            break

    print(f"Triplet amount : {len(triplets)} ({skipped} skipped)")

    for t in triplets:
        neg = random.choice(passages)
        while neg == t["pos"] and len(passages) > 1:
            neg = random.choice(passages)
        t["neg"] = neg

    _write_jsonl(triplets[:train_size], train_path)
    _write_jsonl(triplets[train_size:train_size + val_size], val_path)
    _print_stats(train_path, val_path)


def chunk_pdfs(
    pdf_paths,
    out_path: str = "data/processed/chunks.jsonl",
    chunk_words: int = 250,
    overlap_words: int = 50,
) -> list[dict]:
    """
    Extract text from PDF(s) and split into overlapping word-level chunks

    Args:
        pdf_paths: path or list of paths to PDF files
        out_path: destination JSONL file
        chunk_words: target words per chunk
        overlap_words: overlap between consecutive chunks
    Returns:
        list of chunk dicts: {"id", "text", "source", "word_count"}
    """
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    chunk_id = 0
    step = chunk_words - overlap_words

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        print(f"Processing {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        raw_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            raw_text += text + " "

        raw_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', raw_text)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        words = raw_text.split()

        for i in range(0, max(1, len(words) - overlap_words), step):
            window = words[i : i + chunk_words]
            if len(window) < 50:
                continue
            chunks.append(
                {
                    "id": f"chunk_{chunk_id:04d}",
                    "text": " ".join(window),
                    "source": pdf_path.stem,
                    "word_count": len(window),
                }
            )
            chunk_id += 1

    _write_jsonl(chunks, out_path)
    wc = [c["word_count"] for c in chunks]
    print("stats:")
    print(f"Created {len(chunks)} chunks")
    print(f"avg {sum(wc)/len(wc):.0f} words | range [{min(wc)}, {max(wc)}]")
    return chunks


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TripletDataset(Dataset):
    """
    Tokenizes (query, positive, negative) triplets for contrastive training
    All three fields are returned and used: neg is passed to InfoNCELoss as a hard negative
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        query_max_len: int = 64,
        passage_max_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

        self.triplets: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.triplets.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.triplets)

    def _tok(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        t = self.triplets[idx]
        q_ids, q_mask = self._tok(t["query"], self.query_max_len)
        p_ids, p_mask = self._tok(t["pos"],   self.passage_max_len)
        n_ids, n_mask = self._tok(t["neg"],   self.passage_max_len)
        return {
            "query_ids":  q_ids,  "query_mask": q_mask,
            "pos_ids":    p_ids,  "pos_mask":   p_mask,
            "neg_ids":    n_ids,  "neg_mask":   n_mask,
        }
