import json
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

_BPE_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


def _split(text: str) -> list[str]:
    """Lowercase + keep alphanumeric, apostrophes, hyphens"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    return text.split()


class WordTokenizer:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]

    PAD_ID = 0
    UNK_ID = 1
    CLS_ID = 2
    SEP_ID = 3

    def __init__(self):
        self.word2id: dict[str, int] = {t: i for i, t in enumerate(self.SPECIAL_TOKENS)}
        self.id2word: dict[int, str] = {i: t for i, t in enumerate(self.SPECIAL_TOKENS)}

    def build_vocab(self,
                    texts: list[str],
                    max_vocab_size: int = 30000,
                    min_freq: int = 2) -> None:
        """
        Build vocabulary from a list of raw text strings
        Call this once on your training corpus before saving
        """
        counter: Counter = Counter()
        for text in texts:
            counter.update(_split(text))

        for word, freq in counter.most_common():
            idx = len(self.word2id)
            if idx >= max_vocab_size:
                break
            if freq < min_freq:
                break
            if word not in self.word2id:
                self.word2id[word] = idx
                self.id2word[idx] = word

        print(f"Vocabulary built: {len(self.word2id):,} tokens ")

    def encode(self,
               text: str,
               max_length: int = 256,
               truncation: bool = True) -> tuple[list[int], list[int]]:
        """Encode string -> (input_ids, attention_mask)"""
        tokens = [self.CLS_ID]
        tokens.extend([self.word2id.get(t, self.UNK_ID) for t in _split(text)])
        tokens.append(self.SEP_ID)
        if truncation:
            tokens = tokens[:max_length]
        mask = [1] * len(tokens)
        return tokens, mask

    def pad_batch(self,
                  batch_ids: list[list[int]],
                  batch_masks: list[list[int]],
                  max_length: int | None = None,
                  pad_to_max: bool = False) -> tuple[list, list]:
        """Pad a batch of variable-length sequences"""
        if pad_to_max and max_length is not None:
            target = max_length
        else:
            target = max(len(ids) for ids in batch_ids)
            if max_length is not None:
                target = min(target, max_length)

        padded_ids, padded_masks = [], []
        for ids, mask in zip(batch_ids, batch_masks):
            pad_len = target - len(ids)
            padded_ids.append(ids + [self.PAD_ID] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
        return padded_ids, padded_masks

    def __call__(self,
                 texts: str | list[str],
                 max_length: int = 256,
                 padding: bool | str = True,
                 truncation: bool = True,
                 return_tensors: bool = False) -> dict:
        """
        Args:
            texts: single string or list of strings
            max_length: maximum sequence length (used when truncation=True or padding="max_length")
            padding: True / "longest" -> pad to longest in batch; "max_length" -> pad to max_length
            truncation: truncate to max_length
            return_tensors: "pt" to return torch.Tensors, None for lists
        Returns:
            dict with "input_ids" and "attention_mask"
        """
        if isinstance(texts, str):
            texts = [texts]

        all_ids, all_masks = [], []
        for text in texts:
            ids, mask = self.encode(text, max_length=max_length, truncation=truncation)
            all_ids.append(ids)
            all_masks.append(mask)

        pad_to_max = (padding == "max_length")
        all_ids, all_masks = self.pad_batch(all_ids, all_masks, max_length=max_length, pad_to_max=pad_to_max)

        if return_tensors:
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_masks, dtype=torch.long),
                }
        return {
            "input_ids": all_ids,
            "attention_mask": all_masks
            }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2id": self.word2id}, f, ensure_ascii=False)
        print(f"Tokenizer saved : {path} (vocab size: {self.vocab_size:,})")

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.word2id = data["word2id"]
        tok.id2word = {int(v): k for k, v in tok.word2id.items()}
        print(f"Custom tokenizer loaded from {path} (vocab size: {tok.vocab_size:,})")
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    def __len__(self) -> int:
        return len(self.word2id)


def build_bpe_tokenizer(
    corpus_paths: list[str],
    extra_texts: list[str] | None = None,
    save_dir: str = "data/processed",
    vocab_size: int = 30_000,
) -> PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer from scratch on triplet corpora + optional raw texts
    Saves to {save_dir}/bpe_tokenizer.json and returns a PreTrainedTokenizerFast

    Args:
        corpus_paths: JSONL files with {"query", "pos", "neg"} records
        extra_texts: additional raw text strings (e.g. J&M corpus chunks)
        save_dir: directory to write bpe_tokenizer.json
        vocab_size: target vocabulary size including special tokens
    """
    def _corpus_iterator():
        for path in corpus_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    for field in ("query", "pos", "neg"):
                        if r.get(field):
                            yield r[field]
        if extra_texts:
            yield from extra_texts

    raw_tok = Tokenizer(BPE(unk_token="[UNK]"))
    raw_tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_BPE_SPECIAL_TOKENS,
        min_frequency=2,
    )
    print(f"Training BPE tokenizer on {len(corpus_paths)} corpus file(s)")
    raw_tok.train_from_iterator(_corpus_iterator(), trainer=trainer)

    save_path = Path(save_dir) / "bpe_tokenizer.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    raw_tok.save(str(save_path))

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(save_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    print(f"BPE tokenizer saved: {save_path}  (vocab_size={len(fast_tok):,})")
    return fast_tok


def load_tokenizer(tok_name: str = "data/processed/bpe_tokenizer.json"):
    """
    Load a tokenizer from a .json file or from HuggingFace by name or path
    """
    p = Path(tok_name)
    if p.suffix == ".json":
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tok_name}. Build it first")
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        if "model" in meta:
            return PreTrainedTokenizerFast(
                tokenizer_file=tok_name,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
            )
        return WordTokenizer.load(tok_name)

    print(f"Loading HuggingFace tokenizer: {tok_name}")
    return AutoTokenizer.from_pretrained(tok_name)


def build_vocab_from_triplets(
    triplets_path: str = "data/processed/train_triplets.jsonl",
    save_path: str = "data/processed/vocab.json",
    max_vocab_size: int = 30_000,
    min_freq: int = 2,
) -> WordTokenizer:
    """
    Read training triplets and build a WordTokenizer vocabulary from all text fields
    Saves the vocab to save_path and returns the fitted tokenizer
    """
    texts = []
    with open(triplets_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["query"])
            texts.append(r["pos"])
            texts.append(r["neg"])

    print(f"Building vocab from {len(texts):,} text fields in {triplets_path}")
    tok = WordTokenizer()
    tok.build_vocab(texts, max_vocab_size=max_vocab_size, min_freq=min_freq)
    tok.save(save_path)
    return tok
