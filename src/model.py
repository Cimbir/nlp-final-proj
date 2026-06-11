import math
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Bi-encoder backbone built entirely from PyTorch primitives
    Shares weights for query and passage encoding

    Output: L2-normalized embeddings of shape (batch, d_model)
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.token_emb.weight[0], 0.0)  # padding idx = 0

    def _mean_pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        padding_mask = attention_mask == 0
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self._mean_pool(x, attention_mask)
        return F.normalize(x, p=2, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list, tokenizer, batch_size: int = 64,
               max_len: int = 256, device: str = "cpu") -> "np.ndarray":
        """Encode a list of strings to a numpy array of shape (N, d_model)"""
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[i : i + batch_size],
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embs = self(input_ids, attention_mask).cpu().numpy()
            all_embs.append(embs)
        return np.vstack(all_embs)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
