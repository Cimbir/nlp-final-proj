import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss
    
    Args:
        temperature : softmax temperature 
                      lower = sharper distribution = harder negatives
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        query_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query_embs: (B, D) L2-normalized query embeddings
            pos_embs:   (B, D) L2-normalized positive passage embeddings
            neg_embs:   (B, D) L2-normalized hard negative embeddings (optional)
        Returns:
            scalar loss
        """
        labels = torch.arange(len(query_embs), device=query_embs.device)

        if neg_embs is not None:
            all_passages = torch.cat([pos_embs, neg_embs], dim=0)
            sim_q = torch.matmul(query_embs, all_passages.T) / self.temperature
            loss_q = self.ce(sim_q, labels)
            sim_p = torch.matmul(pos_embs, query_embs.T) / self.temperature
            loss_p = self.ce(sim_p, labels)
        else:
            sim = torch.matmul(query_embs, pos_embs.T) / self.temperature
            loss_q = self.ce(sim, labels)
            loss_p = self.ce(sim.T, labels)

        return (loss_q + loss_p) / 2
