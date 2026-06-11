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

    def forward(self, query_embs: torch.Tensor, pos_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embs: (B, D) L2-normalized query embeddings
            pos_embs:   (B, D) L2-normalized positive passage embeddings
        Returns:
            scalar loss
        """
        sim = torch.matmul(query_embs, pos_embs.T) / self.temperature
        labels = torch.arange(len(query_embs), device=sim.device)
        loss_q = self.ce(sim, labels)
        loss_p = self.ce(sim.T, labels)
        return (loss_q + loss_p) / 2
