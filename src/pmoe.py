import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Sparsemax  
class Sparsemax(nn.Module):
    def forward(self, logits: Tensor) -> Tensor:
        z_sorted, _ = torch.sort(logits, dim=-1, descending=True)
        K = logits.size(-1)
        cumsum = torch.cumsum(z_sorted, dim=-1)
        k_range = torch.arange(1, K + 1, device=logits.device, dtype=logits.dtype)
        is_gt = z_sorted > (cumsum - 1) / k_range
        k_max = torch.max(is_gt * k_range, dim=-1, keepdim=True)[0]
        threshold = (torch.gather(cumsum, -1, k_max - 1) - 1) / k_max.to(logits.dtype)
        return torch.relu(logits - threshold)

class MLPGating(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_experts, dropout):
        super().__init__()
        # Input e features + lat/lon (conforme o paper do MoE original)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        self.sparsemax = Sparsemax()

    def forward(self, x, coords):
        # concatenar features locais com coords espaciais
        # coords devem tar normalizadas entre -1 e 1 senao o gating morre
        gating_input = torch.cat([x, coords], dim=-1)
        return self.sparsemax(self.net(gating_input))

class PMoES(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ... expert pool setup ...
        # (meto o gating e os experts aqui p/ ser self-contained)
        pass 

    def forward(self, x, coords):
        # TODO: implementar o forward igual ao da GNN mas s/ o edge_index
        # isto e so p/ a fase 1 de testes rapidos
        pass