import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Sparsemax ─────────────────────────────────────────────────────
class Sparsemax(nn.Module):
    # Implementaçao do prof andre martins (ICML 2016)
    # Isto e melhor q softmax pq poe pesos a zero msm, o q ajuda na coerencia espacial
    def forward(self, logits: Tensor) -> Tensor:
        z_sorted, _ = torch.sort(logits, dim=-1, descending=True)
        K = logits.size(-1)
        cumsum = torch.cumsum(z_sorted, dim=-1)
        k_range = torch.arange(1, K + 1, device=logits.device, dtype=logits.dtype)
        
        # thresholding p/ ver quantos experts ficam ativos
        is_gt = z_sorted > (cumsum - 1) / k_range
        k_max = torch.max(is_gt * k_range, dim=-1, keepdim=True)[0]
        
        # FIXME: as vezes o k_max da problemas c/ fp16, ver dps
        threshold = (torch.gather(cumsum, -1, k_max - 1) - 1) / k_max.to(logits.dtype)
        return torch.relu(logits - threshold)

class EdgeAwareConv(nn.Module):
    # Uma especie de GraphSAGE simples mas q olha p/ a distancia das arestas
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__()
        self.msg = nn.Linear(in_dim + edge_dim, out_dim)
        
    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        # concatenar features do nó c/ os atributos da aresta (dist, etc)
        m = self.msg(torch.cat([h[src], edge_attr], dim=-1))
        
        # scatter add p/ agregar mensagens no destino
        out = torch.zeros_like(h)
        out.index_add_(0, dst, m)
        return out

class GNNGating(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, num_layers, num_experts, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        # Projecçao inicial das features do solo
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Empilhar camadas de GNN p/ propagar info espacial
        self.conv_layers = nn.ModuleList([
            EdgeAwareConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.routing_head = nn.Linear(hidden_dim, num_experts)
        self.sparsemax = Sparsemax()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        h = self.input_proj(x)

        for conv in self.conv_layers:
            # Skip connections p/ nao perder a info local nas camadas profundas
            h_new = conv(h, edge_index, edge_attr)
            h = h + F.dropout(h_new, p=self.dropout, training=self.training)
            h = F.gelu(h)

        logits = self.routing_head(h)
        pi = self.sparsemax(logits) # pesos de routing finais
        return pi, h