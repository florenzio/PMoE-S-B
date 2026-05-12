import torch
import torch.nn as nn
from torch import Tensor
from src.gnn_gating import GNNGating
from src.experts import ExpertPool

class PMoESGNN(nn.Module):
    # Versao GNN - a ideia aqui e usar o contexto dos vizinhos 
    # p/ o routing ser mais "suave" no mapa
    def __init__(self, cfg, mu_init=3.3):
        super().__init__()
        
        # Gating network agora e uma GNN (GraphSAGE-like)
        self.gating = GNNGating(
            in_dim=cfg['model']['in_dim'],
            edge_dim=cfg['model']['edge_dim'],
            hidden_dim=cfg['model']['gate_hidden'],
            num_layers=cfg['model']['gate_layers'],
            num_experts=cfg['model']['num_experts'],
            dropout=cfg['model']['gate_dropout']
        )

        # os experts continuam a ser MLPs locais, nao precisam de ver os vizinhos
        self.experts = ExpertPool(
            in_dim=cfg['model']['in_dim'],
            hidden_dim=cfg['model']['exp_hidden'],
            n_layers=cfg['model']['exp_layers'],
            num_experts=cfg['model']['num_experts'],
            dropout=cfg['model']['exp_dropout']
        )

        # inicializar o bias da mu head c/ o log-mean do SOC p/ convergir + rapido
        # FIXME: ver se isto nao enviesa mto os experts no inicio do treino
        for expert in self.experts.experts:
            nn.init.constant_(expert.mu_head.bias, mu_init)
            nn.init.zeros_(expert.mu_head.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> dict:
        # pi_i = routing espacialmente aware
        pi, _ = self.gating(x, edge_index, edge_attr)

        # calcular mu e sigma p/ cada expert (local features only)
        mus, vars_ = zip(*[e(x) for e in self.experts.experts])
        mu     = torch.cat(mus,   dim=-1)
        sigma2 = torch.cat(vars_, dim=-1)

        # media da mistura: sum pi_k * mu_k
        pred_mean = (pi * mu).sum(-1, keepdim=True)
        
        # decomposicao da variancia (lei da variancia total)
        var_ale   = (pi * sigma2).sum(-1, keepdim=True)
        var_epi   = (pi * mu**2).sum(-1, keepdim=True) - pred_mean**2
        
        return {
            "pi": pi, 
            "mu": mu, 
            "sigma2": sigma2,
            "pred_mean": pred_mean, 
            "pred_var": var_ale + var_epi,
            "var_ale": var_ale, 
            "var_epi": var_epi
        }