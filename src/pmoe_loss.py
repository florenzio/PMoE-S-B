import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def nll_mixture_gaussian(y, mu, sigma2, pi):
    # Log-likelihood p/ cada expert indvidual
    log_prob = (
        -0.5 * math.log(2 * math.pi)
        - 0.5 * torch.log(sigma2)
        - 0.5 * (y - mu) ** 2 / sigma2
    )

    # Sparsemax da valores exatos de zero. 
    # Temos de mascarar p/ o log nao dar -inf / nan
    mask = (pi > 0).float()
    log_prob = log_prob * mask + (1 - mask) * (-1e10) # mto pequeno p/ ignorar no exp
    
    log_pi = torch.log(pi.clamp(min=1e-10))
    # log-sum-exp trick p/ estabilidade numerica
    return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()

def load_balancing_loss(pi):
    # L_bal: p/ evitar q o modelo use so 1 experto p/ tudo (expert collapse)
    K = pi.size(-1)
    mean_pi = pi.mean(0)
    target = torch.ones(K, device=pi.device) / K
    return F.mse_loss(mean_pi, target)

def spatial_coherence_loss(pi, edge_index, dist_km):
    # A nossa L_spa (Dirichlet energy)
    # se o ponto i e j sao vizinhos, o pi(i) e pi(j) teem de ser parecidos
    src, dst = edge_index[0], edge_index[1]
    # penalizar diferenca ao quadrado, pesando pela distacia (vizinhos proximos pesam +)
    diff = ((pi[src] - pi[dst]) ** 2).sum(dim=-1)
    return (diff / (dist_km + 1.0)).mean()

class PMoESLoss(nn.Module):
    def __init__(self, l_bal=0.05, l_spa=0.005):
        super().__init__()
        self.l_bal = l_bal
        self.l_spa = l_spa

    def forward(self, y, out, edge_index, dist_km):
        nll = nll_mixture_gaussian(y, out["mu"], out["sigma2"], out["pi"])
        bal = load_balancing_loss(out["pi"])
        spa = spatial_coherence_loss(out["pi"], edge_index, dist_km)
        
        # Loss final p/ o gradiente
        total = nll + self.l_bal * bal + self.l_spa * spa
        
        return {"loss": total, "nll": nll, "spa": spa}