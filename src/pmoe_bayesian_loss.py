import math
import torch
import torch.nn.functional as F
from src.pmoe_loss import load_balancing_loss, spatial_coherence_loss

class PMoESBayesianLoss(torch.nn.Module):
    def __init__(self, lambda_bal=0.05, lambda_spa=0.002, beta=1.0, n_data=1):
        super().__init__()
        self.lambda_bal = lambda_bal
        self.lambda_spa = lambda_spa
        self.beta = beta # p/ o KL warm-up
        self.n_data = n_data # preciso disto p/ fazer o scale do KL p/ cada batch

    def forward(self, y, out, edge_index, dist_km):
        # nll so com 1 sample do peso (MC training)
        l_nll = nll_mixture_gaussian(y, out["mu"], out["sigma2"], out["pi"])
        
        l_bal = load_balancing_loss(out["pi"])
        l_spa = spatial_coherence_loss(out["pi"], edge_index, dist_km)
        
        # KL scaling - crucial p/ o ELBO estar certo
        l_kl = out["kl"] / self.n_data
        
        # TODO: testar se beta > 1 ajuda a regularizar mais os experts
        total = l_nll + self.lambda_bal * l_bal + self.lambda_spa * l_spa + self.beta * l_kl
        
        return {"loss": total, "nll": l_nll, "kl": l_kl, "spa": l_spa}