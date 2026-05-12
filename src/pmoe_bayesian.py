import warnings
import torch
import torch.nn as nn
from torch import Tensor
from src.pmoe import MLPGating, Sparsemax
from src.bayesian_experts import BayesianExpertPool

class PMoESBayesian(nn.Module):
    # Versao Bayesiana (PMoE-S-B)
    # Gating e deterministico mas os experts teem pesos probabilisticos (VI)
    def __init__(self, cfg: dict):
        super().__init__()
        self.k = cfg['model']['num_experts']
        
        self.gating = MLPGating(
            in_dim=cfg['model']['in_dim'],
            hidden_dim=cfg['model']['gate_hidden'],
            num_experts=self.k,
            dropout=cfg['model']['gate_dropout']
        )
        
        # Pool de experts Bayesianos
        self.experts = BayesianExpertPool(
            in_dim=cfg['model']['in_dim'],
            hidden_dim=cfg['model']['exp_hidden'],
            n_layers=cfg['model']['exp_layers'],
            num_experts=self.k,
            dropout=cfg['model']['exp_dropout'],
            prior_var=cfg['model'].get('prior_var', 1.0),
            mu_init=cfg['model'].get('mu_init', 3.3)
        )

    def forward(self, x: Tensor, coords: Tensor, n_samples: int = 1) -> dict:
        # o pi (routing) e calculado uma vez so
        pi = self.gating(x, coords) 

        if n_samples == 1:
            # Passagem rapida p/ treino (ELBO)
            mu, sigma2 = self.experts(x)
            kl = self.experts.kl_divergence()
            return {"pi": pi, "mu": mu, "sigma2": sigma2, "kl": kl}

        # MC Sampling p/ inferencia (estimar incerteza epistemica)
        all_means, all_vars = [], []
        for _ in range(n_samples):
            mu_s, sigma2_s = self.experts(x)
            all_means.append(mu_s)
            all_vars.append(sigma2_s)

        # TODO: otimizar este stack, ta a comer mta memoria c/ muitos samples
        means_stack = torch.stack(all_means, dim=0) # (T, N, K)
        vars_stack  = torch.stack(all_vars,  dim=0)

        # E[sum pi * mu] -> Media preditiva final
        mixture_means = (pi.unsqueeze(0) * means_stack).sum(-1)
        pred_mean = mixture_means.mean(0, keepdim=True).T

        # Incerteza Aleatoria (media das variancias)
        var_ale = (pi.unsqueeze(0) * vars_stack).sum(-1).mean(0, keepdim=True).T

        # Incerteza Epistemica (variancia das medias)
        # HACK: usar variancia nao enviesada p/ T pequeno
        var_epi = mixture_means.var(0, unbiased=True, keepdim=True).T

        return {
            "pi": pi,
            "pred_mean": pred_mean,
            "pred_var": var_ale + var_epi,
            "var_ale": var_ale,
            "var_epi": var_epi
        }