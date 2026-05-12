import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class BayesianLinear(nn.Module):
    # MLP layer c/ incerteza nos pesos (VI)
    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parametros do posterior q(W)
        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
        # rho em vez de sigma p/ garantir positividade dps do softplus
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3, -3))
        
        self.register_buffer('prior_var', torch.tensor(prior_var))

    def forward(self, x):
        sigma = F.softplus(self.rho_weight) + 1e-6
        eps = torch.randn_like(sigma)
        w = self.mu_weight + sigma * eps # reparametrization trick
        return F.linear(x, w)

    def kl_divergence(self):
        # KL entre q(theta) e p(theta) - assumindo prior Gaussiano fixo
        sigma = F.softplus(self.rho_weight) + 1e-6
        var = sigma**2
        # TODO: isto ta a assumir mean zero no prior, confirmar se e o melhor p/ SOC
        kl = 0.5 * (torch.log(self.prior_var / var) + (var + self.mu_weight**2) / self.prior_var - 1).sum()
        return kl

class BayesianExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, dropout, prior_var, mu_init):
        super().__init__()
        layers = []
        curr = in_dim
        for _ in range(n_layers):
            layers.append(BayesianLinear(curr, hidden_dim, prior_var))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            curr = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        self.mu_head = BayesianLinear(hidden_dim, 1, prior_var)
        self.sigma_head = nn.Linear(hidden_dim, 1) # aleatoric noise e deterministico aqui

        # init com a media do log-SOC p/ nao comecar do zero
        nn.init.constant_(self.mu_head.mu_weight, 0.0)
        # HACK: forcar bias positivo p/ evitar predicoes negativas de SOC no inicio
        nn.init.constant_(self.mu_head.mu_weight.data, mu_init) 

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        # softplus p/ variancia ser positiva. +1e-4 p/ estabilidade
        sigma2 = F.softplus(self.sigma_head(h)) + 1e-4
        return mu, sigma2