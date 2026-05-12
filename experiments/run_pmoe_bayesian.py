import sys, yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import get_lucas_native_features
from src.pmoe_bayesian import PMoESBayesian
from src.pmoe_bayesian_loss import PMoESBayesianLoss

def train():
    # TODO: o warm-up do KL e super sensivel. 
    # Se o beta subir mto rapido os experts colapsam p/ o prior.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # carregar config e dados
    with open("experiments/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data"]["processed_path"])
    fcols = get_lucas_native_features(df)
    
    model = PMoESBayesian(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = PMoESBayesianLoss(n_data=len(df))

    log_oc = np.log1p(df["OC"].values)
    
    # loop de treino simplificado p/ o paper
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        # aqui o beta do KL aumenta gradualmente (KL annealing)
        beta = min(1.0, epoch / 100.0) 
        criterion.beta = beta
        
        # forward pass c/ 1 sample (ELBO)
        # FIXME: ver se o batch size de 1024 nao ta a causar instabilidade no gradiente
        optimizer.zero_grad()
        out = model(x_batch, coords_batch, n_samples=1)
        loss_dict = criterion(y_batch, out, edge_index, dist_km)
        loss_dict["loss"].backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss_dict['loss'].item():.4f} (KL: {loss_dict['kl']:.4f})")

    # Guardar os pesos p/ usar nas figuras dps
    torch.save(model.state_dict(), "data/models/pmoe_bayesian_final.pt")

if __name__ == "__main__":
    train()