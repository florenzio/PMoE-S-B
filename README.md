# PMoE-S-B — Spatially Coherent Mixture-of-Experts via Laplacian Routing Regularisation for Soil Organic Carbon Mapping 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Keywords: Mixture-of-experts (MoE), Spatial non-stationarity, Uncertainty quantification(UQ), Gaussian markov random field (GMRF), SOC mapping.

This repository contains the code for PMoE-S-B as described in:
https://arxiv.org/abs/2405.18953 

## Project Structure

```text
gnn-pmoe-soc/
├── data/
│   ├── raw/           # Raw LUCAS CSV (licensed data)
│   ├── processed/     # Cleaned features + Sentinel-2 covariates
│   ├── graphs/        # Serialized PyG (PyTorch Geometric) graphs
│   └── splits/        # Spatial block cross-validation indices
├── src/
│   ├── data_loading.py   # LUCAS dataset loading and cleaning
│   ├── graph_builder.py  # kNN spatial graph construction
│   ├── gnn_gating.py     # GNN + Sparsemax (core contribution)
│   ├── experts.py        # K Probabilistic MLP Experts
│   ├── model.py          # Integrated GNN-PMoE architecture
│   ├── loss.py           # NLL + load balancing + spatial coherence loss
│   ├── baselines.py      # RF and QRF implementations for comparison
│   └── utils.py          # Helper functions and logging utilities
└── experiments/
    ├── config.yaml       # Global hyperparameters and experiment settings
    ├── run_baseline.py   # Baseline model training script
    └── run_gnnpmoe.py    # GNN-PMoE training script
```

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, please follow the specific installation instructions for your environment at https://pyg.org/

## Data Acquisition

1. Request access to the LUCAS TOPSOIL dataset via the European Soil Data Centre (ESDAC) at https://esdac.jrc.ec.europa.eu/
2. Place the downloaded file at `data/raw/LUCAS_TOPSOIL_v2.csv`
3. Extract Sentinel-2 features using Google Earth Engine (see `notebooks/01_eda.ipynb`)

## Quick Start

Execute the pipeline in the following order:

```bash
# 1. Preprocess raw data and generate features
python src/data_loading.py experiments/config.yaml

# 2. Construct the spatial graph (k-Nearest Neighbors)
python src/graph_builder.py experiments/config.yaml

# 3. Train baseline models (RF, QRF)
python experiments/run_baseline.py experiments/config.yaml

# 4. Train the proposed PMoE-S-B model
python experiments/spatial_cv_bayesian.py experiments/config.yaml
```
## Citation

```bibtex
@misc{florencio2026pmoesb,
      title={Spatially Coherent Mixture-of-Experts via Laplacian Routing Regularisation for Soil Organic Carbon Mapping},
      author={Salvador Flor\^encio and Gustau Camps-Valls},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/},
}
```
