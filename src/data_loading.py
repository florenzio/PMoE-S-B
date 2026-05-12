import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

# ... logs config ...

def load_lucas(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    raw_path = Path(cfg['data']['raw_path'])
    # admitir csv ou excel conforme o que tiver na pasta
    if raw_path.suffix == '.csv':
        df = pd.read_csv(raw_path)
    else:
        df = pd.read_excel(raw_path)

    # so queremos a camada de topo (0-20cm)
    df = df[df['Depth'] == '0-20 cm']
    
    # normalizar nomes das colunas de coord
    df = df.rename(columns={'TH_LAT': 'lat', 'TH_LONG': 'lon'})
    
    # Limpeza de strings no SOC (LOD = Limit of Detection)
    # se for "< LOD", usamos metade do valor do limite
    def clean_val(v):
        if isinstance(v, str) and '<' in v:
            return float(v.split()[-1]) / 2
        try: return float(v)
        except: return np.nan

    df['OC'] = df['OC'].apply(clean_val)
    df = df.dropna(subset=['OC', 'lat', 'lon'])
    df = df[df['OC'] > 0] # SOC negativo nao existe

    # log transform pq a dist e muito skewed (cauda longa)
    df['log_oc'] = np.log1p(df['OC'])
    
    return df