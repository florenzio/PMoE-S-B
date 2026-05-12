# Data

## LUCAS 2018 Topsoil Dataset
Download from ESDAC (free academic registration):
https://esdac.jrc.ec.europa.eu/content/lucas2018-topsoil-data

Place at: data/raw/LUCAS_Topsoil_2018.csv

## Sentinel-2 Features
Extracted via Google Earth Engine.
Parameters: B2-B12, NDVI, NDWI, BSI, April-October 2018,
            cloud fraction < 20%, 20 m resolution, median composite.

## CHELSA V2.1 Climate Variables
Download from: https://chelsa-climate.org/downloads/
Files needed:
  CHELSA_bio1_1981-2010_V.2.1.tif   (MAT)
  CHELSA_bio12_1981-2010_V.2.1.tif  (MAP)

## Processed Feature Matrix
After downloading all sources, run:
    python src/data_loading.py

This produces: data/processed/lucas_sentinel_chelsa.csv
(18711 rows x 86 columns)

## Pretrained Weights
Download from Zenodo: https://zenodo.org/records/20129313
    mv best_pmoe_bayesian.pt data/
