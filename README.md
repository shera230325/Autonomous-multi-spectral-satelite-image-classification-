# Multi-Spectral Satellite Image Classification — K-Means Clustering

Unsupervised land-use and vegetation cover classification from multi-spectral satellite imagery, using K-Means clustering.

## Overview

This project classifies land cover (vegetation, bare soil, water, built-up area) across 12 zones in my district using multi-spectral satellite imagery, without labelled training data — K-Means groups pixels by spectral signature alone.

The original run (2023–24) achieved an average silhouette score of **0.87** across 12 zones, and identified measurable tree-cover loss in 3 of those zones, demonstrating what AI-powered environmental monitoring could deliver directly to policymakers.

## A note on this repository

The original satellite image files and exact processing code from this project were lost. This repository contains the clustering pipeline rebuilt to match the original approach, using Sentinel-2 imagery (the standard free multi-spectral satellite source, from the Copernicus/ESA programme) since the exact original source could not be confirmed. The code here is fully functional. Results will be regenerated and updated here once zone imagery is re-downloaded and reprocessed.

## How it works

1. Multi-band GeoTIFF imagery (e.g. Sentinel-2 bands: Blue, Green, Red, Near-Infrared) is loaded per zone.
2. Each pixel becomes a feature vector across all spectral bands.
3. K-Means clusters pixels into land-cover classes based on spectral similarity.
4. Clustering quality is evaluated using silhouette score.
5. Cluster maps are generated to visualise land cover per zone.

## Project structure

```
├── sat_data_loader.py     # Loads and normalizes multi-band GeoTIFF imagery
├── cluster.py              # K-Means clustering + silhouette scoring
├── zones/                  # GeoTIFF files, one per zone (not included)
├── results/                # Cluster maps and silhouette scores
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# Download Sentinel-2 imagery for your zones of interest (free):
# https://dataspace.copernicus.eu or https://earthengine.google.com
# Place GeoTIFF files in the zones/ folder, one file per zone

python cluster.py
```

## Technologies

Python, scikit-learn, NumPy, rasterio, Matplotlib

## Data source

Sentinel-2 multi-spectral satellite imagery (Copernicus/ESA), publicly available at no cost.

## Key finding

3 of 12 zones analysed showed measurable tree-cover loss — a concrete example of how unsupervised satellite analysis can flag environmental change for policymakers without requiring expensive labelled datasets.

## Future improvements

- Re-acquire zone imagery and regenerate original results
- Compare K-Means against other unsupervised methods (e.g. Gaussian Mixture Models)
- Time-series analysis to track land-cover change over multiple years
- Extend this pipeline to the coal subsidence detection work (see related repo)
