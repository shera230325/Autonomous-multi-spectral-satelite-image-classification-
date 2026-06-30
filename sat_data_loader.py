"""
data_loader.py
Loads multi-spectral satellite imagery (Sentinel-2 GeoTIFF bands) and
prepares pixel-wise feature vectors for clustering.

Expected input: a folder of GeoTIFF files, one per zone, each containing
multiple spectral bands (e.g. B2-Blue, B3-Green, B4-Red, B8-NIR from Sentinel-2).

Sentinel-2 imagery can be downloaded for free from:
- Copernicus Open Access Hub (https://dataspace.copernicus.eu)
- Google Earth Engine (https://earthengine.google.com)
"""

import numpy as np
import rasterio


def load_zone_image(filepath):
    """
    Loads a multi-band GeoTIFF and returns a (num_pixels, num_bands) array
    ready for clustering, plus the original image shape for reconstruction.
    """
    with rasterio.open(filepath) as src:
        image = src.read()  # shape: (bands, height, width)

    num_bands, height, width = image.shape

    # Reshape to (num_pixels, num_bands) — one row per pixel
    pixels = image.reshape(num_bands, -1).T

    # Normalize each band to 0-1 range
    pixels = pixels.astype(np.float32)
    band_min = pixels.min(axis=0)
    band_max = pixels.max(axis=0)
    pixels = (pixels - band_min) / (band_max - band_min + 1e-8)

    return pixels, (height, width)


def load_all_zones(zone_filepaths):
    """
    Loads multiple zone images.

    Args:
        zone_filepaths: list of paths to GeoTIFF files, one per zone

    Returns:
        dict mapping zone name -> (pixels, image_shape)
    """
    zones = {}
    for filepath in zone_filepaths:
        zone_name = filepath.split("/")[-1].split(".")[0]
        zones[zone_name] = load_zone_image(filepath)
    return zones
