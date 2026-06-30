"""
cluster.py
Runs K-Means clustering on multi-spectral satellite imagery to classify
land cover (e.g. vegetation, bare soil, water, built-up area), then
evaluates clustering quality using silhouette score.

Usage:
    python cluster.py
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sat_data_loader import load_all_zones

NUM_CLUSTERS = 5  # e.g. water, bare soil, sparse vegetation, dense vegetation, built-up
ZONE_DATA_DIR = "zones"  # folder containing one GeoTIFF per zone
SILHOUETTE_SAMPLE_SIZE = 2000  # subsample for speed; silhouette score is O(n^2)


def cluster_zone(pixels, num_clusters=NUM_CLUSTERS, seed=42):
    """
    Runs K-Means on a single zone's pixel data and returns labels + score.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Silhouette score on a random subsample for speed, since computing it
    # on every pixel in a large image is very slow
    if len(pixels) > SILHOUETTE_SAMPLE_SIZE:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(pixels), SILHOUETTE_SAMPLE_SIZE, replace=False)
        score = silhouette_score(pixels[sample_idx], labels[sample_idx])
    else:
        score = silhouette_score(pixels, labels)

    return labels, score


def save_cluster_map(labels, image_shape, zone_name, output_dir="results"):
    cluster_map = labels.reshape(image_shape)
    plt.figure(figsize=(6, 6))
    plt.imshow(cluster_map, cmap="tab10")
    plt.title(f"Land cover clusters — {zone_name}")
    plt.colorbar(label="Cluster ID")
    plt.axis("off")
    plt.savefig(f"{output_dir}/{zone_name}_clusters.png", bbox_inches="tight")
    plt.close()


def main():
    zone_filepaths = sorted(glob.glob(f"{ZONE_DATA_DIR}/*.tif"))

    if not zone_filepaths:
        print(f"No .tif files found in '{ZONE_DATA_DIR}/'. "
              f"Add Sentinel-2 GeoTIFF files for each zone before running.")
        return

    zones = load_all_zones(zone_filepaths)

    scores = {}
    for zone_name, (pixels, image_shape) in zones.items():
        print(f"Clustering {zone_name}...")
        labels, score = cluster_zone(pixels)
        scores[zone_name] = score
        save_cluster_map(labels, image_shape, zone_name)
        print(f"  Silhouette score: {score:.3f}")

    avg_score = np.mean(list(scores.values()))
    print(f"\nAverage silhouette score across {len(scores)} zones: {avg_score:.3f}")

    with open("results/silhouette_scores.txt", "w") as f:
        for zone_name, score in scores.items():
            f.write(f"{zone_name}: {score:.3f}\n")
        f.write(f"\nAverage: {avg_score:.3f}\n")


if __name__ == "__main__":
    main()
