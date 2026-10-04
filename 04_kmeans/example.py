"""K-Means usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

KMeans = load_impl(__file__).KMeans

# Three well-separated 2-D blobs. We keep the true labels only to grade the clustering.
rng = np.random.default_rng(7)
true_centers = np.array([[0, 0], [6, 0], [3, 5]])
X = np.vstack([rng.normal(c, 0.8, size=(150, 2)) for c in true_centers])
true_labels = np.repeat([0, 1, 2], 150)

model = KMeans(k=3, seed=0).fit(X)
print("found centroids:\n", np.round(model.centroids[np.lexsort(model.centroids.T[::-1])], 2))
print("true centers:\n", true_centers[np.lexsort(true_centers.T[::-1])])

# Cluster ids are arbitrary (cluster 0 is not necessarily blob 0), so measure purity:
# for each found cluster, the fraction of its points that come from its most common true blob.
purity = sum(np.bincount(true_labels[model.labels_ == j]).max() for j in range(3)) / len(X)
print(f"purity: {purity:.3f}")

# Elbow method: inertia always falls as k grows, but the drop flattens after the true k.
print("k  inertia")
for k in range(1, 7):
    print(f"{k}  {KMeans(k=k, seed=0).fit(X).inertia_:9.1f}")

# Assign new points to the learned clusters.
new_points = np.array([[0.1, 0.2], [5.9, -0.3]])
new_labels = model.predict(new_points)
print("new points go to clusters", new_labels)

assert purity > 0.98
assert new_labels[0] != new_labels[1]
assert KMeans(k=3, seed=0).fit(X).inertia_ < KMeans(k=2, seed=0).fit(X).inertia_
print("All checks passed.")
