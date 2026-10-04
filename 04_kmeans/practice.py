# K-Means practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-6, seed=0):
        # 1. Store k.
        # 2. Store max_iters.
        # 3. Store tol.
        # 4. Store seed.
        raise NotImplementedError("write __init__")

    def fit(self, X):
        # 1. rng = a NumPy random Generator seeded with self.seed.
        # 2. self.centroids = k distinct random rows of X (rng.choice without replacement).
        # 3. Loop max_iters times:
        # 4.     labels = the nearest centroid for each point (call self.predict).
        # 5.     new_centroids = for each cluster j, the mean of its points
        #        (keep the old centroid if the cluster is empty). Shape (k, n_features).
        # 6.     shift = the norm of (new_centroids - self.centroids).
        # 7.     Assign new_centroids to self.centroids.
        # 8.     If shift < tol:
        # 9.         break
        # 10. self.labels_ = the final assignment of X.
        # 11. self.inertia_ = the sum of squared distances from each point to its assigned centroid.
        # 12. Return self.
        raise NotImplementedError("write fit")

    def predict(self, X):
        # 1. dists = squared distance from every point to every centroid, shape (n, k) (broadcast with None).
        # 2. Return the argmin over the centroid axis.
        raise NotImplementedError("write predict")
