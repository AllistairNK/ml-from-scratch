# K-Means clustering (Lloyd's algorithm), explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Repeat until the centroids stop moving:
#   1. Assignment step: give each point the label of its nearest centroid.
#   2. Update step:     move each centroid to the mean of the points assigned to it.
# Each step can only lower the inertia (the sum of squared distances to the assigned centroid),
# so the algorithm always converges, but only to a local minimum. That is why initialisation matters.

import numpy as np


class KMeans:
    # k clusters; stop after max_iters or once the centroids move less than tol; seed fixes the init.
    def __init__(self, k=3, max_iters=100, tol=1e-6, seed=0):
        # Number of clusters (you must choose it, e.g. with the elbow method).
        self.k = k
        # Safety cap on iterations.
        self.max_iters = max_iters
        # Convergence threshold on the total centroid movement.
        self.tol = tol
        # Random seed, so runs are reproducible.
        self.seed = seed

    # Unsupervised: fit only takes X, no labels.
    def fit(self, X):
        # A local random generator (better practice than seeding the global np.random).
        rng = np.random.default_rng(self.seed)
        # Initialise the centroids as k distinct random data points. Shape (k, n_features).
        # (k-means++ is the smarter init: pick each new centroid far from the existing ones.)
        self.centroids = X[rng.choice(len(X), self.k, replace=False)]
        # Lloyd iterations.
        for _ in range(self.max_iters):
            # Assignment step: index of the nearest centroid for each point. Shape (n_samples,).
            labels = self.predict(X)
            # Update step: the new centroid j is the mean of the points labelled j.
            new_centroids = np.array([
                # If a cluster ends up empty, keep its old centroid (otherwise mean() returns nan).
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                # One row per cluster -> shape (k, n_features).
                for j in range(self.k)
            ])
            # How far the centroids moved this iteration (Frobenius norm of the difference).
            shift = np.linalg.norm(new_centroids - self.centroids)
            # Commit the update.
            self.centroids = new_centroids
            # Converged: the assignments will not change any more.
            if shift < self.tol:
                # Stop early.
                break
        # Final labels for the training data (sklearn-style trailing underscore = learned attribute).
        self.labels_ = self.predict(X)
        # Inertia: total squared distance from each point to its own centroid. Used for the elbow method.
        self.inertia_ = np.sum((X - self.centroids[self.labels_]) ** 2)
        # Allow chaining.
        return self

    # Assign each row of X to its nearest centroid (used inside fit and on new data).
    def predict(self, X):
        # Squared Euclidean distances, (n, 1, f) - (1, k, f) -> (n, k, f) -> sum -> (n, k).
        # No sqrt is needed: it does not change which centroid is the closest.
        dists = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        # Column index of the smallest distance = cluster id.
        return np.argmin(dists, axis=1)
