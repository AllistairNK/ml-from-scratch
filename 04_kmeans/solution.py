import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-6, seed=0):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.centroids = X[rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.max_iters):
            labels = self.predict(X)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break
        self.labels_ = self.predict(X)
        self.inertia_ = np.sum((X - self.centroids[self.labels_]) ** 2)
        return self

    def predict(self, X):
        dists = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(dists, axis=1)
