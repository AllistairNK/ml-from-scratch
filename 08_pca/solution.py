import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        cov = X_centered.T @ X_centered / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        self.components = eigvecs[:, :self.n_components].T
        self.explained_variance = eigvals[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / eigvals.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean
