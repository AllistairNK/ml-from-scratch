# PCA via eigendecomposition of the covariance matrix, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Goal: find the orthogonal directions of maximum variance and project the data onto the top k of them.
#   1. Center the data (subtract the per-feature mean).
#   2. Covariance matrix C = Xc^T Xc / (n - 1)               (f x f, symmetric)
#   3. Eigenvectors of C = principal directions; eigenvalues = the variance along each direction.
#   4. Sort by eigenvalue, largest first, and keep the top k.
# (An alternative: SVD of Xc gives the same directions without forming C, which is more stable.)

import numpy as np


class PCA:
    # How many dimensions to keep.
    def __init__(self, n_components):
        # Store k.
        self.n_components = n_components

    def fit(self, X):
        # Per-feature mean. Shape (n_features,). It is needed again in transform/inverse_transform.
        self.mean = X.mean(axis=0)
        # Center the data. Without this, the first "component" would just point at the mean.
        X_centered = X - self.mean
        # Sample covariance matrix. (f,n) @ (n,f) -> (f,f). The (n-1) gives the unbiased estimate.
        cov = X_centered.T @ X_centered / (len(X) - 1)
        # eigh is for symmetric matrices: it returns real eigenvalues in ASCENDING order,
        # and the eigenvectors are the COLUMNS of eigvecs.
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Indices that sort the eigenvalues in descending order.
        order = np.argsort(eigvals)[::-1]
        # Reorder the eigenvalues, and the eigenvector columns to match.
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        # Keep the top k eigenvectors, stored as ROWS -> (k, n_features) (the same layout as sklearn).
        self.components = eigvecs[:, :self.n_components].T
        # The variance captured by each kept component (= its eigenvalue).
        self.explained_variance = eigvals[:self.n_components]
        # The fraction of the total variance that each component explains (use it to choose k).
        self.explained_variance_ratio = self.explained_variance / eigvals.sum()
        # Allow chaining.
        return self

    # Project onto the principal components: (n,f) @ (f,k) -> (n,k).
    def transform(self, X):
        # Center with the TRAINING mean, then take the dot product with each component.
        return (X - self.mean) @ self.components.T

    # Map back to the original space (lossy if k < n_features): (n,k) @ (k,f) -> (n,f).
    def inverse_transform(self, Z):
        # Rebuild from the components, then add the mean back.
        return Z @ self.components + self.mean
