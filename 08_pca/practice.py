# PCA practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


class PCA:
    def __init__(self, n_components):
        # 1. Store n_components.
        raise NotImplementedError("write __init__")

    def fit(self, X):
        # 1. self.mean = the per-feature mean of X.
        # 2. X_centered = X minus the mean.
        # 3. cov = the sample covariance matrix (f x f), dividing by n - 1.
        # 4. eigvals, eigvecs = np.linalg.eigh(cov).
        # 5. order = the indices that sort eigvals in DESCENDING order.
        # 6. Reorder eigvals, and the COLUMNS of eigvecs, by order.
        # 7. self.components = the top n_components eigenvectors as rows, shape (k, f).
        # 8. self.explained_variance = the top n_components eigenvalues.
        # 9. self.explained_variance_ratio = explained_variance / the sum of all eigenvalues.
        # 10. Return self.
        raise NotImplementedError("write fit")

    def transform(self, X):
        # 1. Return the centered X projected onto the components -> (n, k).
        raise NotImplementedError("write transform")

    def inverse_transform(self, Z):
        # 1. Return Z mapped back through the components, plus the mean -> (n, f).
        raise NotImplementedError("write inverse_transform")
