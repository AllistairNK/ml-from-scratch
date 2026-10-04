"""PCA usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

PCA = load_impl(__file__).PCA

# 5-D data that really lives on a 2-D plane (plus a little noise), shifted away from the origin.
rng = np.random.default_rng(0)
latent = rng.normal(size=(400, 2)) * [3.0, 1.0]       # 2 hidden factors with different spreads
mixing = rng.normal(size=(2, 5))                       # embed them in 5-D
X = latent @ mixing + rng.normal(scale=0.05, size=(400, 5)) + 10.0

full = PCA(n_components=5).fit(X)
print("explained variance ratio (all 5):", np.round(full.explained_variance_ratio, 4))
print("cumulative:", np.round(np.cumsum(full.explained_variance_ratio), 4))

pca = PCA(n_components=2).fit(X)
Z = pca.transform(X)
X_rec = pca.inverse_transform(Z)
rec_err = np.mean((X - X_rec) ** 2)
print("projected shape:", Z.shape)
print(f"reconstruction MSE with 2 components: {rec_err:.5f}")

# Properties to know for interviews:
gram = pca.components @ pca.components.T
print("components are orthonormal (C C^T = I):\n", np.round(gram, 6))
print("variance of each projected column = explained_variance:",
      np.round(Z.var(axis=0, ddof=1), 3), np.round(pca.explained_variance, 3))
print("projected columns are uncorrelated, corr =", round(np.corrcoef(Z.T)[0, 1], 6))

# Cross-check against SVD of the centered data (directions match up to sign).
_, _, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
print("matches SVD directions (up to sign):", np.allclose(np.abs(Vt[:2]), np.abs(pca.components), atol=1e-6))

assert np.cumsum(full.explained_variance_ratio)[1] > 0.99
assert rec_err < 0.01
assert np.allclose(gram, np.eye(2))
assert np.allclose(Z.var(axis=0, ddof=1), pca.explained_variance)
assert np.allclose(np.abs(Vt[:2]), np.abs(pca.components), atol=1e-6)
print("All checks passed.")
