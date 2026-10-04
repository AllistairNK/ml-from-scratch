"""Logistic Regression usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

impl = load_impl(__file__)
LogisticRegression, sigmoid = impl.LogisticRegression, impl.sigmoid

# Two overlapping 2-D Gaussian blobs: class 0 around (-1, -1), class 1 around (1, 1).
# For equal-variance Gaussians the ideal answer is w = (mu1 - mu0) / var = [2, 2] and b = 0.
rng = np.random.default_rng(3)
X = np.vstack([rng.normal(-1, 1.0, size=(200, 2)), rng.normal(1, 1.0, size=(200, 2))])
y = np.repeat([0, 1], 200)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
X_train, X_test, y_train, y_test = X[:300], X[300:], y[:300], y[300:]

model = LogisticRegression(lr=0.1, n_iters=2000).fit(X_train, y_train)
print("weights:", np.round(model.w, 3), " bias:", round(model.b, 3), " (ideal: [2, 2], 0)")
print(f"loss: {model.losses[0]:.4f} (= ln 2 at start) -> {model.losses[-1]:.4f}")

acc = np.mean(model.predict(X_test) == y_test)
print(f"test accuracy: {acc:.3f}")

# Probabilities for a few points: deep in class 0, on the boundary, deep in class 1.
pts = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
print("P(y=1):", np.round(model.predict_proba(pts), 3))

# Moving the threshold trades precision for recall.
for t in [0.3, 0.5, 0.7]:
    pred = model.predict(X_test, threshold=t)
    tp = np.sum((pred == 1) & (y_test == 1))
    precision = tp / max(pred.sum(), 1)
    recall = tp / y_test.sum()
    print(f"threshold={t}: precision={precision:.3f} recall={recall:.3f}")

# sigmoid must not overflow or return nan at extreme inputs.
with np.errstate(over="raise"):
    extremes = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
print("sigmoid([-1000, 0, 1000]) =", extremes)

assert np.isclose(model.losses[0], np.log(2))
assert model.losses[-1] < model.losses[0]
assert acc > 0.85
assert np.allclose(extremes, [0.0, 0.5, 1.0])
print("All checks passed.")
