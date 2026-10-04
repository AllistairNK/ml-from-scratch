"""KNN usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

KNN = load_impl(__file__).KNN

# Three 2-D Gaussian blobs, one per class.
rng = np.random.default_rng(42)
centers = np.array([[0, 0], [5, 5], [0, 5]])
X = np.vstack([rng.normal(c, 1.0, size=(100, 2)) for c in centers])
y = np.repeat([0, 1, 2], 100)

# Shuffle, then split 80/20 into train and test.
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
X_train, X_test, y_train, y_test = X[:240], X[240:], y[:240], y[240:]

for k in [1, 3, 5, 15]:
    model = KNN(k=k).fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)
    print(f"k={k:<2}  test accuracy = {acc:.3f}")

# Classify single new points (predict always takes a 2-D array).
model = KNN(k=5).fit(X_train, y_train)
new_points = np.array([[0.2, -0.1], [4.8, 5.3], [-0.5, 4.9]])
print("predictions for", new_points.tolist(), "->", model.predict(new_points))

assert model.predict(new_points).tolist() == [0, 1, 2]
assert np.mean(model.predict(X_test) == y_test) > 0.9
print("All checks passed.")
