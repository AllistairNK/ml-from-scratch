"""Gaussian Naive Bayes usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

GaussianNaiveBayes = load_impl(__file__).GaussianNaiveBayes

# Three classes with different means and spreads in 4-D. The labels are strings on purpose:
# NB returns the original labels, not 0..C-1.
rng = np.random.default_rng(3)
specs = {"setosa": (0.0, 0.5), "versicolor": (2.0, 1.0), "virginica": (4.0, 0.7)}
X = np.vstack([rng.normal(mu, sd, size=(120, 4)) for mu, sd in specs.values()])
y = np.repeat(list(specs), 120)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
X_train, X_test, y_train, y_test = X[:270], X[270:], y[:270], y[270:]

model = GaussianNaiveBayes().fit(X_train, y_train)
print("classes:", model.classes)
print("priors: ", np.round(model.priors, 3))
print("means (feature 0 per class):", np.round(model.means[:, 0], 2), " expected about [0, 2, 4]")
print("std   (feature 0 per class):", np.round(np.sqrt(model.vars[:, 0]), 2), " expected about [0.5, 1.0, 0.7]")

acc = np.mean(model.predict(X_test) == y_test)
print(f"test accuracy: {acc:.3f}")

# Posterior probabilities for a clear point and an ambiguous one.
pts = np.array([[0.1, -0.2, 0.0, 0.3], [3.0, 3.0, 3.0, 3.0]])
print("P(class | x):\n", np.round(model.predict_proba(pts), 3))
print("predictions:", model.predict(pts))

# Log space matters: with 500 features, the raw likelihood product underflows to 0.
X_wide = rng.normal(size=(50, 500))
y_wide = np.array([0, 1] * 25)
wide = GaussianNaiveBayes().fit(X_wide, y_wide)
print("500-feature probabilities still finite:", np.isfinite(wide.predict_proba(X_wide)).all())

assert acc > 0.9
assert model.predict(pts)[0] == "setosa"
assert np.allclose(model.predict_proba(X_test).sum(axis=1), 1.0)
assert np.isfinite(wide.predict_proba(X_wide)).all()
print("All checks passed.")
