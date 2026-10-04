"""Linear Regression usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

LinearRegression = load_impl(__file__).LinearRegression

# Synthetic data with known parameters: y = 3*x0 - 2*x1 + 0.5*x2 + 4 + noise
rng = np.random.default_rng(0)
true_w, true_b = np.array([3.0, -2.0, 0.5]), 4.0
X = rng.normal(size=(500, 3))
y = X @ true_w + true_b + rng.normal(scale=0.3, size=500)

model = LinearRegression(lr=0.05, n_iters=1000).fit(X, y)
print("learned w:", np.round(model.w, 3), " true w:", true_w)
print("learned b:", round(model.b, 3), "        true b:", true_b)
print(f"loss: first step {model.losses[0]:.3f} -> last step {model.losses[-1]:.4f}")

# Compare with the exact least-squares answer (add a column of ones for the bias).
X_aug = np.hstack([X, np.ones((len(X), 1))])
closed_form = np.linalg.lstsq(X_aug, y, rcond=None)[0]
print("closed-form w, b:", np.round(closed_form, 3))

# Predict for new inputs.
X_new = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
print("predictions:", np.round(model.predict(X_new), 3), " expected about [7, 3]")

# Feature scaling matters for GD: with a huge feature scale, the same lr diverges.
with np.errstate(over="ignore", invalid="ignore"):
    bad = LinearRegression(lr=0.05, n_iters=50).fit(X * 100, y)
print("unscaled features, final loss:", bad.losses[-1], "(diverged; standardise your features)")

assert np.allclose(model.w, closed_form[:3], atol=1e-3)
assert abs(model.b - closed_form[3]) < 1e-3
assert model.losses[-1] < model.losses[0]
print("All checks passed.")
