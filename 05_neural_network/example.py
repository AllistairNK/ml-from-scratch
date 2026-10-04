"""Neural Network usage example and sanity checks (including a numerical gradient check).

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

MLP = load_impl(__file__).MLP

# Three interleaved spirals: no straight line separates them, so a linear model cannot fit this.
rng = np.random.default_rng(0)
n_per_class, n_classes = 100, 3
X = np.zeros((n_per_class * n_classes, 2))
y = np.repeat(np.arange(n_classes), n_per_class)
for c in range(n_classes):
    r = np.linspace(0.0, 1, n_per_class)
    t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, 0.2, n_per_class)
    X[c * n_per_class:(c + 1) * n_per_class] = np.c_[r * np.sin(t), r * np.cos(t)]

# 1) Gradient check: compare backward() with finite differences on a small model.
small = MLP(n_in=2, n_hidden=5, n_out=3, seed=1)
Xs, ys = X[5::30], y[5::30]  # skip X[0] = (0, 0), which sits exactly on the ReLU kink
small.loss(Xs, ys)
analytic = small.backward(ys)
eps, worst = 1e-5, 0.0
for name in ["W1", "b1", "W2", "b2"]:
    param = getattr(small, name)
    numeric = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + eps
        plus = small.loss(Xs, ys)
        param[i] = old - eps
        minus = small.loss(Xs, ys)
        param[i] = old
        numeric[i] = (plus - minus) / (2 * eps)
    rel = np.abs(numeric - analytic[name]).max() / max(np.abs(numeric).max(), 1e-8)
    worst = max(worst, rel)
    print(f"grad check {name}: max relative error {rel:.2e}")

# 2) Train on the spirals.
model = MLP(n_in=2, n_hidden=64, n_out=3, lr=1.0, seed=0).fit(X, y, epochs=5000)
acc = np.mean(model.predict(X) == y)
print(f"loss {model.losses[0]:.3f} (= ln 3 ~ 1.099 at start) -> {model.losses[-1]:.3f}")
print(f"training accuracy: {acc:.3f}")

# 3) Predict new points: each row of forward() is a probability distribution over the classes.
new_points = np.array([[0.0, 0.0], [0.5, -0.5]])
print("class probabilities:\n", np.round(model.forward(new_points), 3))
print("predicted classes:", model.predict(new_points))

assert worst < 1e-4, "backward() does not match the numerical gradient"
assert np.allclose(model.forward(new_points).sum(axis=1), 1.0)
assert acc > 0.95
print("All checks passed.")
