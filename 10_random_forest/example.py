"""Random Forest usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

impl = load_impl(__file__)
RandomForest, DecisionTree = impl.RandomForest, impl.DecisionTree

# A noisy problem where a single deep tree overfits: a curved class boundary in 2 informative
# features, 8 pure-noise features, and 15% of the labels flipped.
rng = np.random.default_rng(1)
n = 450
X = rng.uniform(-1, 1, size=(n, 10))
y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5).astype(int)
flip = rng.random(n) < 0.15
y[flip] = 1 - y[flip]
X_train, X_test, y_train, y_test = X[:300], X[300:], y[:300], y[300:]
clean_test = (X_test[:, 0] ** 2 + X_test[:, 1] ** 2 < 0.5).astype(int)

tree = DecisionTree(max_depth=20, seed=0).fit(X_train, y_train)
tree_acc = np.mean(tree.predict(X_test) == clean_test)
print(f"single deep tree   test accuracy (vs clean labels): {tree_acc:.3f}")

forest = RandomForest(n_trees=30, max_depth=20, seed=0).fit(X_train, y_train)
acc = np.mean(forest.predict(X_test) == clean_test)

# Accuracy as the ensemble grows: majority vote of the first k trees (binary labels, ties -> 0).
votes = np.array([t.predict(X_test) for t in forest.trees])
for k in [1, 5, 15, 30]:
    k_acc = np.mean((votes[:k].mean(axis=0) > 0.5).astype(int) == clean_test)
    print(f"forest, {k:>2} trees   test accuracy (vs clean labels): {k_acc:.3f}")

# The trees really are different (bagging + feature subsampling), and the vote smooths them out.
print(f"average agreement between a tree and the forest vote: {np.mean(votes == forest.predict(X_test)):.3f}")

# Predict new points: the centre is class 1 (inside the circle), the corners are class 0.
new_points = np.zeros((2, 10))
new_points[1, :2] = [0.95, 0.95]
print("predictions for centre / corner:", forest.predict(new_points), " expected [1 0]")

assert acc > tree_acc, "the forest should beat a single overfit tree"
assert forest.predict(new_points).tolist() == [1, 0]
assert len(forest.trees) == 30
print("All checks passed.")
