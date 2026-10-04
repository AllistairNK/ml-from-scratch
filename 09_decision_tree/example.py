"""Decision Tree usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

impl = load_impl(__file__)
DecisionTree, gini = impl.DecisionTree, impl.gini


def print_tree(node, names, indent=""):
    if node.is_leaf():
        print(indent + "-> predict", node.value)
        return
    print(f"{indent}if {names[node.feature]} <= {node.threshold:.2f}:")
    print_tree(node.left, names, indent + "    ")
    print(indent + "else:")
    print_tree(node.right, names, indent + "    ")


# Gini basics.
print("gini([0,0,0,0]) =", gini(np.array([0, 0, 0, 0])), " (pure)")
print("gini([0,0,1,1]) =", gini(np.array([0, 0, 1, 1])), " (50/50, max for 2 classes)")

# A rule-based dataset: approved if income > 50 and debt <= 20, plus a pure-noise feature.
rng = np.random.default_rng(0)
income = rng.uniform(0, 100, 400)
debt = rng.uniform(0, 40, 400)
noise = rng.uniform(0, 1, 400)
X = np.c_[income, debt, noise]
y = ((income > 50) & (debt <= 20)).astype(int)
X_train, X_test, y_train, y_test = X[:300], X[300:], y[:300], y[300:]

tree = DecisionTree(max_depth=3).fit(X_train, y_train)
print("\nlearned tree (it should rediscover the rule and ignore 'noise'):")
print_tree(tree.root, ["income", "debt", "noise"])
acc = np.mean(tree.predict(X_test) == y_test)
print(f"test accuracy: {acc:.3f}")

# Depth controls over/underfitting on noisy labels (10% of the labels flipped).
y_noisy = y.copy()
flip = rng.random(400) < 0.1
y_noisy[flip] = 1 - y_noisy[flip]
print("\ndepth  train_acc  test_acc   (noisy labels)")
for depth in [1, 2, 3, 5, 20]:
    t = DecisionTree(max_depth=depth).fit(X_train, y_noisy[:300])
    tr = np.mean(t.predict(X_train) == y_noisy[:300])
    te = np.mean(t.predict(X_test) == y_test)
    print(f"{depth:>5}  {tr:9.3f}  {te:8.3f}")

# Predict for new applicants.
applicants = np.array([[80, 10, 0.5], [80, 30, 0.5], [20, 5, 0.5]])
print("\napplicants", applicants[:, :2].tolist(), "->", tree.predict(applicants), " expected [1 0 0]")

assert gini(np.array([1, 1, 1])) == 0
assert np.isclose(gini(np.array([0, 0, 1, 1])), 0.5)
assert acc > 0.95
assert tree.predict(applicants).tolist() == [1, 0, 0]
deep = DecisionTree(max_depth=50).fit(X_train, y_noisy[:300])
assert np.mean(deep.predict(X_train) == y_noisy[:300]) == 1.0, "an unlimited-depth tree should memorise the training set"
print("All checks passed.")
