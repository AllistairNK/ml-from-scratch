# Random Forest practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import os
import sys

import numpy as np

# Setup (given): reuse the decision tree from 09_decision_tree.
# To use YOUR tree instead, change "solution.py" to "practice.py" below.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from loader import load_module  # noqa: E402

tree_module = load_module(os.path.join(ROOT, "09_decision_tree", "solution.py"), "decision_tree")
DecisionTree, most_common = tree_module.DecisionTree, tree_module.most_common


class RandomForest:
    def __init__(self, n_trees=25, max_depth=10, min_samples_split=2, n_features=None, seed=0):
        # 1. Store n_trees.
        # 2. Store max_depth.
        # 3. Store min_samples_split.
        # 4. Store n_features.
        # 5. Store seed.
        raise NotImplementedError("write __init__")

    def fit(self, X, y):
        # 1. rng = a NumPy random Generator seeded with self.seed.
        # 2. Unpack n_samples and n_cols from X.shape.
        # 3. n_features = self.n_features, or by default max(1, int(sqrt(n_cols))).
        # 4. self.trees = [].
        # 5. Loop n_trees times:
        # 6.     idxs = a bootstrap sample: n_samples row indices drawn WITH replacement.
        # 7.     tree = a DecisionTree with max_depth, min_samples_split, n_features,
        #               and its own seed (an int drawn from rng).
        # 8.     Fit the tree on X[idxs], y[idxs].
        # 9.     Append the tree to self.trees.
        # 10. Return self.
        raise NotImplementedError("write fit")

    def predict(self, X):
        # 1. all_preds = each tree's predictions stacked into shape (n_trees, n_samples).
        # 2. Return the majority vote (most_common) of each column (transpose first).
        raise NotImplementedError("write predict")
