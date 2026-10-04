import os
import sys

import numpy as np

# Reuse the decision tree from 09_decision_tree (setup, not part of the exercise).
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from loader import load_module  # noqa: E402

tree_module = load_module(os.path.join(ROOT, "09_decision_tree", "solution.py"), "decision_tree")
DecisionTree, most_common = tree_module.DecisionTree, tree_module.most_common


class RandomForest:
    def __init__(self, n_trees=25, max_depth=10, min_samples_split=2, n_features=None, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_samples, n_cols = X.shape
        n_features = self.n_features or max(1, int(np.sqrt(n_cols)))
        self.trees = []
        for _ in range(self.n_trees):
            idxs = rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=n_features, seed=int(rng.integers(1_000_000_000)))
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([most_common(column) for column in all_preds.T])
