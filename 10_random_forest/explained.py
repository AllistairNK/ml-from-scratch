# Random Forest classifier, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# A random forest is many deep decision trees whose errors are made as UNcorrelated as possible:
#   1. Bagging: each tree trains on a bootstrap sample (n rows drawn WITH replacement).
#   2. Feature subsampling: each split considers only a random subset of the features (default sqrt(f)).
# Prediction = a majority vote over the trees. Averaging many low-bias, high-variance trees cuts the variance.

import os
import sys

import numpy as np

# Reuse the decision tree from 09_decision_tree (setup, not part of the exercise).
# Both folders have a file called solution.py, so load the tree by path under a different module name.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Make loader.py (at the repo root) importable.
sys.path.insert(0, ROOT)
# A helper that imports a .py file from a path.
from loader import load_module  # noqa: E402

# Load 09_decision_tree/solution.py as the module "decision_tree".
tree_module = load_module(os.path.join(ROOT, "09_decision_tree", "solution.py"), "decision_tree")
# The tree class (it already supports n_features) and the majority-vote helper.
DecisionTree, most_common = tree_module.DecisionTree, tree_module.most_common


class RandomForest:
    # n_trees: more is never worse for accuracy, only slower. The other parameters are passed to each tree.
    def __init__(self, n_trees=25, max_depth=10, min_samples_split=2, n_features=None, seed=0):
        # Number of trees in the ensemble.
        self.n_trees = n_trees
        # Per-tree depth limit (forests usually use deep trees; the averaging handles the overfitting).
        self.max_depth = max_depth
        # Per-tree minimum node size for splitting.
        self.min_samples_split = min_samples_split
        # Features to try per split (None -> sqrt(n_features), the classification default).
        self.n_features = n_features
        # Seed for reproducibility.
        self.seed = seed

    def fit(self, X, y):
        # One generator drives all the randomness (the bootstraps and the per-tree seeds).
        rng = np.random.default_rng(self.seed)
        # Dataset shape.
        n_samples, n_cols = X.shape
        # Default feature-subset size: sqrt of the number of columns (at least 1).
        n_features = self.n_features or max(1, int(np.sqrt(n_cols)))
        # The fitted trees.
        self.trees = []
        # Train every tree independently (this is easy to parallelise).
        for _ in range(self.n_trees):
            # Bootstrap: n row indices WITH replacement. About 63% of the unique rows appear in each sample;
            # the rest are "out-of-bag" and can be used for validation.
            idxs = rng.choice(n_samples, n_samples, replace=True)
            # A tree with feature subsampling. Give each tree its own seed so the trees differ.
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=n_features, seed=int(rng.integers(1_000_000_000)))
            # Fit the tree on its bootstrap sample.
            tree.fit(X[idxs], y[idxs])
            # Keep it.
            self.trees.append(tree)
        # Allow chaining.
        return self

    def predict(self, X):
        # Every tree's predictions, stacked -> (n_trees, n_samples).
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to iterate per sample (one column = all the trees' votes for that sample) and take the majority.
        return np.array([most_common(column) for column in all_preds.T])
