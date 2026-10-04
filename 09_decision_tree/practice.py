# Decision Tree practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # 1. Store feature.
        # 2. Store threshold.
        # 3. Store left.
        # 4. Store right.
        # 5. Store value.
        raise NotImplementedError("write Node.__init__")

    def is_leaf(self):
        # 1. Return whether value is not None (careful: class 0 is falsy).
        raise NotImplementedError("write is_leaf")


def gini(y):
    # 1. counts = how many of each class (np.unique with return_counts=True).
    # 2. p = the class proportions.
    # 3. Return 1 - sum(p^2).
    raise NotImplementedError("write gini")


def most_common(y):
    # 1. values, counts = np.unique with return_counts=True.
    # 2. Return the value with the largest count.
    raise NotImplementedError("write most_common")


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None, seed=None):
        # 1. Store max_depth.
        # 2. Store min_samples_split.
        # 3. Store n_features.
        # 4. self.rng = a NumPy random Generator seeded with seed.
        raise NotImplementedError("write __init__")

    def fit(self, X, y):
        # 1. self.n_features_ = all the columns if n_features is None, else min(n_features, the number of columns).
        # 2. self.root = self.grow(X, y, depth=0).
        # 3. Return self.
        raise NotImplementedError("write fit")

    def grow(self, X, y, depth):
        # 1. If too deep, OR too few samples, OR only one class remains:
        # 2.     Return a leaf Node holding the most common label.
        # 3. feat_idxs = n_features_ random column indices (no replacement).
        # 4. feature, threshold = self.best_split(X, y, feat_idxs).
        # 5. If feature is None:
        # 6.     Return a leaf Node holding the most common label.
        # 7. left = a boolean mask of the rows where X[:, feature] <= threshold.
        # 8. left_child = recurse on the left rows, depth + 1.
        # 9. right_child = recurse on the other rows, depth + 1.
        # 10. Return an internal Node(feature, threshold, left_child, right_child).
        raise NotImplementedError("write grow")

    def best_split(self, X, y, feat_idxs):
        # 1. best_gain, best_feature, best_threshold = 0.0, None, None.
        # 2. parent_impurity = gini(y).
        # 3. For each feature in feat_idxs:
        # 4.     For each unique value of that column as the threshold:
        # 5.         left = the mask of rows <= threshold.
        # 6.         n_left = the number of rows going left.
        # 7.         If no rows or all rows go left:
        # 8.             continue
        # 9.         child_impurity = the size-weighted average of the left and right gini.
        # 10.        gain = parent_impurity - child_impurity.
        # 11.        If gain > best_gain:
        # 12.            Update best_gain, best_feature, best_threshold.
        # 13. Return best_feature, best_threshold.
        raise NotImplementedError("write best_split")

    def predict(self, X):
        # 1. Return an array of self.traverse(x, self.root) for each row x.
        raise NotImplementedError("write predict")

    def traverse(self, x, node):
        # 1. If node is a leaf:
        # 2.     Return node.value.
        # 3. If x[node.feature] <= node.threshold:
        # 4.     Recurse left.
        # 5. Otherwise recurse right.
        raise NotImplementedError("write traverse")
