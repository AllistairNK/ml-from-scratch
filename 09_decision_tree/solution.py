import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)


def most_common(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None, seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.n_features_ = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.root = self.grow(X, y, depth=0)
        return self

    def grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=most_common(y))
        feat_idxs = self.rng.choice(X.shape[1], self.n_features_, replace=False)
        feature, threshold = self.best_split(X, y, feat_idxs)
        if feature is None:
            return Node(value=most_common(y))
        left = X[:, feature] <= threshold
        left_child = self.grow(X[left], y[left], depth + 1)
        right_child = self.grow(X[~left], y[~left], depth + 1)
        return Node(feature, threshold, left_child, right_child)

    def best_split(self, X, y, feat_idxs):
        best_gain, best_feature, best_threshold = 0.0, None, None
        parent_impurity = gini(y)
        for feature in feat_idxs:
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold
                n_left = left.sum()
                if n_left == 0 or n_left == len(y):
                    continue
                child_impurity = (n_left * gini(y[left]) + (len(y) - n_left) * gini(y[~left])) / len(y)
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self.traverse(x, self.root) for x in X])

    def traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)
