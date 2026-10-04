# Decision Tree classifier (CART-style, Gini impurity), explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Build the tree recursively, top-down and greedily:
#   - Stop and make a leaf if the node is pure, too small, or too deep.
#   - Otherwise try every (feature, threshold) pair and pick the one that most reduces the
#     weighted Gini impurity of the two children (the information gain).
#   - Split the data and recurse into the left (x <= threshold) and right (x > threshold) children.
# Predict by walking from the root down to a leaf.

import numpy as np


# A tree node: either an internal split (feature, threshold, children) or a leaf (value).
class Node:
    # All fields are optional: a leaf sets only value; an internal node sets the other four.
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Index of the column this node splits on.
        self.feature = feature
        # Go left if x[feature] <= threshold, else go right.
        self.threshold = threshold
        # Subtree for x[feature] <= threshold.
        self.left = left
        # Subtree for x[feature] > threshold.
        self.right = right
        # The predicted class (leaves only).
        self.value = value

    # A leaf is a node that stores a prediction.
    def is_leaf(self):
        # Use "is not None", not truthiness: class 0 is a valid value, but it is falsy.
        return self.value is not None


# Gini impurity = 1 - sum_k p_k^2. It is 0 for a pure node and at its maximum when the classes are evenly mixed.
def gini(y):
    # How many samples of each class are in this node.
    _, counts = np.unique(y, return_counts=True)
    # Class proportions.
    p = counts / len(y)
    # Gini impurity. (Entropy -sum p log p works too and usually gives similar trees.)
    return 1 - np.sum(p ** 2)


# Majority class: the prediction stored in a leaf.
def most_common(y):
    # The unique labels and their counts.
    values, counts = np.unique(y, return_counts=True)
    # The label with the highest count.
    return values[np.argmax(counts)]


class DecisionTree:
    # max_depth and min_samples_split limit growth (regularisation against overfitting).
    # n_features: how many randomly chosen features to try at each split (None = all). Random Forest uses this.
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None, seed=None):
        # Maximum depth of the tree.
        self.max_depth = max_depth
        # Nodes with fewer samples than this become leaves.
        self.min_samples_split = min_samples_split
        # Number of features considered per split.
        self.n_features = n_features
        # Random generator for the feature subsampling.
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        # Resolve n_features: all of them by default, and never more than X has.
        self.n_features_ = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        # Build the whole tree recursively, starting from the root at depth 0.
        self.root = self.grow(X, y, depth=0)
        # Allow chaining.
        return self

    # Recursively build the subtree for the samples (X, y) that reached this node.
    def grow(self, X, y, depth):
        # Stopping criteria: too deep, too few samples, or already pure.
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            # Make a leaf that predicts the majority class.
            return Node(value=most_common(y))
        # Choose which features to consider at this split (all of them unless n_features is set).
        feat_idxs = self.rng.choice(X.shape[1], self.n_features_, replace=False)
        # Find the best split among those features.
        feature, threshold = self.best_split(X, y, feat_idxs)
        # No split reduces the impurity (e.g. identical feature rows with different labels).
        if feature is None:
            # So this node becomes a leaf.
            return Node(value=most_common(y))
        # Boolean mask of the samples that go left.
        left = X[:, feature] <= threshold
        # Recurse on the left partition.
        left_child = self.grow(X[left], y[left], depth + 1)
        # Recurse on the right partition (~ is the boolean NOT).
        right_child = self.grow(X[~left], y[~left], depth + 1)
        # An internal node joining the two subtrees.
        return Node(feature, threshold, left_child, right_child)

    # An exhaustive search over the candidate features and every observed value as the threshold.
    def best_split(self, X, y, feat_idxs):
        # Start at gain 0, so only splits that actually improve things are accepted.
        best_gain, best_feature, best_threshold = 0.0, None, None
        # Impurity before splitting.
        parent_impurity = gini(y)
        # Try each candidate feature ...
        for feature in feat_idxs:
            # ... and each unique value of it as a threshold.
            for threshold in np.unique(X[:, feature]):
                # Which samples would go left.
                left = X[:, feature] <= threshold
                # How many of them there are.
                n_left = left.sum()
                # A split that sends everything one way is not a split.
                if n_left == 0 or n_left == len(y):
                    # Skip it.
                    continue
                # Children's impurity, weighted by the share of samples in each child.
                child_impurity = (n_left * gini(y[left]) + (len(y) - n_left) * gini(y[~left])) / len(y)
                # Information gain = the impurity reduction.
                gain = parent_impurity - child_impurity
                # Keep the best split seen so far.
                if gain > best_gain:
                    # Record it.
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        # (None, None) if nothing improved.
        return best_feature, best_threshold

    # Predict each row independently by walking the tree.
    def predict(self, X):
        # One traversal per sample.
        return np.array([self.traverse(x, self.root) for x in X])

    # Follow the splits from node down to a leaf.
    def traverse(self, x, node):
        # At a leaf: return its class.
        if node.is_leaf():
            # The leaf prediction.
            return node.value
        # Go left if the feature value is at or below the threshold ...
        if x[node.feature] <= node.threshold:
            # ... recursing into the left subtree.
            return self.traverse(x, node.left)
        # ... otherwise go right.
        return self.traverse(x, node.right)
