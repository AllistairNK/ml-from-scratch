# KNN practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


class KNN:
    def __init__(self, k=3):
        # 1. Store k on self.
        raise NotImplementedError("write __init__")

    def fit(self, X, y):
        # 1. Store the training features X on self.
        # 2. Store the training labels y on self.
        # 3. Return self.
        raise NotImplementedError("write fit")

    def predict(self, X):
        # 1. dists: Euclidean distance from every test row to every training row.
        #    Broadcast (n_test, 1, f) - (1, n_train, f), square, sum over the last axis, sqrt.
        #    Result shape: (n_test, n_train).
        # 2. nearest: indices of the k smallest distances in each row (argsort, slice). Shape (n_test, k).
        # 3. nearest_labels: look up the training labels at those indices. Shape (n_test, k).
        # 4. Return an array with the majority label of each row (bincount + argmax).
        raise NotImplementedError("write predict")
