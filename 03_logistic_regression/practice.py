# Logistic Regression practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


def sigmoid(z):
    # 1. Clip z to [-500, 500] to avoid overflow in exp.
    # 2. Return 1 / (1 + e^-z).
    raise NotImplementedError("write sigmoid")


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        # 1. Store the learning rate on self.
        # 2. Store the number of iterations on self.
        raise NotImplementedError("write __init__")

    def fit(self, X, y):
        # 1. Unpack n_samples and n_features from X.shape.
        # 2. Initialise self.w as zeros, one per feature.
        # 3. Initialise self.b to 0.0.
        # 4. Create an empty list self.losses.
        # 5. Loop n_iters times:
        # 6.     p = sigmoid of the linear score (probability that y = 1).
        # 7.     p_safe = p clipped to [1e-12, 1 - 1e-12].
        # 8.     Append the binary cross-entropy (using p_safe) to self.losses.
        # 9.     error = p - y.
        # 10.    dw = X^T error / n_samples.
        # 11.    db = mean of error.
        # 12.    Update self.w.
        # 13.    Update self.b.
        # 14. Return self.
        raise NotImplementedError("write fit")

    def predict_proba(self, X):
        # 1. Return the sigmoid of the linear score.
        raise NotImplementedError("write predict_proba")

    def predict(self, X, threshold=0.5):
        # 1. Return 1 where the probability is >= threshold, else 0 (as ints).
        raise NotImplementedError("write predict")
