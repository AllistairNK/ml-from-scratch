# Linear Regression (gradient descent) practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        # 1. Store the learning rate on self.
        # 2. Store the number of iterations on self.
        raise NotImplementedError("write __init__")

    def fit(self, X, y):
        # 1. Unpack n_samples and n_features from X.shape.
        # 2. Initialise the weights self.w as zeros, one per feature.
        # 3. Initialise the bias self.b to 0.0.
        # 4. Create an empty list self.losses.
        # 5. Loop n_iters times:
        # 6.     y_pred = linear prediction for all samples.
        # 7.     error = y_pred minus the true y.
        # 8.     Append the MSE of this step to self.losses.
        # 9.     dw = gradient of the MSE w.r.t. w (uses X.T, error and 2/n).
        # 10.    db = gradient of the MSE w.r.t. b.
        # 11.    Update self.w with a gradient-descent step.
        # 12.    Update self.b with a gradient-descent step.
        # 13. Return self.
        raise NotImplementedError("write fit")

    def predict(self, X):
        # 1. Return the linear prediction X w + b.
        raise NotImplementedError("write predict")
