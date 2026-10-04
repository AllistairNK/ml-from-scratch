# Linear Regression with batch gradient descent, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Model:     y_hat = X @ w + b
# Loss:      MSE = mean((y_hat - y)^2)
# Gradients: dL/dw = (2/n) * X^T (y_hat - y)     dL/db = (2/n) * sum(y_hat - y)
# Update:    w -= lr * dL/dw,   b -= lr * dL/db

import numpy as np


class LinearRegression:
    # lr = step size, n_iters = number of gradient-descent steps.
    def __init__(self, lr=0.01, n_iters=1000):
        # Too large an lr makes the loss explode (diverge); too small makes training slow.
        self.lr = lr
        # Each iteration uses the whole dataset once (full-batch GD).
        self.n_iters = n_iters

    def fit(self, X, y):
        # X: (n_samples, n_features), y: (n_samples,). Unpack the shape for the gradient scale.
        n_samples, n_features = X.shape
        # One weight per feature. Zeros is fine here: the loss is convex, so there is no symmetry to break.
        self.w = np.zeros(n_features)
        # The bias (intercept) is a single scalar.
        self.b = 0.0
        # Record the loss each step so you can plot it and check that it decreases.
        self.losses = []
        # The gradient-descent loop.
        for _ in range(self.n_iters):
            # Forward pass: predictions for every sample at once. Shape (n_samples,).
            y_pred = X @ self.w + self.b
            # Residuals. Their sign matters: (pred - true) gives the gradient direction below.
            error = y_pred - y
            # MSE for this step (bookkeeping only; not needed for the update).
            self.losses.append(np.mean(error ** 2))
            # dMSE/dw: X^T @ error sums each feature times its residual over all samples -> (n_features,).
            dw = (2 / n_samples) * (X.T @ error)
            # dMSE/db: the bias touches every sample with coefficient 1, so just sum the residuals.
            db = (2 / n_samples) * np.sum(error)
            # Step downhill: move against the gradient.
            self.w -= self.lr * dw
            # Same for the bias.
            self.b -= self.lr * db
        # Allow chaining: LinearRegression().fit(X, y).predict(X_new)
        return self

    def predict(self, X):
        # Prediction is the same linear function used in training.
        return X @ self.w + self.b
