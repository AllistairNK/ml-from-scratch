# Logistic Regression (binary) with gradient descent, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Model:     p = sigmoid(X @ w + b)                  probability that y = 1
# Loss:      binary cross-entropy  -mean(y log p + (1-y) log(1-p))
# Gradients: dL/dw = X^T (p - y) / n      dL/db = mean(p - y)
#            (sigmoid + cross-entropy cancel out neatly, so this matches the linear regression
#             gradient with no factor of 2.)

import numpy as np


# Squashes any real number into (0, 1), so we can read the output as a probability.
def sigmoid(z):
    # exp(-z) overflows for very negative z (around z < -710). Clipping avoids the warning
    # and changes nothing, because sigmoid is already 0 or 1 to float precision out there.
    z = np.clip(z, -500, 500)
    # The logistic function 1 / (1 + e^-z).
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    # Same hyperparameters as linear regression.
    def __init__(self, lr=0.1, n_iters=1000):
        # Gradient-descent step size.
        self.lr = lr
        # Number of full-batch steps.
        self.n_iters = n_iters

    # y must contain 0/1 labels.
    def fit(self, X, y):
        # Shapes: X (n_samples, n_features), y (n_samples,).
        n_samples, n_features = X.shape
        # Zero initialisation is fine: the loss is convex.
        self.w = np.zeros(n_features)
        # Scalar bias.
        self.b = 0.0
        # Loss history, for checking that training converges.
        self.losses = []
        # The gradient-descent loop.
        for _ in range(self.n_iters):
            # Forward pass: linear score -> sigmoid -> probability of class 1. Shape (n_samples,).
            p = sigmoid(X @ self.w + self.b)
            # log(0) = -inf, so keep the probabilities strictly inside (0, 1) before taking logs.
            p_safe = np.clip(p, 1e-12, 1 - 1e-12)
            # Binary cross-entropy, averaged over samples (bookkeeping only).
            self.losses.append(-np.mean(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe)))
            # dLoss/dz for each sample simplifies to (p - y): the key fact to remember.
            error = p - y
            # Chain rule through z = Xw + b: dL/dw = X^T (p - y) / n. Shape (n_features,).
            dw = (X.T @ error) / n_samples
            # dL/db = average of (p - y).
            db = np.mean(error)
            # Gradient-descent update for the weights.
            self.w -= self.lr * dw
            # Gradient-descent update for the bias.
            self.b -= self.lr * db
        # Allow chaining.
        return self

    # Probabilities are often more useful than hard labels (ROC curves, custom thresholds).
    def predict_proba(self, X):
        # Same forward pass as in training.
        return sigmoid(X @ self.w + self.b)

    # Hard 0/1 predictions. The threshold trades precision against recall.
    def predict(self, X, threshold=0.5):
        # A boolean comparison turned into ints 0/1.
        return (self.predict_proba(X) >= threshold).astype(int)
