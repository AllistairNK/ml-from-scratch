import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            p_safe = np.clip(p, 1e-12, 1 - 1e-12)
            self.losses.append(-np.mean(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe)))
            error = p - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
