import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        dists = np.sqrt(((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        nearest = np.argsort(dists, axis=1)[:, :self.k]
        nearest_labels = self.y_train[nearest]
        return np.array([np.bincount(row).argmax() for row in nearest_labels])
