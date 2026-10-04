# K-Nearest Neighbours, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Idea: KNN has no training step. To classify a point, find the k training
# points closest to it and take a majority vote of their labels.

# NumPy gives us vectorised array maths (no Python loops over samples).
import numpy as np


# One class holding the hyperparameter (k) and the memorised training data.
class KNN:
    # k = how many neighbours vote. Small k -> noisy/overfits, large k -> smooth/underfits.
    def __init__(self, k=3):
        # Store k so predict() can use it later.
        self.k = k

    # "Training" a KNN is just remembering the data (it is a lazy learner).
    def fit(self, X, y):
        # X: (n_train, n_features) feature matrix. Keep a reference to it.
        self.X_train = X
        # y: (n_train,) integer class labels 0..C-1. Keep a reference to it.
        self.y_train = y
        # Returning self lets you write KNN(k=5).fit(X, y).predict(X_test).
        return self

    # All the real work happens at prediction time.
    def predict(self, X):
        # Euclidean distance from every test point to every training point.
        # X[:, None, :]            -> (n_test, 1, n_features)
        # self.X_train[None, :, :] -> (1, n_train, n_features)
        # Subtracting broadcasts to (n_test, n_train, n_features);
        # square, sum over features (axis=2), then sqrt -> dists is (n_test, n_train).
        dists = np.sqrt(((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        # argsort each row (smallest distance first) and keep the first k columns:
        # nearest[i] holds the indices of the k closest training points to test point i.
        # Shape (n_test, k).
        nearest = np.argsort(dists, axis=1)[:, :self.k]
        # Fancy indexing: swap each training index for its label. Shape (n_test, k).
        nearest_labels = self.y_train[nearest]
        # For each test row, bincount counts how often each label appears;
        # argmax picks the most frequent one (ties go to the smallest label).
        return np.array([np.bincount(row).argmax() for row in nearest_labels])
