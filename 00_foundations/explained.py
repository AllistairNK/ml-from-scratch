# Foundations drills, explained line by line.
# Same code as solution.py; every line has a comment above it.
# Each function is one NumPy trick that the algorithm code reuses. Lecture F1 explains them all.

import numpy as np


# Per-feature mean of an (n, f) matrix -> (f,).  [used in: PCA, standardisation, Naive Bayes]
def column_means(X):
    # axis=0 collapses the rows, leaving one value per column.
    return X.mean(axis=0)


# Per-sample total, kept 2-D -> (n, 1), so it can divide X row by row.  [softmax]
def row_sums(X):
    # keepdims=True keeps the shape (n, 1) instead of (n,), so broadcasting works against (n, f).
    return X.sum(axis=1, keepdims=True)


# Subtract each feature's mean.  [PCA]
def center(X):
    # (n, f) - (f,) broadcasts: the mean row is subtracted from every row.
    return X - X.mean(axis=0)


# Zero mean and unit variance per feature.  [preprocessing for KNN, K-Means, GD models]
def standardize(X):
    # Center, then divide each column by its standard deviation (both broadcast as (f,)).
    return (X - X.mean(axis=0)) / X.std(axis=0)


# Squared distance between every row of A (n, f) and every row of B (m, f) -> (n, m).  [KNN, K-Means]
def pairwise_sq_dists(A, B):
    # (n,1,f) - (1,m,f) -> (n,m,f); square, then sum over the features.
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


# A linear model's prediction for every sample.  [linear/logistic regression, NN layers]
def linear_predict(X, w, b):
    # (n, f) @ (f,) -> (n,), plus the scalar bias.
    return X @ w + b


# Each row's probability for its true class.  [cross-entropy loss]
def true_class_probs(probs, y):
    # Pair row i with column y[i] using fancy indexing.
    return probs[np.arange(len(y)), y]


# The mean feature vector of each class -> (n_classes, f).  [Naive Bayes, K-Means update]
def class_means(X, y):
    # A boolean mask picks the rows of class c; average them; one row per class, in sorted order.
    return np.array([X[y == c].mean(axis=0) for c in np.unique(y)])


# Integer labels -> one-hot matrix (n, n_classes).  [softmax gradient: probs - one_hot]
def one_hot(y, n_classes):
    # Start from all zeros.
    out = np.zeros((len(y), n_classes))
    # Put a 1 at (i, y[i]) for every row at once.
    out[np.arange(len(y)), y] = 1
    # Return the matrix.
    return out


# A row-wise, numerically stable softmax.  [NN output, attention]
def softmax_rows(Z):
    # Subtract each row's max so exp() cannot overflow (the result is unchanged).
    Z = Z - Z.max(axis=1, keepdims=True)
    # Exponentiate.
    exp_z = np.exp(Z)
    # Normalise each row to sum to 1.
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# The column indices of the k smallest values in each row -> (n, k).  [KNN neighbours]
def k_smallest_indices(D, k):
    # argsort sorts each row's indices by value, ascending; keep the first k.
    return np.argsort(D, axis=1)[:, :k]


# The most frequent non-negative int in each row.  [KNN vote]
def majority_vote(labels):
    # bincount counts the occurrences of 0, 1, 2, ...; argmax picks the most common (ties go to the smaller label).
    return np.array([np.bincount(row).argmax() for row in labels])
