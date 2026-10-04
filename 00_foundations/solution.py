import numpy as np


def column_means(X):
    return X.mean(axis=0)


def row_sums(X):
    return X.sum(axis=1, keepdims=True)


def center(X):
    return X - X.mean(axis=0)


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def pairwise_sq_dists(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


def linear_predict(X, w, b):
    return X @ w + b


def true_class_probs(probs, y):
    return probs[np.arange(len(y)), y]


def class_means(X, y):
    return np.array([X[y == c].mean(axis=0) for c in np.unique(y)])


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    exp_z = np.exp(Z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def k_smallest_indices(D, k):
    return np.argsort(D, axis=1)[:, :k]


def majority_vote(labels):
    return np.array([np.bincount(row).argmax() for row in labels])
