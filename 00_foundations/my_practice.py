# Foundations drills: twelve one-line NumPy functions. Replace each `raise` with one line of code.
# Check your work with:  python example.py practice
# Read lecture F1 first. Every trick here appears again in the algorithms.

import numpy as np


def column_means(X):
    # 1. Return the mean of each column of X (shape (f,)).
    raise NotImplementedError("write column_means")


def row_sums(X):
    # 1. Return the sum of each row, keeping a 2-D shape (n, 1).
    raise NotImplementedError("write row_sums")


def center(X):
    # 1. Return X with each column's mean subtracted.
    raise NotImplementedError("write center")


def standardize(X):
    # 1. Return X with each column centered and divided by its standard deviation.
    raise NotImplementedError("write standardize")


def pairwise_sq_dists(A, B):
    # 1. Return the (n, m) matrix of squared distances between the rows of A (n, f) and B (m, f).
    #    Hint: A[:, None, :] and B[None, :, :].
    raise NotImplementedError("write pairwise_sq_dists")


def linear_predict(X, w, b):
    # 1. Return X times w, plus b (one number per row).
    raise NotImplementedError("write linear_predict")


def true_class_probs(probs, y):
    # 1. Return probs[i, y[i]] for every row i (fancy indexing with np.arange).
    raise NotImplementedError("write true_class_probs")


def class_means(X, y):
    # 1. Return an array whose row c is the mean of X's rows with label c (classes in sorted order).
    raise NotImplementedError("write class_means")


def one_hot(y, n_classes):
    # 1. out = a zeros matrix of shape (len(y), n_classes).
    # 2. Set out[i, y[i]] = 1 for all i in one line.
    # 3. Return out.
    raise NotImplementedError("write one_hot")


def softmax_rows(Z):
    # 1. Subtract each row's max (keepdims=True).
    # 2. Exponentiate.
    # 3. Return each row divided by its sum (keepdims=True).
    raise NotImplementedError("write softmax_rows")


def k_smallest_indices(D, k):
    # 1. Return the indices of the k smallest values in each row (argsort + slice).
    raise NotImplementedError("write k_smallest_indices")


def majority_vote(labels):
    # 1. Return the most common value in each row (bincount + argmax).
    raise NotImplementedError("write majority_vote")
