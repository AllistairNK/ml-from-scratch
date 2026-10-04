# Gaussian Naive Bayes practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.

import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        # 1. Store var_smoothing.
        raise NotImplementedError("write __init__")

    def fit(self, X, y):
        # 1. self.classes = the unique labels in y.
        # 2. self.means = per-class feature means, shape (n_classes, n_features).
        # 3. self.vars = per-class feature variances + var_smoothing, same shape.
        # 4. self.priors = the fraction of samples in each class, shape (n_classes,).
        # 5. Return self.
        raise NotImplementedError("write fit")

    def joint_log_likelihood(self, X):
        # 1. diff = X broadcast against self.means -> (n, n_classes, n_features).
        # 2. log_pdf = -0.5 * (log(2*pi*var) + diff^2 / var), broadcast the same way.
        # 3. Return log(priors) + log_pdf summed over the features -> (n, n_classes).
        raise NotImplementedError("write joint_log_likelihood")

    def predict_proba(self, X):
        # 1. jll = the joint log likelihood.
        # 2. Subtract each row's max (keepdims=True).
        # 3. probs = exp(jll).
        # 4. Return probs normalised so each row sums to 1.
        raise NotImplementedError("write predict_proba")

    def predict(self, X):
        # 1. Return self.classes indexed by the argmax of the joint log likelihood along axis 1.
        raise NotImplementedError("write predict")
