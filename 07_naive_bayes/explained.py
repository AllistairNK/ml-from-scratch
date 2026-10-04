# Gaussian Naive Bayes, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Bayes:  P(c | x) is proportional to P(c) * P(x | c)
# "Naive": the features are assumed independent given the class, so P(x | c) = prod_j P(x_j | c)
# "Gaussian": each P(x_j | c) is a 1-D normal with a per-class mean and variance.
# We work in log space: the product of many small numbers underflows to 0, but a sum of logs does not.
#   log P(c | x) = log P(c) + sum_j log N(x_j; mu_cj, var_cj) + const

import numpy as np


class GaussianNaiveBayes:
    # var_smoothing is added to every variance so a zero-variance feature cannot cause a divide by zero.
    def __init__(self, var_smoothing=1e-9):
        # Store it.
        self.var_smoothing = var_smoothing

    # "Training" just estimates per-class statistics in closed form (no gradient descent).
    def fit(self, X, y):
        # The sorted unique labels. Row i of every table below belongs to self.classes[i].
        self.classes = np.unique(y)
        # Per-class feature means. Shape (n_classes, n_features).
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        # Per-class feature variances (MLE, ddof=0) plus the smoothing. Shape (n_classes, n_features).
        self.vars = np.array([X[y == c].var(axis=0) for c in self.classes]) + self.var_smoothing
        # Class priors P(c) = the fraction of the training samples in class c. Shape (n_classes,).
        self.priors = np.array([np.mean(y == c) for c in self.classes])
        # Allow chaining.
        return self

    # log P(c) + log P(x | c) for every sample and class -> (n_samples, n_classes).
    # This is the unnormalised log posterior.
    def joint_log_likelihood(self, X):
        # Distance of each sample from each class mean: (n,1,f) - (1,C,f) -> (n,C,f).
        diff = X[:, None, :] - self.means[None, :, :]
        # Log of the Gaussian pdf, per feature:
        #   log N(x; mu, var) = -0.5 * [ log(2*pi*var) + (x - mu)^2 / var ]
        log_pdf = -0.5 * (np.log(2 * np.pi * self.vars)[None, :, :] + diff ** 2 / self.vars[None, :, :])
        # The naive independence assumption: sum the per-feature logs, then add the log prior. -> (n, C).
        return np.log(self.priors) + log_pdf.sum(axis=2)

    # Normalised posteriors P(c | x).
    def predict_proba(self, X):
        # The unnormalised log posteriors.
        jll = self.joint_log_likelihood(X)
        # Log-sum-exp trick: subtract the row max so exp() does not underflow to all zeros.
        jll = jll - jll.max(axis=1, keepdims=True)
        # Back to (unnormalised) probabilities.
        probs = np.exp(jll)
        # Normalise each row to sum to 1.
        return probs / probs.sum(axis=1, keepdims=True)

    # MAP prediction: the class with the highest posterior. No normalisation is needed for argmax.
    def predict(self, X):
        # Map the column index back to the original class label (the labels need not be 0..C-1).
        return self.classes[np.argmax(self.joint_log_likelihood(X), axis=1)]
