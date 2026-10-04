import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.array([X[y == c].var(axis=0) for c in self.classes]) + self.var_smoothing
        self.priors = np.array([np.mean(y == c) for c in self.classes])
        return self

    def joint_log_likelihood(self, X):
        diff = X[:, None, :] - self.means[None, :, :]
        log_pdf = -0.5 * (np.log(2 * np.pi * self.vars)[None, :, :] + diff ** 2 / self.vars[None, :, :])
        return np.log(self.priors) + log_pdf.sum(axis=2)

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        jll = jll - jll.max(axis=1, keepdims=True)
        probs = np.exp(jll)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes[np.argmax(self.joint_log_likelihood(X), axis=1)]
