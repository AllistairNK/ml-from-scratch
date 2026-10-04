import numpy as np


def first_item(values):
    return values[0]


def last_item(values):
    return values[-1]


def first_n(values, n):
    return values[:n]


def double_all(values):
    return [v * 2 for v in values]


def count_matches(labels, target):
    count = 0
    for label in labels:
        if label == target:
            count += 1
    return count


def average(values):
    return sum(values) / len(values)


def get_row(X, i):
    return X[i]


def get_column(X, j):
    return X[:, j]


def rows_with_label(X, y, label):
    return X[y == label]


def column_averages(X):
    return X.mean(axis=0)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


class MajorityClassifier:
    def fit(self, X, y):
        self.label = np.bincount(y).argmax()
        return self

    def predict(self, X):
        return np.full(len(X), self.label)
