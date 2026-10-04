# Start-here exercises. Replace each `raise NotImplementedError(...)` line with your code.
# Check your work with the "Run checks" button in the app (or: python example.py my_practice).
# Each exercise is checked separately, so you can do them one at a time.
# Read B1-B3 first. Stuck? Look at explained.py.

import numpy as np


def first_item(values):
    # 1. Return the first item of the list (remember: indexes start at 0).
    raise NotImplementedError("write first_item")


def last_item(values):
    # 1. Return the last item of the list (hint: a negative index).
    raise NotImplementedError("write last_item")


def first_n(values, n):
    # 1. Return the first n items of the list (hint: a slice).
    raise NotImplementedError("write first_n")


def double_all(values):
    # 1. Return a NEW list with every value multiplied by 2 (hint: a list comprehension).
    raise NotImplementedError("write double_all")


def count_matches(labels, target):
    # 1. Make a variable count that starts at 0.
    # 2. Loop over each label in labels:
    # 3.     If the label equals target:
    # 4.         Add 1 to count.
    # 5. After the loop, return count.
    raise NotImplementedError("write count_matches")


def average(values):
    # 1. Return the total of the values divided by how many there are (hint: sum and len).
    raise NotImplementedError("write average")


def get_row(X, i):
    # 1. Return row i of the 2-D array X.
    raise NotImplementedError("write get_row")


def get_column(X, j):
    # 1. Return column j of X, for all the rows (hint: X[rows, columns] and ":").
    raise NotImplementedError("write get_column")


def rows_with_label(X, y, label):
    # 1. Return only the rows of X whose label in y equals label (hint: a mask).
    raise NotImplementedError("write rows_with_label")


def column_averages(X):
    # 1. Return the mean of each column of X (hint: which axis disappears?).
    raise NotImplementedError("write column_averages")


def accuracy(y_true, y_pred):
    # 1. Return the fraction of positions where y_true equals y_pred (hint: np.mean of a comparison).
    raise NotImplementedError("write accuracy")


class MajorityClassifier:
    def fit(self, X, y):
        # 1. Store the most common label in y as self.label (hint: np.bincount(y).argmax()).
        # 2. Return self.
        raise NotImplementedError("write fit")

    def predict(self, X):
        # 1. Return an array with self.label repeated once per row of X (hint: np.full and len).
        raise NotImplementedError("write predict")
