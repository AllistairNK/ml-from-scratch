# Start-here exercises, explained line by line.
# Same code as solution.py; every line has a comment above it.
# Functions 1-6 work on plain Python lists; 7-12 work on NumPy arrays.

# Load NumPy and call it np (needed from exercise 7 onwards).
import numpy as np


# 1. The first item of a list.
def first_item(values):
    # Indexes start at 0, so index 0 is the first item.
    return values[0]


# 2. The last item of a list.
def last_item(values):
    # A negative index counts from the end: -1 is the last item, -2 the one before it.
    return values[-1]


# 3. The first n items.
def first_n(values, n):
    # A slice start:stop. Leaving out the start means "from the beginning"; stop n is NOT included,
    # so you get the indexes 0..n-1, which is n items.
    return values[:n]


# 4. A new list with every value doubled.
def double_all(values):
    # A list comprehension: "v * 2, for each v in values".
    return [v * 2 for v in values]


# 5. How many labels equal target (for example, how many apples).
def count_matches(labels, target):
    # Start a counter at zero.
    count = 0
    # Look at each label in turn.
    for label in labels:
        # == asks "is it equal?" (a single = would try to store a value).
        if label == target:
            # += 1 adds one to the counter.
            count += 1
    # Hand back the final count (it must be indented level with the for, i.e. after the loop ends).
    return count


# 6. The average (mean) of a list of numbers.
def average(values):
    # sum adds everything up; len counts the items.
    return sum(values) / len(values)


# 7. Row i of a 2-D array: everything about sample i.
def get_row(X, i):
    # One index on a 2-D array selects a whole row.
    return X[i]


# 8. Column j of a 2-D array: one feature for every sample.
def get_column(X, j):
    # [rows, columns]: ":" means all the rows, j picks the column.
    return X[:, j]


# 9. Only the rows whose label equals `label` (for example, all the apples).
def rows_with_label(X, y, label):
    # y == label gives a True/False mask; X[mask] keeps the rows where it's True.
    return X[y == label]


# 10. The average of each column (each feature).
def column_averages(X):
    # axis=0 averages DOWN the rows, leaving one value per column.
    return X.mean(axis=0)


# 11. The fraction of predictions that are correct.
def accuracy(y_true, y_pred):
    # Comparing gives True/False per sample; the mean counts True as 1 and False as 0.
    return np.mean(y_true == y_pred)


# 12. The simplest possible model: always predict the most common label seen in training.
#     Real models in this course have exactly this shape (fit, then predict), just smarter insides.
class MajorityClassifier:
    # Learn from the training data. X is unused here (this model ignores the features!).
    def fit(self, X, y):
        # bincount counts how many times each label 0, 1, 2, ... appears; argmax gives the index of
        # the biggest count, which is the most common label. Store it on self so predict can use it.
        self.label = np.bincount(y).argmax()
        # Return the model itself, so you can write MajorityClassifier().fit(X, y).predict(X).
        return self

    # Predict a label for every row of X.
    def predict(self, X):
        # np.full(n, value) makes an array of n copies of value: one prediction per sample.
        return np.full(len(X), self.label)
