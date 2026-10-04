"""Foundations drills: checks for each function.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

impl = load_impl(__file__)

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
checks = [
    ("column_means", lambda: impl.column_means(X), np.array([3.0, 5.0])),
    ("row_sums", lambda: impl.row_sums(X), np.array([[3.0], [7.0], [14.0]])),
    ("center", lambda: impl.center(X), np.array([[-2.0, -3.0], [0.0, -1.0], [2.0, 4.0]])),
    ("standardize (column means)", lambda: impl.standardize(X).mean(axis=0), np.zeros(2)),
    ("standardize (column stds)", lambda: impl.standardize(X).std(axis=0), np.ones(2)),
    ("pairwise_sq_dists", lambda: impl.pairwise_sq_dists(X[:2], np.array([[1.0, 2.0], [0.0, 0.0], [4.0, 6.0]])),
     np.array([[0.0, 5.0, 25.0], [8.0, 25.0, 5.0]])),
    ("linear_predict", lambda: impl.linear_predict(X, np.array([1.0, -1.0]), 0.5), np.array([-0.5, -0.5, -3.5])),
    ("true_class_probs", lambda: impl.true_class_probs(np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]), np.array([0, 1, 1])),
     np.array([0.7, 0.8, 0.4])),
    ("class_means", lambda: impl.class_means(X, np.array([1, 0, 1])), np.array([[3.0, 4.0], [3.0, 5.5]])),
    ("one_hot", lambda: impl.one_hot(np.array([2, 0, 1]), 3), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]])),
    ("softmax_rows", lambda: impl.softmax_rows(np.array([[2.0, 1.0, 0.0], [1000.0, 1000.0, 1000.0]])),
     np.array([[0.66524096, 0.24472847, 0.09003057], [1 / 3, 1 / 3, 1 / 3]])),
    ("k_smallest_indices", lambda: impl.k_smallest_indices(np.array([[5.0, 1.0, 3.0, 2.0], [0.0, 9.0, 8.0, 7.0]]), 2),
     np.array([[1, 3], [0, 3]])),
    ("majority_vote", lambda: impl.majority_vote(np.array([[1, 1, 0], [2, 0, 2], [0, 0, 1]])), np.array([1, 2, 0])),
]

failed = 0
for name, run, expected in checks:
    try:
        got = run()
        ok = np.shape(got) == np.shape(expected) and np.allclose(got, expected)
        detail = "" if ok else f"   expected {expected.tolist()} (shape {np.shape(expected)}), got {np.asarray(got).tolist()} (shape {np.shape(got)})"
    except NotImplementedError as e:
        ok, detail = False, f"   not written yet ({e})"
    except Exception as e:  # show the error and keep checking the rest
        ok, detail = False, f"   {type(e).__name__}: {e}"
    failed += not ok
    print(("PASS " if ok else "FAIL ") + name + detail)

if failed:
    print(f"{failed} check(s) failed.")
    sys.exit(1)
print("All checks passed.")
