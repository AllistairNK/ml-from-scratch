"""Start-here exercises: checks for each function.

    python example.py               # check solution.py
    python example.py my_practice   # check your copy of the exercises
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

impl = load_impl(__file__)

# The fruit table from the lessons: weight, smoothness. Labels: 0 = apple, 1 = orange.
X = np.array([[150, 9], [170, 3], [140, 8], [180, 2], [155, 7]])
y = np.array([0, 1, 0, 1, 0])


def majority_check():
    model = impl.MajorityClassifier()
    returned = model.fit(X, y)
    assert returned is model, "fit should end with: return self"
    return model.predict(np.zeros((4, 2)))


checks = [
    ("1  first_item", lambda: impl.first_item([7, 8, 9]), 7),
    ("2  last_item", lambda: impl.last_item([7, 8, 9]), 9),
    ("3  first_n", lambda: impl.first_n([7, 8, 9, 10], 2), [7, 8]),
    ("4  double_all", lambda: impl.double_all([1, 5, 10]), [2, 10, 20]),
    ("5  count_matches", lambda: impl.count_matches([0, 1, 0, 0, 1], 0), 3),
    ("6  average", lambda: impl.average([2, 4, 9]), 5.0),
    ("7  get_row", lambda: impl.get_row(X, 1), np.array([170, 3])),
    ("8  get_column", lambda: impl.get_column(X, 0), np.array([150, 170, 140, 180, 155])),
    ("9  rows_with_label", lambda: impl.rows_with_label(X, y, 1), np.array([[170, 3], [180, 2]])),
    ("10 column_averages", lambda: impl.column_averages(X), np.array([159.0, 5.8])),
    ("11 accuracy", lambda: impl.accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])), 0.75),
    ("12 MajorityClassifier", majority_check, np.array([0, 0, 0, 0])),
]

failed = 0
for name, run, expected in checks:
    try:
        got = run()
        if isinstance(expected, np.ndarray):
            ok = np.shape(got) == expected.shape and np.allclose(got, expected)
        else:
            ok = got == expected
        detail = "" if ok else f"   expected {expected!r}, got {got!r}"
    except NotImplementedError:
        ok, detail = False, "   not written yet"
    except Exception as e:  # show the error and keep checking the rest
        ok, detail = False, f"   {type(e).__name__}: {e}"
    failed += not ok
    print(("PASS " if ok else "FAIL ") + name + detail)

if failed:
    print(f"\n{len(checks) - failed}/{len(checks)} passing. Keep going!")
    sys.exit(1)
print("\nAll checks passed.")
