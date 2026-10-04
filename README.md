# ML From Scratch

Relearning core ML algorithms by implementing them from a blank file in NumPy.

**Goal:** each algorithm "interview-ready", meaning I can write it from a blank file in 15–20 min without looking anything up.
**Pace:** 1–2 hrs/day. Finish the 6 priority algorithms first, because a technical round could come within 2–3 weeks.

## Progress

Status key: ⬜ not started · 📖 Day 1 (reference) · ✏️ Day 2 (blank file) · ⏱️ Day 3+ (timed) · 🔁 spaced repeat done · ✅ interview-ready

### Priority set (first 2 weeks)

| # | Algorithm | Est. | Status | Best time | Last practised | Folder |
|---|-----------|------|--------|-----------|----------------|--------|
| 1 | KNN | ½ day | ⬜ | | | [01_knn](01_knn/) |
| 2 | Linear Regression (GD) | 1–2 days | ⬜ | | | [02_linear_regression](02_linear_regression/) |
| 3 | Logistic Regression | 1–2 days | ⬜ | | | [03_logistic_regression](03_logistic_regression/) |
| 4 | K-Means | 2–3 days | ⬜ | | | [04_kmeans](04_kmeans/) |
| 5 | Neural Network + Backprop | 4–5 days | ⬜ | | | [05_neural_network](05_neural_network/) |
| 6 | Attention / Multi-Head | 3–4 days | ⬜ | | | [06_attention](06_attention/) |

### Later

| # | Algorithm | Est. | Status | Best time | Last practised | Folder |
|---|-----------|------|--------|-----------|----------------|--------|
| 7 | Naive Bayes | 1–2 days | ⬜ | | | [07_naive_bayes](07_naive_bayes/) |
| 8 | PCA (np.linalg) | 1 day | ⬜ | | | [08_pca](08_pca/) |
| 9 | Decision Tree | 4–5 days | ⬜ | | | [09_decision_tree](09_decision_tree/) |
| 10 | Random Forest | +1–2 days | ⬜ | | | [10_random_forest](10_random_forest/) |

### Explain, don't code

| Algorithm | Status | Notes |
|-----------|--------|-------|
| SVM | ⬜ | [notes](explain_only/svm.md) |
| XGBoost | ⬜ | [notes](explain_only/xgboost.md) |

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## What's in each algorithm folder

| File | What it is |
|------|------------|
| `solution.py` | The clean, correct implementation. The interview-length version to aim for. |
| `explained.py` | The same code with a comment above every line explaining what it does, why, and the array shapes. |
| `practice.py` | The function signatures with numbered hints, one per line of the solution. Write the code under each hint. |
| `example.py` | A usage example on synthetic data, with sanity checks (accuracy, gradient check, shapes, …). |
| `LOG.md` | Times, sticking points and the next review date. |

`example.py` can run against any of your files, so it doubles as a test:

```
python 01_knn/example.py                       # solution.py (should print "All checks passed.")
python 01_knn/example.py practice              # your practice.py
python 01_knn/example.py day2_blank            # your blank-file rewrite
python 01_knn/example.py attempts/2026-10-04   # a dated timed attempt
```

Your file must use the same class and function names as `solution.py`.

## Practice routine for each algorithm

| Step | What to do | Where it goes |
|------|------------|---------------|
| Day 1 | Read `explained.py` and run `example.py`. Then fill in `practice.py` from the hints, peeking only when stuck. | `practice.py` + log |
| Day 2 | Rewrite from a blank file. Peek only when stuck, and note where. | `day2_blank.py` + log |
| Day 3+ | Rewrite timed, no peeking. | `attempts/YYYY-MM-DD.py` + log |
| 3–4 days later | Rewrite again from scratch (spaced repeat). | `attempts/YYYY-MM-DD.py` + log |

You're done with a step when `python example.py <your file>` prints "All checks passed." To reset `practice.py`, run `git checkout -- <folder>/practice.py`.

## Daily log

See [LOG.md](LOG.md).
