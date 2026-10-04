# ML From Scratch

A self-paced course for relearning the core ML algorithms by implementing them from a blank file in NumPy.

**Goal:** each algorithm "interview-ready", meaning I can write it from a blank file within its target time
(8–20 min) without looking anything up.
**Pace:** 1–2 hrs/day. Finish the 6 priority algorithms first, because a technical round could come within 2–3 weeks.

## Getting started

```
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python course.py              # opens the course app at http://127.0.0.1:8000
```

The **course app** is a local web page with:

- the lectures, with the maths rendered
- an editor for your practice code, with a **Run checks** button
- a progress tracker that ticks steps off automatically when your code passes, times your attempts, and tells you when a spaced review is due

Everything is saved to [progress.json](progress.json) and your files in the repo. Click **Commit progress** in
the app (or run `git commit` yourself) to save a snapshot to git. Push when you want a backup on GitHub.
The app needs an internet connection to load its libraries (Markdown, maths and editor) from a CDN.

## Syllabus

| Week | Unit | Lecture | Target time |
|------|------|---------|------------:|
| 0 | Foundations: NumPy, linear algebra, calculus, probability, ML basics | [00_foundations](00_foundations/README.md) | |
| 1 | K-Nearest Neighbours | [lecture](01_knn/LECTURE.md) | 10 min |
| 1–2 | Linear Regression (gradient descent) | [lecture](02_linear_regression/LECTURE.md) | 8 min |
| 2 | Logistic Regression | [lecture](03_logistic_regression/LECTURE.md) | 10 min |
| 2–3 | K-Means | [lecture](04_kmeans/LECTURE.md) | 12 min |
| 3–4 | Neural Network + Backprop | [lecture](05_neural_network/LECTURE.md) | 20 min |
| 4–5 | Attention / Multi-Head | [lecture](06_attention/LECTURE.md) | 15 min |
| 5 | Gaussian Naive Bayes | [lecture](07_naive_bayes/LECTURE.md) | 10 min |
| 6 | PCA | [lecture](08_pca/LECTURE.md) | 8 min |
| 6–7 | Decision Tree | [lecture](09_decision_tree/LECTURE.md) | 20 min |
| 7 | Random Forest | [lecture](10_random_forest/LECTURE.md) | 8 min |
| 8 | Explain, don't code: [SVM](explain_only/svm.md), [XGBoost](explain_only/xgboost.md) | your own notes | |

Weeks 1–5 are the priority set. If an interview comes early, do those first and skim the rest.

## How each unit works

Every algorithm unit has eight steps. The app ticks the "auto" ones when your code passes the checks.

| # | Step | How |
|---|------|-----|
| 1 | Read the lecture | `LECTURE.md`: intuition, maths, a worked example by hand, maths-to-code mapping, pitfalls |
| 2 | Tutorial questions | at the end of the lecture; the answers are hidden until you click |
| 3 | Study the code | `explained.py` (every line commented), then run `example.py` |
| 4 | Practice (auto) | fill in your copy of `practice.py` (one numbered hint per line) until the checks pass |
| 5 | Blank rewrite (auto) | `day2_blank.py`: everything from memory, peeking only when stuck |
| 6 | Timed attempt (auto) | `attempts/DATE.py` with a running timer |
| 7 | Spaced repeat (auto) | another timed pass 3+ days after the first |
| 8 | Interview-ready (auto) | a timed pass under the target time without peeking |

## Files in each unit folder

| File | What it is |
|------|------------|
| `LECTURE.md` | The lecture notes |
| `solution.py` | The clean, correct implementation |
| `explained.py` | The same code with a comment above every line explaining what it does, why, and the array shapes |
| `practice.py` | The hints template (the app copies it to `my_practice.py` for you to edit; the template is never changed) |
| `example.py` | A usage example with sanity checks; it doubles as the test for your code |
| `LOG.md` | Your own notes |

## Without the app

Everything also works from the terminal. `example.py` runs against whichever file you name:

```
python 01_knn/example.py                       # solution.py (should print "All checks passed.")
python 01_knn/example.py my_practice           # your practice copy (cp practice.py my_practice.py first)
python 01_knn/example.py day2_blank            # your blank-file rewrite
python 01_knn/example.py attempts/2026-10-04   # a dated attempt
```

Your file must use the same class and function names as `solution.py`. Progress is only tracked
automatically when the checks are run from the app.
