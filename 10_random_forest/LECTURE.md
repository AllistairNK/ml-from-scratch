# Week 7: Random Forest

**Prerequisites:** [Decision Trees](../09_decision_tree/LECTURE.md) (the forest reuses that tree), [F5 bias–variance](../00_foundations/F5_ml_basics.md).

**Learning objectives.** After this lecture you can:

- explain why averaging many trees beats one tree (variance reduction)
- explain the two sources of randomness: bootstrap rows and random features per split
- compute the out-of-bag fraction (≈ 36.8%)
- implement a random forest on top of a decision tree

---

## 1. The problem with one tree

A deep tree has **low bias** (it can fit anything) but **high variance**: change a few training rows and you get a very
different tree. We want to keep the low bias and kill the variance.

## 2. The wisdom of crowds

If you average $B$ models whose errors are **independent**, the variance of the average drops by a factor of $B$. Tree
errors are correlated (they see the same data), so we inject randomness to **de-correlate** them:

1. **Bagging (bootstrap aggregating):** each tree trains on a **bootstrap sample**, $n$ rows drawn **with replacement**.
2. **Feature subsampling:** at each split, a tree considers only a random subset of $m$ features (default $m = \sqrt{f}$ for classification).
   Without this, every tree would split on the same strong feature first and look alike.

Predict with a **majority vote** (classification) or the **mean** (regression).

## 3. Bootstrap maths: the out-of-bag fraction

The probability that a particular row is **never** picked in $n$ draws with replacement:

$$\left(1 - \frac1n\right)^n \xrightarrow{n\to\infty} e^{-1} \approx 0.368$$

(For n = 100: 0.366.) So each tree sees about **63.2%** of the unique rows. The unseen 36.8% are its **out-of-bag (OOB)** samples:
a free validation set. Each row can be scored by the trees that didn't train on it.

## 4. Worked example: voting

Five trees predict for three samples:

| | Sample 1 | Sample 2 | Sample 3 |
|---|---|---|---|
| Tree 1 | 1 | 0 | 1 |
| Tree 2 | 1 | 0 | 0 |
| Tree 3 | 0 | 0 | 1 |
| Tree 4 | 1 | 1 | 0 |
| Tree 5 | 1 | 0 | 0 |
| **Vote** | **1** (4–1) | **0** (4–1) | **0** (3–2) |

In code that's `all_preds` with shape `(n_trees, n_samples)`: one row per tree. Transpose it, so each row is a sample, and take the most common value.

The lab shows the effect: one deep tree gets 77% on clean test labels, and 30 trees get about 87–90%.

## 5. From maths to code

| Concept | Code |
|---------|------|
| $m = \sqrt f$ | `n_features = self.n_features or max(1, int(np.sqrt(n_cols)))` |
| bootstrap sample | `idxs = rng.choice(n_samples, n_samples, replace=True)` |
| a de-correlated tree | `DecisionTree(..., n_features=n_features, seed=int(rng.integers(1_000_000_000)))` |
| fit on the bootstrap | `tree.fit(X[idxs], y[idxs])` |
| all the votes | `np.array([tree.predict(X) for tree in self.trees])` gives `(n_trees, n)` |
| majority | `most_common(column)` for each column of `all_preds` |

## 6. Pitfalls and facts

- **Same seed for every tree** gives identical trees, so there's no benefit. Each tree needs its own randomness.
- **Forgetting `replace=True`** means every "bootstrap" is just a shuffle of the full data, so there's no diversity from bagging.
- **More trees never overfit** (the accuracy plateaus); they only cost time. The depth still matters a little.
- **Trees are independent**, so training parallelises perfectly (`n_jobs=-1` in sklearn).
- **Feature importance:** the total impurity decrease contributed by each feature, averaged over the trees.
- **RF vs boosting** (XGBoost, see [explain_only](../explain_only/xgboost.md)): RF builds deep, independent trees **in parallel** to cut variance. Boosting builds
  shallow trees **in sequence**, each fixing the previous ones' errors, to cut bias.

## 7. Tutorial questions

1. Why does feature subsampling help even though each tree becomes individually *worse*?
2. With 1000 training rows, roughly how many unique rows does each tree see?
3. Why is a random forest not very interpretable, even though trees are?
4. You set `n_features = f` (all of them). What algorithm do you have now?

<details>
<summary>Answers</summary>

1. It de-correlates the trees. The variance of an average depends on the correlation between its members, so less correlated trees average out much better, more than making up for each tree being weaker.
2. About 632.
3. The prediction is a vote over hundreds of different trees; there's no single flowchart to read.
4. Plain **bagged trees** (bagging without feature randomness).

</details>

## 8. Interview questions

<details>
<summary>How does a random forest reduce overfitting?</summary>

By averaging many high-variance, low-bias trees trained on bootstrap samples with random feature subsets. The randomness de-correlates their errors, so averaging cancels much of the variance without adding bias.
</details>

<details>
<summary>What is the OOB score?</summary>

Each sample is predicted only by the trees whose bootstrap missed it (about 37% of them). The accuracy of those predictions estimates the test accuracy without a separate validation set.
</details>

<details>
<summary>Random forest vs gradient boosting: when would you use each?</summary>

RF: a robust default with little tuning, parallel training, hard to overfit. Boosting: usually higher accuracy on tabular data after tuning (learning rate, depth, number of rounds), but sequential and easier to overfit.
</details>

## 9. Lab

Read [explained.py](explained.py) and run `python example.py` (about 15 seconds). Only `fit` and `predict` are new here; the tree comes from Weeks 6–7. Target: **8 minutes**.
