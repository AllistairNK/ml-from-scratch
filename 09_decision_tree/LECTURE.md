# Weeks 6–7: Decision Trees

**Prerequisites:** [F1 boolean masks and np.unique](../00_foundations/F1_numpy.md), [F5 overfitting](../00_foundations/F5_ml_basics.md), recursion in Python.

**New words?** [class](../00_basics/B1_glossary.md#class) · [label](../00_basics/B1_glossary.md#label) · [feature](../00_basics/B1_glossary.md#feature) · [threshold](../00_basics/B1_glossary.md#threshold) · [overfitting](../00_basics/B1_glossary.md#overfitting) · [mask](../00_basics/B1_glossary.md#mask) · or the full [glossary](../00_basics/B1_glossary.md).

**Learning objectives.** After this lecture you can:

- explain how a tree makes predictions and how it's grown greedily
- compute Gini impurity and the information gain of a split by hand
- write the recursive build and predict functions
- control overfitting with the depth and minimum-sample limits

---

## 1. The problem

**Classification** (or regression) with a model a human can read: a flowchart of yes/no questions such as
"income ≤ 50? → debt ≤ 20? → approve".

## 2. How a tree predicts

Start at the root. At each internal node ask "is $x_{\text{feature}} \le \text{threshold}$?", go left if yes and right if no,
and repeat until you reach a **leaf**, which stores the predicted class (the majority class of the training samples that ended there).

## 3. How a tree is grown (CART, greedy)

```
grow(samples, depth):
    if stop (pure, too deep, too few samples): return a leaf with the majority class
    for every feature, for every candidate threshold:
        measure how much the split reduces the impurity
    take the best split; recurse into the left and right halves
```

It's **greedy**: it picks the best split *now* without looking ahead. That's fast, but not globally optimal.

## 4. Measuring purity: Gini impurity

$$G = 1 - \sum_k p_k^2$$

where $p_k$ is the fraction of class $k$ in the node. $G = 0$ means the node is pure. For 2 classes the maximum is $0.5$ (a 50/50 split).

**Information gain** of a split is the parent impurity minus the size-weighted child impurity:

$$\text{Gain} = G_{\text{parent}} - \left(\frac{n_L}{n} G_L + \frac{n_R}{n} G_R\right)$$

(Entropy, $-\sum p_k \log_2 p_k$, is an alternative that usually gives similar trees.)

## 5. Worked example

One feature $x$ = 1, 2, 3, 4, 5, 6 with labels 0, 0, 1, 0, 1, 1. Parent: 3 vs 3, so $G = 1 - (0.5^2+0.5^2) = 0.5$.

| Threshold (left: x ≤ t) | Left labels | $G_L$ | Right labels | $G_R$ | Weighted | **Gain** |
|---|---|---|---|---|---|---|
| 1 | {0} | 0 | {0,1,0,1,1} | 0.48 | 0.40 | 0.100 |
| 2 | {0,0} | 0 | {1,0,1,1} | 0.375 | 0.25 | **0.250** |
| 3 | {0,0,1} | 0.444 | {0,1,1} | 0.444 | 0.444 | 0.056 |
| 4 | {0,0,1,0} | 0.375 | {1,1} | 0 | 0.25 | **0.250** |
| 5 | {0,0,1,0,1} | 0.48 | {1} | 0 | 0.40 | 0.100 |

Example: for t = 2, $G_R = 1 - (0.25^2 + 0.75^2) = 0.375$, and the weighted impurity is $\frac{2}{6}\cdot 0 + \frac{4}{6}\cdot0.375 = 0.25$.

t = 2 and t = 4 tie. Our code keeps the **first** one found (strict `>`), so it splits at $x \le 2$, and then
recursion keeps splitting the impure right side.

## 6. From maths to code

| Concept | Code |
|---------|------|
| a node | the `Node` class: `feature, threshold, left, right` or `value` (leaf) |
| Gini | `_, counts = np.unique(y, return_counts=True); p = counts / len(y); 1 - np.sum(p**2)` |
| stopping rules | `depth >= max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1` |
| candidate thresholds | `np.unique(X[:, feature])` |
| split mask | `left = X[:, feature] <= threshold` and the right side is `~left` |
| weighted child impurity | `(n_left * gini(y[left]) + n_right * gini(y[~left])) / n` |
| predict | `traverse` walks down recursively to a leaf |
| feature subsampling (for RF) | `self.rng.choice(n_cols, self.n_features_, replace=False)` |

## 7. Pitfalls

- **Overfitting:** an unlimited tree memorises the training set (100% train accuracy). Limit `max_depth`, `min_samples_split`, or prune.
- **`is_leaf` with class 0:** test `value is not None`, not `if value:` (0 is falsy).
- **Degenerate splits:** skip thresholds that send every sample one way.
- **No gain possible** (XOR-like data, where no single split helps): it becomes a leaf. A known weakness of greedy splitting.
- **Cost:** $O(f \cdot n \cdot t)$ per node ($t$ = the number of candidate thresholds). Real libraries sort each feature once and sweep it, making it $O(f\, n \log n)$.
- **Strengths:** no feature scaling needed, handles mixed feature types, interpretable. **Weakness:** high variance (a small data change can make a very different tree), which is why we use forests.

## 8. Tutorial questions

1. Compute the Gini impurity of a node with class counts (8, 2), and of (5, 5).
2. In the worked example, after splitting at $x \le 2$, what is the best split for the right node {3:1, 4:0, 5:1, 6:1}?
3. Why does `best_split` start with `best_gain = 0` rather than `-inf`?
4. A depth-20 tree gets 100% train and 90% test; depth-2 gets 87% train and 99% test (the example output). Explain.

<details>
<summary>Answers</summary>

1. $1 - (0.64 + 0.04) = 0.32$; $0.5$.
2. Parent {1,0,1,1}: G = 0.375. Try x ≤ 3: {1} | {0,1,1}, giving 0 + ¾·0.444 = 0.333, gain 0.042. Try x ≤ 4: {1,0} | {1,1}, giving ½·0.5 + 0 = 0.25, gain **0.125**. Try x ≤ 5: {1,0,1} | {1}, giving ¾·0.444 = 0.333, gain 0.042. So the best is x ≤ 4.
3. So that only splits that **actually reduce** the impurity are accepted. Otherwise a useless split could be chosen and the recursion would go on pointlessly.
4. The deep tree memorised the 10% flipped (noisy) labels: overfitting. The shallow tree captured the true rule and ignored the noise.

</details>

## 9. Interview questions

<details>
<summary>Gini vs entropy?</summary>

Both measure impurity and usually give similar splits. Gini is slightly cheaper to compute (no log); entropy has the information-theoretic meaning and leans slightly towards more balanced splits.
</details>

<details>
<summary>How does a regression tree differ?</summary>

The impurity is the variance (MSE) of the targets in the node, and a leaf predicts the mean of its targets.
</details>

<details>
<summary>What is pruning?</summary>

Pre-pruning means early stopping (max depth, min samples, min gain). Post-pruning means growing the full tree and then removing subtrees that don't improve the validation performance (cost-complexity pruning).
</details>

## 10. Lab

Read [explained.py](explained.py) and run `python example.py` (it prints the learned tree!). It's the longest implementation, so plan several days. Target: **20 minutes**.
