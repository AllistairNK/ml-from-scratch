# Week 6: Principal Component Analysis (PCA)

**Prerequisites:** [F2 eigenvectors](../00_foundations/F2_linear_algebra.md), [F1 centering and axis](../00_foundations/F1_numpy.md).

**New words?** [variance](../00_basics/B1_glossary.md#variance) · [mean](../00_basics/B1_glossary.md#mean) · [matrix](../00_basics/B1_glossary.md#matrix) · [vector](../00_basics/B1_glossary.md#vector) · [feature](../00_basics/B1_glossary.md#feature) · [supervised and unsupervised](../00_basics/B1_glossary.md#supervised-and-unsupervised) · or the full [glossary](../00_basics/B1_glossary.md).

**Learning objectives.** After this lecture you can:

- explain PCA as "find the directions of maximum variance"
- compute a covariance matrix, its eigenvectors and the projections
- choose the number of components with the explained-variance ratio
- relate PCA to SVD

---

## 1. The problem

**Dimensionality reduction (unsupervised):** you have many correlated features and want fewer, uncorrelated ones that keep
most of the information. It's used for visualisation (down to 2-D), compression, denoising and speeding up other models.

## 2. Intuition

Picture a cloud of points shaped like a tilted cigar. Most of the spread lies along the cigar's long axis. PCA rotates
the coordinate system so that:

- axis 1 (**PC1**) points along the direction of **greatest variance**
- PC2 is perpendicular to PC1 and captures the most of what remains, and so on

Keeping only the first few axes gives a compressed version that loses the least variance.

## 3. The recipe

1. **Center:** $X_c = X - \bar x$
2. **Covariance:** $C = \frac{1}{n-1} X_c^\top X_c$, shape `(f, f)`. $C_{jk}$ measures how features $j$ and $k$ move together.
3. **Eigendecomposition:** $C v = \lambda v$. The eigenvectors $v$ are the principal directions, and the eigenvalue $\lambda$ is the variance along each one.
4. **Sort** by $\lambda$, largest first, and keep the top $k$ eigenvectors as rows of `components` $(k, f)$.
5. **Project:** $Z = X_c\,V_k^\top$, shape `(n, k)`.
6. (Optional) **Reconstruct:** $\hat X = Z V_k + \bar x$.

**Explained-variance ratio:** $\lambda_i / \sum_j \lambda_j$. Keep enough components to reach, say, 95%.

## 4. Worked example

Points: (1, 2), (2, 3), (3, 5), (4, 6).

| Step | Result |
|------|--------|
| mean | (2.5, 4) |
| centered $X_c$ | (−1.5, −2), (−0.5, −1), (0.5, 1), (1.5, 2) |
| $C = X_c^\top X_c / 3$ | $\begin{pmatrix}1.667 & 2.333\\ 2.333 & 3.333\end{pmatrix}$ |
| eigenvalues | 4.978 and 0.022 |
| PC1 (eigenvector for 4.978) | (0.576, 0.817) |
| explained-variance ratio | **99.55%** and 0.45% |
| 1-D projection $Z = X_c \cdot \text{PC1}$ | −2.50, −1.11, 1.11, 2.50 |

The 2-D data is almost perfectly a line, so one number per point keeps 99.5% of the variance.
(The eigenvector's sign is arbitrary: (−0.576, −0.817) is equally valid.)

## 5. From maths to code

| Maths | Code |
|-------|------|
| center | `self.mean = X.mean(axis=0); X_centered = X - self.mean` |
| covariance | `X_centered.T @ X_centered / (len(X) - 1)` |
| eigendecomposition | `np.linalg.eigh(cov)` (ascending order! the eigenvectors are the **columns**) |
| sort descending | `order = np.argsort(eigvals)[::-1]`, then `eigvecs[:, order]` |
| top k as rows | `eigvecs[:, :k].T` gives `(k, f)` |
| project / reconstruct | `(X - self.mean) @ self.components.T` and `Z @ self.components + self.mean` |

## 6. Pitfalls

- **Forgetting to center:** PC1 then points at the data mean instead of along the spread.
- **`eigh` order:** it returns **ascending** eigenvalues, so forgetting to reverse them gives the *least* important components.
- **Rows vs columns:** `eigh` gives the eigenvectors as columns. Mix this up and you project onto garbage.
- **Scale:** PCA is variance-based, so a feature in big units dominates. Standardise first when the units differ.
- **Transform new data with the training mean**, not the new data's mean.
- **It's linear:** it can't unroll curved manifolds (use t-SNE/UMAP for visualisation, or kernel PCA).
- **The sign is arbitrary:** compare the components to other implementations up to ±.

## 7. Tutorial questions

1. Why are the principal components orthogonal?
2. If all 3 eigenvalues of a covariance matrix are equal, what does the data look like, and is PCA useful?
3. After projecting, what is the variance of column 1 of $Z$?
4. How do you get the same components from `np.linalg.svd`?

<details>
<summary>Answers</summary>

1. $C$ is symmetric, and the eigenvectors of a symmetric matrix (with distinct eigenvalues) are orthogonal.
2. A round "ball" with no preferred direction. No: every direction keeps the same variance, so there is nothing to compress.
3. Exactly $\lambda_1$ (the example checks this).
4. `U, S, Vt = np.linalg.svd(X_centered)`: the rows of `Vt` are the components, and $\lambda_i = S_i^2/(n-1)$. SVD is more numerically stable because it never forms $X^\top X$.

</details>

## 8. Interview questions

<details>
<summary>PCA vs feature selection?</summary>

PCA builds **new** features (linear combinations of all the original ones); feature selection keeps a subset of the original features. PCA components are harder to interpret.
</details>

<details>
<summary>Is PCA supervised?</summary>

No. It ignores the labels. The highest-variance direction isn't necessarily the most predictive one (contrast with LDA, which uses the labels).
</details>

## 9. Lab

Read [explained.py](explained.py) and run `python example.py`: the explained variance, the reconstruction, and the cross-check against SVD. Target: **8 minutes**.
