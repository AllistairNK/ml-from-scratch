# Week 2–3: K-Means Clustering

**Prerequisites:** [F1 broadcasting and boolean masks](../00_foundations/F1_numpy.md), [F2 distance](../00_foundations/F2_linear_algebra.md), [KNN](../01_knn/LECTURE.md) (the same distance trick).

**Learning objectives.** After this lecture you can:

- explain unsupervised learning and what K-Means optimises
- run Lloyd's algorithm by hand until it converges
- explain why the initialisation matters, and what k-means++ does
- choose k with the elbow method

---

## 1. The problem

**Clustering (unsupervised):** we have `X` with **no labels**, and we want to split it into $k$ groups of similar points.

## 2. What K-Means minimises

Each cluster $j$ has a **centroid** $\mu_j$ (its centre). Minimise the **inertia** (the within-cluster sum of squares):

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2$$

where $c_i$ is the cluster of point $i$. Small $J$ means tight clusters.

## 3. Lloyd's algorithm

```
pick k initial centroids (e.g. k random data points)
repeat until the centroids stop moving:
    ASSIGN:  each point joins its nearest centroid
    UPDATE:  each centroid moves to the mean of its points
```

**Why it converges:** the assign step can only lower $J$ (each point picks its closest centroid), and the update step can only lower
$J$ (the mean is the point that minimises the sum of squared distances to a set). $J$ never increases and is bounded below by 0, so the algorithm stops.
But it stops at a **local** minimum, not necessarily the best one.

## 4. Worked example (1-D, by hand)

Points: 1, 2, 3, 10, 11, 12. $k = 2$. Bad initial centroids: $\mu = (1, 2)$.

| Iteration | Assignments (closest centroid) | New centroids |
|-----------|-------------------------------|---------------|
| 1 | 1→μ₁; 2, 3, 10, 11, 12→μ₂ | μ₁ = 1, μ₂ = (2+3+10+11+12)/5 = **7.6** |
| 2 | 1, 2, 3→μ₁ (3 is closer to 1 than to 7.6); 10, 11, 12→μ₂ | μ₁ = **2**, μ₂ = **11** |
| 3 | unchanged | unchanged, so **converged** |

Final inertia: $(1-2)^2 + 0 + (3-2)^2 + (10-11)^2 + 0 + (12-11)^2 = 4$.

Even from a poor start, it found the obvious clusters in 2 iterations.

## 5. From maths to code

| Maths | Code |
|-------|------|
| random initial centroids | `self.centroids = X[rng.choice(len(X), self.k, replace=False)]` |
| squared distance to each centroid, `(n, k)` | `((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)` |
| assign | `np.argmin(dists, axis=1)` |
| update: the mean of the cluster's points | `X[labels == j].mean(axis=0)` for each j |
| convergence | `np.linalg.norm(new_centroids - self.centroids) < self.tol` |
| inertia $J$ | `np.sum((X - self.centroids[self.labels_]) ** 2)` |

## 6. Pitfalls and extensions

- **Initialisation:** a bad start gives a bad local minimum. Fixes: run several times with different seeds and keep the lowest inertia,
  or use **k-means++** (choose each new centroid with probability proportional to its squared distance from the nearest existing one, which spreads the centroids out).
- **Empty clusters:** if no point picks a centroid, `mean()` of an empty array is `nan`. Keep the old centroid (as we do) or re-seed it.
- **Choosing k:** the **elbow method** (plot the inertia vs k and look for the bend), or the silhouette score. Inertia always falls as k grows, so you can't just minimise it.
- **Assumptions:** roughly spherical, similar-size clusters. It fails on rings, moons and very different densities (use DBSCAN or GMMs).
- **Scale features first** (it's distance-based).

## 7. Tutorial questions

1. Redo the worked example starting from $\mu = (10, 11)$. Where does it converge?
2. Why does `predict` use squared distances without a sqrt?
3. Your inertia for k = 1..6 is 5900, 3100, 500, 450, 400, 365. Which k would you pick, and why?
4. Can K-Means' inertia ever **increase** between iterations?

<details>
<summary>Answers</summary>

1. Iteration 1: 1, 2, 3 and 10 are closer to 10 than to 11; 11 and 12 go to 11. So μ₁ = (1+2+3+10)/4 = 4 and μ₂ = 11.5.
   Iteration 2: 1, 2, 3→4; 10, 11, 12→11.5 (10 is 6 from 4 but only 1.5 from 11.5). So μ = (2, 11).
   Iteration 3: nothing changes. It reaches the same answer.
2. The sqrt doesn't change which centroid is nearest, so it's wasted work.
3. k = 3: the big drop stops there (the elbow).
4. No. Both steps can only lower or keep $J$.

</details>

## 8. Interview questions

<details>
<summary>K-Means vs KNN?</summary>

They are totally different. K-Means is unsupervised clustering; KNN is supervised classification. The only thing they share is the distance computation.
</details>

<details>
<summary>What is the complexity per iteration?</summary>

O(n·k·f) for the assign step, O(n·f) for the update step.
</details>

<details>
<summary>How does k-means++ work?</summary>

Pick the first centroid uniformly at random. Pick each next centroid from the data with probability ∝ D(x)², the squared distance to the nearest chosen centroid. This gives O(log k)-competitive results in expectation.
</details>

## 9. Lab

Read [explained.py](explained.py), run `python example.py` (look at the elbow table), fill in your practice file. Target: **12 minutes**.
