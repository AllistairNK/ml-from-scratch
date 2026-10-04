# Week 5: Gaussian Naive Bayes

**Prerequisites:** [F4 Bayes' rule, Gaussians, logs](../00_foundations/F4_probability.md), [F1 boolean masks](../00_foundations/F1_numpy.md).

**Learning objectives.** After this lecture you can:

- derive the Naive Bayes classifier from Bayes' rule
- say what "naive" assumes, and why it works anyway
- classify a point by hand with Gaussian likelihoods
- explain why the code works in log space

---

## 1. The problem

**Classification**, often with many features and little data (text, spam, medical screening). It's fast and needs no gradient descent.

## 2. From Bayes' rule to a classifier

$$P(c \mid x) \propto P(c)\, P(x \mid c)$$

Estimating $P(x \mid c)$ for a whole feature vector is hard. The **naive assumption**: the features are
**independent given the class**:

$$P(x \mid c) = \prod_{j=1}^{f} P(x_j \mid c)$$

Each factor is now 1-D and easy. **Gaussian** NB models each one as a normal distribution with a
per-class, per-feature mean $\mu_{cj}$ and variance $\sigma^2_{cj}$.

Predict the class with the largest posterior. In logs (to avoid underflow):

$$\hat c = \arg\max_c \Big[\log P(c) + \sum_j \log \mathcal N(x_j; \mu_{cj}, \sigma^2_{cj})\Big]$$

$$\log \mathcal N(x;\mu,\sigma^2) = -\tfrac12\Big[\log(2\pi\sigma^2) + \tfrac{(x-\mu)^2}{\sigma^2}\Big]$$

**Training** is just counting and averaging: the priors, means and variances per class. There's no iteration.

## 3. Worked example (1 feature)

Class 0: {1, 2, 3}. Class 1: {6, 7, 8}. Query $x = 4$.

| | Class 0 | Class 1 |
|---|---|---|
| prior | 3/6 = 0.5 | 0.5 |
| mean $\mu$ | 2 | 7 |
| variance $\sigma^2$ (divide by n) | ((1)²+0+(1)²)/3 = 0.667 | 0.667 |
| $\log \mathcal N(4)$ | $-\frac12[\log(2\pi\cdot0.667) + \frac{(4-2)^2}{0.667}] = -3.716$ | $-\frac12[1.432 + \frac{9}{0.667}] = -7.466$ |
| + log prior (−0.693) | **−4.409** | **−8.159** |

Class 0 wins. The posterior: $\frac{e^{-4.409}}{e^{-4.409}+e^{-8.159}} = $ **0.977**.

Point 4 is 2 units from class 0's mean but 3 from class 1's. With such tight variances, that's decisive.

## 4. From maths to code

| Maths | Code |
|-------|------|
| priors | `np.array([np.mean(y == c) for c in self.classes])` |
| $\mu_{cj}$, $\sigma^2_{cj}$ | `X[y == c].mean(axis=0)`, `X[y == c].var(axis=0) + var_smoothing` |
| $(x_j - \mu_{cj})$ for all the samples and classes | `X[:, None, :] - self.means[None, :, :]` gives `(n, C, f)` |
| log Gaussian, summed over the features | `-0.5 * (log(2πσ²) + diff² / σ²)` then `.sum(axis=2)` |
| + log prior, then argmax | `np.log(self.priors) + ...`, then `np.argmax(..., axis=1)` |
| normalised posterior | subtract the row max, `exp`, divide by the row sum (log-sum-exp) |

## 5. Pitfalls and extensions

- **Zero variance** (a constant feature in one class) means division by zero. Add `var_smoothing`.
- **Underflow:** never multiply the raw probabilities. Sum logs.
- **Correlated features** violate the assumption: the evidence is double-counted, so the probabilities come out **overconfident**. The *ranking* (argmax) is often still good.
- **Other variants:** **Multinomial NB** (word counts, the classic spam filter) and **Bernoulli NB** (binary features). For counts, use **Laplace smoothing** (add 1 to every count) so an unseen word doesn't zero out the whole product.
- **Generative vs discriminative:** NB models $P(x \mid c)$ (how the data is generated); logistic regression models $P(c \mid x)$ directly.
  NB learns fast with little data, while LR usually wins with lots of data.

## 6. Tutorial questions

1. In the worked example, at what $x$ are the two classes equally likely?
2. If class 1 had prior 0.9 instead of 0.5, would $x = 4$ change its prediction?
3. Why does `predict` not need the normalisation step that `predict_proba` does?
4. Two features are exact copies of each other. What happens to the predicted probabilities?

<details>
<summary>Answers</summary>

1. With equal priors and equal variances, it's the midpoint between the means: 4.5.
2. Class 1's score becomes −7.466 + ln 0.9 = −7.571, and class 0's becomes −3.716 + ln 0.1 = −6.018. Class 0 still wins.
3. Normalising divides every class by the same number, so the argmax is unchanged.
4. The evidence from that feature is counted twice, so the posteriors become more extreme (overconfident).

</details>

## 7. Interview questions

<details>
<summary>Why is it called "naive", and why does it work anyway?</summary>

It assumes conditional independence of the features, which is almost never true. Classification only needs the correct argmax, not calibrated probabilities, and the bias from the assumption often doesn't flip the ranking. Its low variance also helps with little data.
</details>

<details>
<summary>What is Laplace smoothing?</summary>

Adding a pseudo-count (usually 1) to every feature/class count, so that an unseen feature value doesn't give a zero probability that wipes out the whole product.
</details>

## 8. Lab

Read [explained.py](explained.py) and run `python example.py`. Note the string class labels and the 500-feature underflow check. Target: **10 minutes**.
