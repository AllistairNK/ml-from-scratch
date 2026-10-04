# Week 2: Logistic Regression

**Prerequisites:** [Linear Regression](../02_linear_regression/LECTURE.md), [F4 sigmoid and cross-entropy](../00_foundations/F4_probability.md).

**Learning objectives.** After this lecture you can:

- explain why we wrap a linear model in a sigmoid for classification
- write the binary cross-entropy loss and its gradient, $X^\top(p - y)/n$
- do one training step by hand
- explain the decision threshold and the precision/recall trade-off

---

## 1. The problem

**Binary classification:** predict 0 or 1 (spam or not, approve or not), ideally as a **probability**.

## 2. The model

Take the linear score and squash it into a probability:

$$z = x\cdot w + b, \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y = 1 \mid x)$$

- $z$ is the **log-odds**: $z = \log\frac{p}{1-p}$. Each $w_j$ is how much the log-odds move per unit of feature $j$.
- The decision boundary $p = 0.5 \iff z = 0$ is a **straight line** (a hyperplane), so logistic regression is a *linear classifier*.

## 3. The loss: binary cross-entropy

Why not MSE? With a sigmoid inside, MSE is non-convex and gives tiny gradients when the model is confidently wrong. Use
the negative log-likelihood instead:

$$L = -\frac1n \sum_i \big[ y_i \log p_i + (1-y_i)\log(1-p_i) \big]$$

If $y = 1$, only $-\log p$ counts; if $y = 0$, only $-\log(1-p)$ counts. Confident and wrong means a huge loss.

## 4. The gradient: the beautiful simplification

The chain rule for one sample: $\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$ and $\frac{\partial p}{\partial z} = p(1-p)$.
Multiply them:

$$\frac{\partial L}{\partial z} = \left(-\frac{y}{p} + \frac{1-y}{1-p}\right) p(1-p) = -y(1-p) + (1-y)p = p - y$$

So:

$$\nabla_w L = \frac1n X^\top (p - y), \qquad \frac{\partial L}{\partial b} = \frac1n\sum(p-y)$$

This has **the same form as linear regression** (prediction minus truth, times the input), without the factor 2.
Remember "p minus y" and you can derive the rest.

## 5. Worked example: one step by hand

Data: $x = (1, 3)$, $y = (0, 1)$. Start with $w = 0$, $b = 0$, $\eta = 1.0$.

| Quantity | Value |
|----------|-------|
| $z$ | (0, 0) |
| $p = \sigma(z)$ | (0.5, 0.5) |
| loss | $-\frac12[\log 0.5 + \log 0.5] = \ln 2 = $ **0.693** |
| $p - y$ | (0.5, −0.5) |
| $dw = \frac12(1 \cdot 0.5 + 3 \cdot (-0.5))$ | **−0.5** |
| $db = \frac12(0.5 - 0.5)$ | **0** |
| new $w$ | 0 − 1.0 × (−0.5) = **0.5** |
| new $p$ | (σ(0.5), σ(1.5)) = (0.622, 0.818) |
| new loss | **0.588** |

The loss went down. Note that the starting loss is always ln 2 = 0.693 with zero weights. That makes a great sanity check.

## 6. From maths to code

| Maths | Code |
|-------|------|
| $\sigma(z)$, with overflow protection | `z = np.clip(z, -500, 500); return 1 / (1 + np.exp(-z))` |
| $p = \sigma(Xw + b)$ | `p = sigmoid(X @ self.w + self.b)` |
| the loss, with log(0) protection | `p_safe = np.clip(p, 1e-12, 1 - 1e-12)`, then the BCE formula |
| $\frac1n X^\top(p-y)$ | `dw = (X.T @ (p - y)) / n_samples` |
| the decision rule | `(self.predict_proba(X) >= threshold).astype(int)` |

## 7. Pitfalls and extensions

- **Numerical stability:** `np.exp(-z)` overflows for z < −710, and `log(0) = -inf`. Clip both.
- **Threshold:** 0.5 isn't sacred. Lower it to catch more positives (higher recall), raise it for higher precision.
- **Perfectly separable data:** the weights grow without bound (the loss keeps falling towards 0). Regularisation (an L2 penalty) fixes it.
- **Multiclass:** use softmax with K weight vectors (multinomial / softmax regression), which is exactly the output layer of the neural network in Week 4–5. Or one-vs-rest.
- **Still linear:** it can't fit XOR or circles without feature engineering. Neural networks learn those features.

## 8. Tutorial questions

1. What is $\sigma(0)$, $\sigma(2)$ and $\sigma(-2)$? (Note $\sigma(-z) = 1 - \sigma(z)$.)
2. A sample has $y = 1$ and $p = 0.01$. What is its loss? What if $p = 0.99$?
3. Show that the decision boundary for 2 features is a line. Write it as $x_2 = \dots$
4. In the example, why is the bias gradient exactly 0 on the first step?

<details>
<summary>Answers</summary>

1. 0.5, 0.881, 0.119.
2. $-\ln 0.01 = 4.61$; $-\ln 0.99 = 0.01$.
3. $w_1 x_1 + w_2 x_2 + b = 0 \Rightarrow x_2 = -(w_1 x_1 + b)/w_2$: a straight line.
4. With p = 0.5 for both samples and one sample of each class, the residuals +0.5 and −0.5 cancel.

</details>

## 9. Interview questions

<details>
<summary>Why is it called "regression" if it classifies?</summary>

It regresses the log-odds (a continuous quantity) linearly on the features; the classification comes from thresholding the probability.
</details>

<details>
<summary>Derive the gradient of the BCE with a sigmoid.</summary>

Section 4: dL/dz = p − y, then the chain rule through z = Xw + b gives Xᵀ(p − y)/n.
</details>

<details>
<summary>Why not train with MSE?</summary>

With a sigmoid, MSE is non-convex, and its gradient includes p(1−p), which vanishes when the model is confidently wrong. BCE is convex and its gradient stays large.
</details>

## 10. Lab

Read [explained.py](explained.py), run `python example.py`, fill in your practice file. Target: **10 minutes**.
