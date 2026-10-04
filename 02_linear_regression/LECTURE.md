# Week 1–2: Linear Regression with Gradient Descent

**Prerequisites:** [F2 dot products and matrices](../00_foundations/F2_linear_algebra.md), [F3 gradient descent](../00_foundations/F3_calculus_and_gradient_descent.md).

**Learning objectives.** After this lecture you can:

- write down the model, the loss and the gradients
- derive the gradient of the MSE
- do one gradient-descent step by hand
- explain feature scaling, the learning rate and the closed-form alternative

---

## 1. The problem

**Regression:** predict a number (a price, a temperature) from features.

## 2. The model

Predict with a weighted sum of the features plus a bias:

$$\hat y = w_1 x_1 + w_2 x_2 + \dots + w_f x_f + b = x \cdot w + b$$

For all the samples at once: $\hat{\mathbf y} = X w + b$, where $X$ is `(n, f)`, $w$ is `(f,)` and $b$ is a scalar.

- $w_j$ is how much $\hat y$ changes when feature $j$ goes up by 1
- $b$ is the prediction when all the features are 0 (the intercept)

## 3. The loss: mean squared error

$$L(w, b) = \frac{1}{n}\sum_{i=1}^{n} (\hat y_i - y_i)^2$$

Squaring makes every error positive and punishes big errors heavily. The loss is a smooth bowl
(convex), so gradient descent will find the best $w, b$.

## 4. The gradients (derivation)

Let $e_i = \hat y_i - y_i$ (the residual). By the chain rule,

$$\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_i 2 e_i \cdot \frac{\partial \hat y_i}{\partial w_j} = \frac{2}{n}\sum_i e_i\, x_{ij}$$

$$\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i e_i$$

In matrix form: $\nabla_w L = \frac{2}{n} X^\top e$, which has shape `(f,)`, the same as $w$.

Then repeat: $w \leftarrow w - \eta \nabla_w L$, $b \leftarrow b - \eta \frac{\partial L}{\partial b}$.

## 5. Worked example: one step by hand

Data: $x = (1, 2, 3)$, $y = (2, 4, 5)$, one feature. Start with $w = 0$, $b = 0$, $\eta = 0.1$.

| Quantity | Value |
|----------|-------|
| $\hat y = wx + b$ | (0, 0, 0) |
| $e = \hat y - y$ | (−2, −4, −5) |
| MSE | (4 + 16 + 25) / 3 = **15.0** |
| $\partial L / \partial w = \frac{2}{3}\sum e_i x_i$ | $\frac23(-2 - 8 - 15) = -16.67$ |
| $\partial L / \partial b = \frac{2}{3}\sum e_i$ | $\frac23(-11) = -7.33$ |
| new $w = 0 - 0.1 \times (-16.67)$ | **1.667** |
| new $b = 0 - 0.1 \times (-7.33)$ | **0.733** |
| new MSE | **0.234** |

One step took the loss from 15 down to 0.23. The exact best fit (closed form) is $w = 1.5$, $b = 0.667$, and further steps converge there.

## 6. From maths to code

| Maths | Code |
|-------|------|
| $w = 0, b = 0$ | `self.w = np.zeros(n_features); self.b = 0.0` |
| $\hat y = Xw + b$ | `y_pred = X @ self.w + self.b` |
| $e = \hat y - y$ | `error = y_pred - y` |
| $\frac{2}{n} X^\top e$ | `dw = (2 / n_samples) * (X.T @ error)` |
| $\frac{2}{n}\sum e$ | `db = (2 / n_samples) * np.sum(error)` |
| update | `self.w -= self.lr * dw` and the same for `b` |

## 7. Pitfalls and extensions

- **Learning rate too big:** the loss explodes to `inf`/`nan`. Too small: very slow. The example demonstrates divergence.
- **Feature scaling:** features with very different ranges make the bowl a long, thin valley, and GD zig-zags. Standardise.
- **Sign of the error:** using `y - y_pred` flips the gradient sign. Then you must `+=` the update. Pick one convention and stick to it.
- **Closed form** (the normal equation): $w = (X^\top X)^{-1} X^\top y$ (append a column of ones for $b$). It's exact, but
  costs $O(f^3)$ and fails when $X^\top X$ is singular. GD scales to huge data.
- **Regularisation:** Ridge adds $\lambda\lVert w\rVert^2$ (gradient $+2\lambda w$), which shrinks the weights. Lasso adds $\lambda\lVert w\rVert_1$, which gives sparse weights.

## 8. Tutorial questions

1. Do a **second** GD step from $w = 1.667$, $b = 0.733$ on the worked example. Does the loss go down?
2. Why is it safe to initialise $w$ to zeros here, but not in a neural network?
3. What is the shape of `X.T @ error` if `X` is `(500, 3)`?
4. Your loss goes `24, 61, 380, 2e4, inf`. What should you change?

<details>
<summary>Answers</summary>

1. $\hat y = (2.4, 4.067, 5.733)$, $e = (0.4, 0.067, 0.733)$, $dw = \frac23(0.4 + 0.133 + 2.2) = 1.822$,
   $db = \frac23(1.2) = 0.8$. So $w = 1.485$, $b = 0.653$, and the MSE drops from 0.234 to 0.058. Yes, it goes down.
2. The loss is convex, so any starting point works. A network with all-zero weights would make every hidden unit identical (symmetry).
3. `(3,)`.
4. Lower the learning rate (and check the feature scaling).

</details>

## 9. Interview questions

<details>
<summary>Derive the gradient of the MSE w.r.t. w.</summary>

$L=\frac1n\lVert Xw+b-y\rVert^2$, so $\nabla_w L = \frac2n X^\top(Xw+b-y)$. (Section 4.)
</details>

<details>
<summary>GD vs the normal equation: when would you use each?</summary>

The normal equation for small f (exact, no learning rate). GD/SGD for large n or f, streaming data, or when (XᵀX) is ill-conditioned.
</details>

<details>
<summary>What assumptions does linear regression make?</summary>

A linear relationship, independent errors, constant error variance (homoscedasticity), no perfect multicollinearity. Normally distributed errors are needed only for the classical confidence intervals.
</details>

## 10. Lab

Read [explained.py](explained.py), run `python example.py`, then fill in your practice file. Target: **8 minutes**.
