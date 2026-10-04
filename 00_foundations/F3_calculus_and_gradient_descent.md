# F3: Calculus and gradient descent

**Learning objectives.** After this lecture you can:

- explain what a derivative and a gradient tell you
- apply the chain rule (the whole of backpropagation is just the chain rule)
- run gradient descent by hand for a few steps
- explain what the learning rate does and what goes wrong when it's too big or too small

---

## 1. "Learning" = minimising a loss

A model has **parameters** (weights). A **loss function** $L$ measures how wrong the model's
predictions are, so smaller is better. Training means: *find the parameters that make $L$ small.*

We can't try every possible weight, so we follow the slope downhill.

## 2. Derivatives: the slope

The derivative $\frac{dL}{dw}$ says how fast $L$ changes when you nudge $w$:

- **positive:** increasing $w$ increases the loss, so move $w$ **down**
- **negative:** increasing $w$ decreases the loss, so move $w$ **up**
- **zero:** you're at a flat spot (hopefully the minimum)

The rules you need:

| Function | Derivative |
|----------|------------|
| $c$ (constant) | $0$ |
| $w^2$ | $2w$ |
| $a w + b$ | $a$ |
| $e^w$ | $e^w$ |
| $\ln w$ | $1/w$ |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| $\text{ReLU}(z)=\max(0,z)$ | $1$ if $z>0$, else $0$ |

**Numerical check (always works):** $\frac{dL}{dw} \approx \frac{L(w+\epsilon) - L(w-\epsilon)}{2\epsilon}$
for a tiny $\epsilon$. This is the "gradient check" used in the neural-network lab.

## 3. Gradients: many slopes at once

With many parameters $w = (w_1, \dots, w_f)$, the **gradient** $\nabla L$ is the vector of the
partial derivatives, one per parameter. It points in the direction of **steepest increase**, so
we step in the **opposite** direction.

> The gradient always has the **same shape** as the parameter it belongs to. If `W` is `(4, 3)`,
> then `dW` is `(4, 3)`.

## 4. The chain rule

If $L$ depends on $p$, which depends on $w$, then

$$\frac{dL}{dw} = \frac{dL}{dp} \cdot \frac{dp}{dw}$$

i.e. multiply the local slopes along the path.

**Worked example.** One sample, $x=3$, $y=4$, model $p = w x$, loss $L = (p - y)^2$, with $w = 2$.

| Step | Value |
|------|-------|
| forward: $p = wx$ | $2 \cdot 3 = 6$ |
| forward: $L = (p-y)^2$ | $(6-4)^2 = 4$ |
| local slope $\frac{dL}{dp} = 2(p-y)$ | $2 \cdot 2 = 4$ |
| local slope $\frac{dp}{dw} = x$ | $3$ |
| chain: $\frac{dL}{dw}$ | $4 \cdot 3 = 12$ |

The slope is positive, so decrease $w$. (The ideal $w$ is $4/3$.)

Backpropagation is exactly this table: a **forward** pass computes and stores the values, then
a **backward** pass multiplies the local slopes from the loss back to each weight.

## 5. Gradient descent

```
repeat many times:
    w = w - learning_rate * gradient
```

Continuing the example with learning rate $\eta = 0.01$:

| Step | $w$ | $p = 3w$ | $L$ | $dL/dw = 6(p-4)$ | new $w$ |
|------|-----|----------|-----|------------------|---------|
| 0 | 2.000 | 6.000 | 4.000 | 12.00 | 2.000 − 0.12 = 1.880 |
| 1 | 1.880 | 5.640 | 2.690 | 9.84 | 1.880 − 0.098 = 1.782 |
| … | → 1.333 | → 4 | → 0 | → 0 | |

The step size shrinks automatically as the slope flattens near the minimum.

**Learning rate:**

- **too small:** it works, but it takes forever
- **too big:** it overshoots the minimum, bounces around, and can **diverge** (the loss explodes to `inf`/`nan`)
- **just right:** the loss drops quickly, then levels off

Always plot or print the loss. If it goes up, lower the learning rate first.

**Variants (good to name in interviews):**

- **batch GD** uses all the data per step (what this course uses)
- **stochastic GD (SGD)** uses one sample per step
- **mini-batch GD** uses, say, 32 samples per step. This is the standard in deep learning.
- **Adam / momentum** are smarter update rules built on top of the gradient.

## 6. Convex vs non-convex

Linear and logistic regression have **convex** losses: a single bowl, so gradient descent always finds
the best answer. Neural networks are **non-convex**: many valleys, and where you end up depends on the
initialisation. That's why neural-network weights start random, and K-Means (also non-convex) depends on its initial centroids.

## Check yourself

1. $L(w) = (w - 5)^2$. What is $dL/dw$ at $w=2$? Which way should $w$ move?
2. Do one gradient-descent step on Q1 with $\eta = 0.1$.
3. $L = (\sigma(z) - y)$ where $z = wx$. Write $dL/dw$ using the chain rule.
4. Your loss goes `0.69, 0.71, 1.3, 9.8, nan`. What is the most likely cause?

<details>
<summary>Answers</summary>

1. $2(w-5) = -6$. It's negative, so increase $w$.
2. $w = 2 - 0.1 \cdot (-6) = 2.6$.
3. $\frac{dL}{dw} = \sigma(z)(1-\sigma(z)) \cdot x$.
4. The learning rate is too high, so it diverged.

</details>

**Next:** [F4: Probability](F4_probability.md)
