# Weeks 3–4: Neural Network and Backpropagation

**Prerequisites:** [F3 chain rule](../00_foundations/F3_calculus_and_gradient_descent.md), [F4 softmax and cross-entropy](../00_foundations/F4_probability.md), [Logistic Regression](../03_logistic_regression/LECTURE.md) (a NN is stacked logistic-style layers).

**Learning objectives.** After this lecture you can:

- explain why we need hidden layers and non-linear activations
- write the forward pass of a 2-layer MLP with shapes
- derive and compute the backward pass (backprop) by hand
- verify gradients with a numerical gradient check

---

## 1. Why a neural network?

Logistic regression draws **one straight line**. Many problems (spirals, images, XOR) need curved
boundaries. A neural network **learns its own features**: the hidden layer transforms the input
into a new space where the classes *are* linearly separable, and the output layer is just
softmax regression on those learned features.

## 2. The architecture (a 2-layer MLP)

```
X (n,d) --[W1 (d,h), b1]--> z1 (n,h) --ReLU--> a1 (n,h) --[W2 (h,c), b2]--> z2 (n,c) --softmax--> probs (n,c)
```

$$z_1 = XW_1 + b_1,\quad a_1 = \text{ReLU}(z_1),\quad z_2 = a_1W_2 + b_2,\quad p = \text{softmax}(z_2)$$

$$L = -\frac1n\sum_i \log p_{i, y_i}$$

**Why the activation?** Without it, $W_2(W_1x) = (W_2W_1)x$ is still linear: stacking linear layers adds no power.
ReLU ($\max(0, z)$) is the simplest non-linearity that works well.

## 3. Backpropagation: the chain rule, layer by layer

Work backwards from the loss. The key results:

| Step | Gradient | Shape | Why |
|------|----------|-------|-----|
| softmax + CE | $dz_2 = (p - \text{onehot}(y)) / n$ | (n,c) | the same "p − y" as logistic regression |
| layer 2 weights | $dW_2 = a_1^\top dz_2$ | (h,c) | $z_2 = a_1W_2$, so each weight's gradient is input × output-gradient |
| layer 2 bias | $db_2 = \sum_{\text{rows}} dz_2$ | (c,) | the bias is added to every row |
| into the hidden layer | $da_1 = dz_2 W_2^\top$ | (n,h) | send the gradient back through the weights |
| through ReLU | $dz_1 = da_1 \odot [z_1 > 0]$ | (n,h) | ReLU's slope is 1 where it was active, 0 elsewhere |
| layer 1 weights | $dW_1 = X^\top dz_1$ | (d,h) | the same pattern as $dW_2$ |
| layer 1 bias | $db_1 = \sum_{\text{rows}} dz_1$ | (h,) | |

**The pattern to memorise** for a linear layer $z = aW + b$, given the upstream gradient $dz$:

- $dW = a^\top dz$
- $db = $ `dz.sum(axis=0)`
- $da = dz\,W^\top$

Check the shapes; they force the transposes.

## 4. Worked example: forward and backward by hand

One sample $x = (1, 2)$, true class $y = 1$, 2 hidden units, 2 classes, biases 0.

$$W_1 = \begin{pmatrix}1 & -1\\ 0.5 & 1\end{pmatrix}, \quad W_2 = \begin{pmatrix}1 & 0\\ -1 & 1\end{pmatrix}$$

**Forward:**

| Quantity | Calculation | Value |
|----------|-------------|-------|
| $z_1 = xW_1$ | $(1\cdot1 + 2\cdot0.5,\ 1\cdot(-1) + 2\cdot1)$ | (2, 1) |
| $a_1 = \text{ReLU}(z_1)$ | both positive | (2, 1) |
| $z_2 = a_1W_2$ | $(2\cdot1 + 1\cdot(-1),\ 2\cdot0 + 1\cdot1)$ | (1, 1) |
| $p = \text{softmax}(z_2)$ | equal scores | (0.5, 0.5) |
| loss | $-\log p_1 = -\log 0.5$ | 0.693 |

**Backward** (n = 1):

| Quantity | Calculation | Value |
|----------|-------------|-------|
| $dz_2 = p - \text{onehot}(1)$ | (0.5, 0.5) − (0, 1) | (0.5, −0.5) |
| $dW_2 = a_1^\top dz_2$ | the column (2, 1) times the row (0.5, −0.5) | $\begin{pmatrix}1 & -1\\ 0.5 & -0.5\end{pmatrix}$ |
| $da_1 = dz_2 W_2^\top$ | $(0.5\cdot1 + (-0.5)\cdot0,\ 0.5\cdot(-1) + (-0.5)\cdot1)$ | (0.5, −1) |
| $dz_1 = da_1 \odot [z_1>0]$ | both units were active | (0.5, −1) |
| $dW_1 = x^\top dz_1$ | the column (1, 2) times the row (0.5, −1) | $\begin{pmatrix}0.5 & -1\\ 1 & -2\end{pmatrix}$ |

Interpretation: $dz_2$ says "push class 1's score up, class 0's down", and that signal flows back to every weight.

## 5. From maths to code

| Maths | Code |
|-------|------|
| He init | `rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_hidden))` |
| forward (cache!) | `self.z1 = X @ self.W1 + self.b1`, `self.a1 = relu(self.z1)`, … |
| $p - \text{onehot}(y)$, divided by n | `dz2 = self.probs.copy(); dz2[np.arange(n), y] -= 1; dz2 /= n` |
| $dW_2 = a_1^\top dz_2$ | `dW2 = self.a1.T @ dz2` |
| through ReLU | `dz1 = (dz2 @ self.W2.T) * (self.z1 > 0)` |
| gradient check | `example.py` compares with $(L(w+\epsilon)-L(w-\epsilon))/2\epsilon$ |

## 6. Pitfalls

- **Zero init** makes every hidden unit identical forever (symmetry). Use random init. **He init** suits ReLU, Xavier suits tanh.
- **Forgetting to divide by n** makes the gradients scale with the batch size.
- **Softmax overflow:** subtract the row max.
- **Wrong cache:** `backward` must use the values from the **same** forward pass.
- **Dead ReLUs:** a unit with $z_1 < 0$ for every input never gets a gradient again. A lower learning rate or Leaky ReLU helps.
- **Gradient check at a kink:** if some $z_1$ is exactly 0, the finite-difference estimate is wrong there. (The example skips such a point.)
- **Debug tip:** the first loss should be ≈ ln(c) (1.10 for 3 classes). Then try to overfit a tiny batch: the loss should approach 0.

## 7. Tutorial questions

1. Re-do the worked example with true class $y = 0$. What are $dz_2$ and $dW_2$?
2. In the worked example, if $z_1$ had been $(2, -1)$, what would $dz_1$ be?
3. For $d=2$, $h=64$, $c=3$, how many parameters does the network have?
4. Why is $db_2$ a sum over rows rather than a mean? (Hint: where did the $1/n$ go?)

<details>
<summary>Answers</summary>

1. $dz_2 = (0.5, 0.5) - (1, 0) = (-0.5, 0.5)$, and $dW_2 = \begin{pmatrix}-1 & 1\\ -0.5 & 0.5\end{pmatrix}$.
2. $(0.5, 0)$: the second unit was inactive, so the ReLU blocks its gradient.
3. $2\cdot64 + 64 + 64\cdot3 + 3 = 387$.
4. The $1/n$ is already inside $dz_2$ (`dz2 /= n`), so the sums give the gradient of the mean loss.

</details>

## 8. Interview questions

<details>
<summary>Explain backprop in two sentences.</summary>

The forward pass computes and caches every intermediate value; the backward pass applies the chain rule from the loss back to each parameter, reusing the upstream gradient at every layer. It costs about as much as one forward pass, so all the gradients come at once.
</details>

<details>
<summary>What do vanishing and exploding gradients mean, and what are the fixes?</summary>

Gradients get multiplied layer after layer and can shrink (sigmoid/tanh saturation) or grow. Fixes: ReLU, careful init (He/Xavier), normalisation (BatchNorm/LayerNorm), residual connections, gradient clipping.
</details>

<details>
<summary>Why does the softmax + cross-entropy gradient simplify to p − y?</summary>

The softmax Jacobian $p_k(\delta_{kj}-p_j)$ combined with $-1/p_y$ from the log collapses to $p_j - [j=y]$. It's the same algebra as sigmoid + BCE.
</details>

## 9. Lab

Read [explained.py](explained.py) and run `python example.py`. The **gradient check** must pass before you trust any training.
This is the hardest algorithm in the course, so give it the most days. Target: **20 minutes**.
