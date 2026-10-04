# F4: Probability for ML

**New words?** [probability](../00_basics/B1_glossary.md#probability) · [class](../00_basics/B1_glossary.md#class) · [mean](../00_basics/B1_glossary.md#mean) · [variance](../00_basics/B1_glossary.md#variance) · [threshold](../00_basics/B1_glossary.md#threshold) · or the full [glossary](../00_basics/B1_glossary.md).

**Learning objectives.** After this lecture you can:

- use conditional probability and Bayes' rule
- read the Gaussian (normal) density and say what the mean and variance control
- explain why ML code works with **log** probabilities
- explain what sigmoid and softmax turn numbers into

---

## 1. Basics

- $P(A)$ is a number from 0 to 1, and the probabilities of all the possible outcomes sum to 1.
- **Conditional probability** $P(A \mid B)$ is the probability of A given that B is known to be true.
- **Independence:** A and B are independent if $P(A \text{ and } B) = P(A)\,P(B)$.

## 2. Bayes' rule

$$P(\text{class} \mid x) = \frac{P(x \mid \text{class}) \; P(\text{class})}{P(x)}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(\text{class})$ | **prior** | how common the class is before seeing $x$ |
| $P(x \mid \text{class})$ | **likelihood** | how typical $x$ is for that class |
| $P(\text{class} \mid x)$ | **posterior** | what we actually want |
| $P(x)$ | evidence | the same for every class, so it can be ignored when picking the best class |

**Example.** 1% of emails are spam. The word "prize" appears in 60% of spam and 2% of normal email.
Given that an email contains "prize":

- spam: $0.60 \times 0.01 = 0.006$
- not spam: $0.02 \times 0.99 = 0.0198$
- normalise: $P(\text{spam} \mid \text{prize}) = 0.006 / (0.006 + 0.0198) \approx 0.23$

Even with a strong clue, the low prior keeps the posterior at 23%. Naive Bayes is this
calculation, done for every feature.

## 3. The Gaussian (normal) distribution

$$\mathcal{N}(x;\ \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $\mu$ (**mean**) is the centre of the bell
- $\sigma^2$ (**variance**) is how wide it is. $\sigma$ is the standard deviation.
- About 68% of the values lie within $\pm 1\sigma$ of the mean, and 95% within $\pm 2\sigma$.

Estimating them from data: `mu = x.mean()`, `var = x.var()`.

## 4. Why logs?

Multiplying many probabilities (each < 1) quickly **underflows** to exactly `0.0` in floating
point. With 500 features at about 0.1 each, the product is $10^{-500}$, which is below the smallest
float. Logs turn products into sums, which stay well-behaved:

$$\log(a \cdot b) = \log a + \log b$$

The log is increasing, so **the class with the largest log-probability also has the largest probability**.

The log of the Gaussian is a nice expression with no `exp`:

$$\log \mathcal{N}(x;\mu,\sigma^2) = -\tfrac12\left[\log(2\pi\sigma^2) + \frac{(x-\mu)^2}{\sigma^2}\right]$$

## 5. Turning scores into probabilities

Models output raw scores ("logits") in $(-\infty, \infty)$. We convert them into probabilities:

**Sigmoid** (2 classes): squashes one score into $(0,1)$.

$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \sigma(0)=0.5,\ \sigma(\text{large}) \to 1,\ \sigma(\text{very negative}) \to 0$$

**Softmax** (K classes): turns a vector of scores into a distribution that sums to 1.

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

Example: $z = (2, 1, 0)$ gives $(0.665, 0.245, 0.090)$.

**Stability trick:** $e^{1000}$ overflows. Subtracting the max from every score first gives the
same answer (the constant cancels top and bottom) and keeps all the exponents ≤ 0.

## 6. Cross-entropy loss

If the model gives the true class probability $p$, the loss is $-\log p$:

| $p$ (true class) | loss $-\log p$ |
|------------------|----------------|
| 0.99 | 0.01 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.001 | 6.91 |

Confident and wrong is punished heavily. Averaged over samples, this is the **cross-entropy**
(also called log-loss or negative log-likelihood) used by logistic regression and neural networks.
A useful sanity check: a model that guesses uniformly over K classes has loss $\ln K$
(0.693 for 2 classes, 1.099 for 3).

## Check yourself

1. A disease affects 1 in 1000 people. A test catches 99% of the sick and has a 5% false-positive rate. You test positive. Roughly what is $P(\text{sick})$?
2. Why can Naive Bayes ignore $P(x)$ when predicting?
3. What is softmax of $(5, 5)$? And of $(1005, 1005)$?
4. Your 3-class network's loss at the very first step is 1.10. Is that suspicious?

<details>
<summary>Answers</summary>

1. $0.99 \times 0.001 = 0.00099$ vs $0.05 \times 0.999 \approx 0.04995$, so about $0.00099 / 0.051 \approx 2\%$.
2. It's the same for every class, so it doesn't change which class is the largest.
3. $(0.5, 0.5)$ both times. Softmax only cares about the differences between scores, which is why subtracting the max is safe.
4. No. $\ln 3 \approx 1.099$ is exactly what an untrained, uniform-guessing model should give.

</details>

**Next:** [F5: ML basics](F5_ml_basics.md)
