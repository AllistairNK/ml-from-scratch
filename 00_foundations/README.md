# Unit 0: Foundations

Every algorithm in this course is built from the same small set of ideas. If an algorithm lecture
feels confusing, the gap is almost always in one of these five lectures. Do them first, in order.

| Lecture | Topic | You need it for |
|---------|-------|-----------------|
| [F1](F1_numpy.md) | NumPy: arrays, shapes, axes, broadcasting, indexing | Every algorithm |
| [F2](F2_linear_algebra.md) | Vectors, dot products, matrix multiplication, distances, eigenvectors | Every algorithm (eigenvectors only for PCA) |
| [F3](F3_calculus_and_gradient_descent.md) | Derivatives, gradients, chain rule, gradient descent | Linear/logistic regression, neural networks |
| [F4](F4_probability.md) | Probability, Bayes' rule, Gaussians, logs | Logistic regression, Naive Bayes, softmax |
| [F5](F5_ml_basics.md) | The ML vocabulary: features, labels, train/test, loss, overfitting, metrics | Every algorithm |

**Lab:** [practice.py](practice.py) has 12 one-line NumPy drills that cover exactly the tricks the
algorithm code uses. Run `python 00_foundations/example.py practice` to check them (or use the course app), and see
[explained.py](explained.py) if you get stuck.

How to use each lecture:

1. Read it with a notebook (paper or a Python shell) open. Type every code snippet yourself.
2. Do the **Check yourself** questions before opening the answers.
3. If a question stumps you, re-read that section. Don't move on with a gap.
