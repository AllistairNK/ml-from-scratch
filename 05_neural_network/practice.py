# Neural Network + Backprop practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice   (it includes a gradient check)
# Stuck? Peek at explained.py, then note the step number in LOG.md.
#
# Shapes: X (n,d)  W1 (d,h)  z1, a1 (n,h)  W2 (h,c)  z2, probs (n,c)

import numpy as np


def relu(z):
    # 1. Return the elementwise max of z and 0.
    raise NotImplementedError("write relu")


def softmax(z):
    # 1. Subtract each row's max (keepdims=True) for numerical stability.
    # 2. Exponentiate.
    # 3. Return each row divided by its sum (keepdims=True).
    raise NotImplementedError("write softmax")


class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr=0.1, seed=0):
        # 1. rng = a NumPy random Generator seeded with seed.
        # 2. self.W1 = normal(0, sqrt(2 / n_in)) of shape (n_in, n_hidden)   (He init).
        # 3. self.b1 = zeros(n_hidden).
        # 4. self.W2 = normal(0, sqrt(2 / n_hidden)) of shape (n_hidden, n_out).
        # 5. self.b2 = zeros(n_out).
        # 6. Store lr.
        raise NotImplementedError("write __init__")

    def forward(self, X):
        # 1. Cache X on self.
        # 2. self.z1 = hidden pre-activation.
        # 3. self.a1 = relu of z1.
        # 4. self.z2 = output logits.
        # 5. self.probs = softmax of z2.
        # 6. Return self.probs.
        raise NotImplementedError("write forward")

    def loss(self, X, y):
        # 1. probs = forward pass.
        # 2. Return the mean cross-entropy: -mean(log(probability of the true class + 1e-12)).
        raise NotImplementedError("write loss")

    def backward(self, y):
        # 1. n = batch size.
        # 2. dz2 = a copy of the cached probs.
        # 3. Subtract 1 at each row's true class.
        # 4. Divide dz2 by n.
        # 5. dW2 = (a1 transposed) @ dz2.
        # 6. db2 = dz2 summed over the batch axis.
        # 7. da1 = dz2 @ (W2 transposed).
        # 8. dz1 = da1 masked by where z1 > 0.
        # 9. dW1 = (X transposed) @ dz1.
        # 10. db1 = dz1 summed over the batch axis.
        # 11. Return a dict {"W1": ..., "b1": ..., "W2": ..., "b2": ...}.
        raise NotImplementedError("write backward")

    def fit(self, X, y, epochs=1000):
        # 1. self.losses = [].
        # 2. Loop for epochs:
        # 3.     Append self.loss(X, y) to self.losses.
        # 4.     grads = self.backward(y).
        # 5.     Update W1.
        # 6.     Update b1.
        # 7.     Update W2.
        # 8.     Update b2.
        # 9. Return self.
        raise NotImplementedError("write fit")

    def predict(self, X):
        # 1. Return the argmax of the forward pass along axis 1.
        raise NotImplementedError("write predict")
