# Two-layer neural network (MLP) with manual backpropagation, explained line by line.
# Same code as solution.py; every line has a comment above it.
#
# Architecture:  X -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> softmax -> probabilities
# Loss:          cross-entropy, -mean(log p[correct class])
#
# Shapes (n = batch size, d = n_in, h = n_hidden, c = n_out):
#   X (n,d)  W1 (d,h)  z1, a1 (n,h)  W2 (h,c)  z2, probs (n,c)
#
# The backprop rule to remember: a gradient always has the same shape as the thing it is a gradient of.
# Use that to work out the transposes:  dW2 (h,c) = a1.T (h,n) @ dz2 (n,c).

import numpy as np


# ReLU activation: keep the positive values, zero out the negative ones.
def relu(z):
    # Elementwise max with 0.
    return np.maximum(0, z)


# Turn each row of scores (logits) into a probability distribution.
def softmax(z):
    # Subtract each row's max first. softmax(z) = softmax(z - c), and this stops exp() from overflowing.
    z = z - z.max(axis=1, keepdims=True)
    # Exponentiate: every value is now positive.
    exp_z = np.exp(z)
    # Normalise so each row sums to 1. keepdims=True keeps the shape (n,1) so it broadcasts.
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class MLP:
    # Layer sizes, learning rate, and a seed for reproducible weights.
    def __init__(self, n_in, n_hidden, n_out, lr=0.1, seed=0):
        # A local random generator.
        rng = np.random.default_rng(seed)
        # He initialisation (std = sqrt(2 / fan_in)) suits ReLU and keeps activations from vanishing or exploding.
        # Weights MUST be random: if they were all equal, every hidden unit would learn the same thing.
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_hidden))
        # Biases can start at zero (the random weights already break the symmetry).
        self.b1 = np.zeros(n_hidden)
        # The second layer gets the same He init, with fan_in = n_hidden.
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), size=(n_hidden, n_out))
        # One output bias per class.
        self.b2 = np.zeros(n_out)
        # Gradient-descent step size.
        self.lr = lr

    # Forward pass. Cache the intermediate values, because backward() needs them.
    def forward(self, X):
        # Cache the input: needed for dW1.
        self.X = X
        # Hidden pre-activation. (n,d) @ (d,h) + (h,) -> (n,h).
        self.z1 = X @ self.W1 + self.b1
        # Hidden activation. Cache it: needed for dW2.
        self.a1 = relu(self.z1)
        # Output logits. (n,h) @ (h,c) + (c,) -> (n,c).
        self.z2 = self.a1 @ self.W2 + self.b2
        # Class probabilities. Cache them: needed for dz2.
        self.probs = softmax(self.z2)
        # Return the probabilities to the caller.
        return self.probs

    # Mean cross-entropy loss. y holds integer class labels 0..c-1.
    def loss(self, X, y):
        # Run the forward pass (this also refreshes the caches for backward).
        probs = self.forward(X)
        # probs[np.arange(n), y] picks each row's probability for its true class.
        # Add 1e-12 so log never sees exactly 0.
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))

    # Backward pass: return the gradient of the loss w.r.t. each parameter.
    # Call it right after forward()/loss() on the same batch.
    def backward(self, y):
        # Batch size (the loss is a mean, so every gradient carries a 1/n).
        n = len(y)
        # Softmax + cross-entropy gradient w.r.t. the logits: probs - one_hot(y).
        # Start from a copy of probs (do not modify the cache) ...
        dz2 = self.probs.copy()
        # ... and subtract 1 at each row's true class.
        dz2[np.arange(n), y] -= 1
        # Divide by n because the loss averages over the batch.
        dz2 /= n
        # z2 = a1 @ W2 + b2  =>  dW2 = a1.T @ dz2.  (h,n) @ (n,c) -> (h,c), same shape as W2.
        dW2 = self.a1.T @ dz2
        # The bias is added to every row, so its gradient is the sum over the batch. Shape (c,).
        db2 = dz2.sum(axis=0)
        # Push the gradient back through W2: da1 = dz2 @ W2.T.  (n,c) @ (c,h) -> (n,h).
        da1 = dz2 @ self.W2.T
        # ReLU backward: the gradient flows only where z1 > 0 (the local derivative is 1 there, 0 elsewhere).
        dz1 = da1 * (self.z1 > 0)
        # Same pattern as dW2, one layer down: (d,n) @ (n,h) -> (d,h).
        dW1 = self.X.T @ dz1
        # Sum over the batch for the hidden bias. Shape (h,).
        db1 = dz1.sum(axis=0)
        # Return a dict keyed by parameter name, so the update step (and a gradient check) is easy.
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # Full-batch gradient descent.
    def fit(self, X, y, epochs=1000):
        # Loss history.
        self.losses = []
        # One step per epoch.
        for _ in range(epochs):
            # Forward pass + loss (this fills the caches).
            self.losses.append(self.loss(X, y))
            # Backward pass: the gradients.
            grads = self.backward(y)
            # Update W1.
            self.W1 -= self.lr * grads["W1"]
            # Update b1.
            self.b1 -= self.lr * grads["b1"]
            # Update W2.
            self.W2 -= self.lr * grads["W2"]
            # Update b2.
            self.b2 -= self.lr * grads["b2"]
        # Allow chaining.
        return self

    # Predict the most likely class for each row.
    def predict(self, X):
        # Argmax of the probabilities (the same as the argmax of the logits).
        return np.argmax(self.forward(X), axis=1)
