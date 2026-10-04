import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def loss(self, X, y):
        probs = self.forward(X)
        return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))

    def backward(self, y):
        n = len(y)
        dz2 = self.probs.copy()
        dz2[np.arange(n), y] -= 1
        dz2 /= n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def fit(self, X, y, epochs=1000):
        self.losses = []
        for _ in range(epochs):
            self.losses.append(self.loss(X, y))
            grads = self.backward(y)
            self.W1 -= self.lr * grads["W1"]
            self.b1 -= self.lr * grads["b1"]
            self.W2 -= self.lr * grads["W2"]
            self.b2 -= self.lr * grads["b2"]
        return self

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
