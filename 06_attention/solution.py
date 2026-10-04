import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        std = 1 / np.sqrt(d_model)
        self.W_q = rng.normal(0, std, size=(d_model, d_model))
        self.W_k = rng.normal(0, std, size=(d_model, d_model))
        self.W_v = rng.normal(0, std, size=(d_model, d_model))
        self.W_o = rng.normal(0, std, size=(d_model, d_model))

    def split_heads(self, x):
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        batch, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.n_heads * self.d_head)

    def forward(self, x, mask=None):
        Q = self.split_heads(x @ self.W_q)
        K = self.split_heads(x @ self.W_k)
        V = self.split_heads(x @ self.W_v)
        out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.merge_heads(out) @ self.W_o
