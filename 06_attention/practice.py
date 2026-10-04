# Attention / Multi-Head practice. Write one line of code under each numbered hint.
# Check your work with:  python example.py practice
# Stuck? Peek at explained.py, then note the step number in LOG.md.
#
# Shapes: batch B, sequence length T, d_model, heads h, d_head = d_model / h.

import numpy as np


def softmax(x, axis=-1):
    # 1. Subtract the max along axis (keepdims=True).
    # 2. Exponentiate.
    # 3. Return the result divided by its sum along axis (keepdims=True).
    raise NotImplementedError("write softmax")


def causal_mask(seq_len):
    # 1. Return a lower-triangular boolean (seq_len, seq_len) matrix (True = may attend).
    raise NotImplementedError("write causal_mask")


def scaled_dot_product_attention(Q, K, V, mask=None):
    # 1. d_k = the size of Q's last dimension.
    # 2. scores = Q times K-transposed (swap only the last two axes), divided by sqrt(d_k).
    # 3. If a mask is given:
    # 4.     Replace scores where the mask is False with -1e9 (np.where).
    # 5. weights = softmax of scores over the last axis.
    # 6. Return (weights @ V, weights).
    raise NotImplementedError("write scaled_dot_product_attention")


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=0):
        # 1. Assert that d_model divides evenly by n_heads.
        # 2. rng = a NumPy random Generator seeded with seed.
        # 3. Store n_heads.
        # 4. self.d_head = d_model // n_heads.
        # 5. std = 1 / sqrt(d_model).
        # 6. self.W_q = normal(0, std) of shape (d_model, d_model).
        # 7. self.W_k = the same.
        # 8. self.W_v = the same.
        # 9. self.W_o = the same.
        raise NotImplementedError("write __init__")

    def split_heads(self, x):
        # 1. Unpack batch and seq_len from x.shape.
        # 2. Reshape to (B, T, h, d_head), then transpose to (B, h, T, d_head).
        raise NotImplementedError("write split_heads")

    def merge_heads(self, x):
        # 1. Unpack batch and seq_len from x.shape (B, h, T, d_head).
        # 2. Transpose back to (B, T, h, d_head), then reshape to (B, T, h * d_head).
        raise NotImplementedError("write merge_heads")

    def forward(self, x, mask=None):
        # 1. Q = split_heads of the query projection of x.
        # 2. K = split_heads of the key projection of x.
        # 3. V = split_heads of the value projection of x.
        # 4. out, self.attn_weights = scaled dot-product attention of Q, K, V with the mask.
        # 5. Return merge_heads(out) projected by W_o.
        raise NotImplementedError("write forward")
