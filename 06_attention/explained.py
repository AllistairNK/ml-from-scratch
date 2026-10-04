# Scaled dot-product attention and multi-head self-attention (forward pass), explained line by line.
# Same code as solution.py; every line has a comment above it.
#
#   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
#
# Each query asks "which keys match me?". The softmax turns the match scores into weights that sum
# to 1, and the output is the weighted average of the values.
# Multi-head attention runs h smaller attentions in parallel on different learned projections,
# then concatenates the results and mixes them with W_o.
#
# Shapes: batch B, sequence length T, model width d_model, heads h, d_head = d_model / h.

import numpy as np


# A numerically stable softmax along any axis (attention uses the last axis = the keys).
def softmax(x, axis=-1):
    # Subtract the max so exp() cannot overflow (the result is unchanged).
    x = x - x.max(axis=axis, keepdims=True)
    # Exponentiate.
    exp_x = np.exp(x)
    # Normalise along the axis so the weights sum to 1.
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# Mask for decoder (GPT-style) attention: position i may only look at positions <= i.
def causal_mask(seq_len):
    # Lower-triangular boolean matrix. True = allowed to attend. Shape (T, T).
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


# The core operation. Works for any leading batch/head dims: Q (..., T_q, d_k), K (..., T_k, d_k), V (..., T_k, d_v).
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Key/query dimension, used for the scaling.
    d_k = Q.shape[-1]
    # Similarity of every query with every key -> (..., T_q, T_k).
    # swapaxes(-2, -1) transposes only the last two dims (K.T would also reverse the batch dims).
    # Dividing by sqrt(d_k) keeps the score variance near 1. Without it, the softmax saturates and the gradients vanish.
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    # Optional mask (True = keep). It broadcasts, e.g. (T, T) over (B, h, T, T).
    if mask is not None:
        # Blocked positions get a huge negative score, so exp() makes their weight about 0.
        scores = np.where(mask, scores, -1e9)
    # Softmax over the keys: each query's weights sum to 1.
    weights = softmax(scores, axis=-1)
    # Weighted sum of the values -> (..., T_q, d_v). Also return the weights for inspection.
    return weights @ V, weights


class MultiHeadAttention:
    # d_model must split evenly across the heads.
    def __init__(self, d_model, n_heads, seed=0):
        # Each head gets d_model / n_heads dims, so this must divide exactly.
        assert d_model % n_heads == 0
        # A local random generator for the weight init.
        rng = np.random.default_rng(seed)
        # Number of parallel heads.
        self.n_heads = n_heads
        # Per-head width. Total compute is about the same as one big head.
        self.d_head = d_model // n_heads
        # Init std of 1/sqrt(d_model) keeps the projections at roughly unit scale.
        std = 1 / np.sqrt(d_model)
        # Query projection. One (d_model, d_model) matrix computes all the heads at once (sliced later).
        self.W_q = rng.normal(0, std, size=(d_model, d_model))
        # Key projection.
        self.W_k = rng.normal(0, std, size=(d_model, d_model))
        # Value projection.
        self.W_v = rng.normal(0, std, size=(d_model, d_model))
        # Output projection: mixes information across the heads after they are concatenated.
        self.W_o = rng.normal(0, std, size=(d_model, d_model))

    # (B, T, d_model) -> (B, h, T, d_head): give each head its own slice of the features.
    def split_heads(self, x):
        # Read the shape (the last dim is d_model).
        batch, seq_len, _ = x.shape
        # Reshape the feature dim into (h, d_head), then move the head axis before time,
        # so the matmuls in attention run independently per (batch, head).
        return x.reshape(batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    # (B, h, T, d_head) -> (B, T, d_model): the exact inverse of split_heads (concatenate the heads).
    def merge_heads(self, x):
        # Read the shape.
        batch, _, seq_len, _ = x.shape
        # Move the head axis back next to d_head, then flatten (h, d_head) into d_model.
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.n_heads * self.d_head)

    # Self-attention: queries, keys and values all come from the same x of shape (B, T, d_model).
    def forward(self, x, mask=None):
        # Project, then split into heads -> (B, h, T, d_head).
        Q = self.split_heads(x @ self.W_q)
        # Same for the keys.
        K = self.split_heads(x @ self.W_k)
        # Same for the values.
        V = self.split_heads(x @ self.W_v)
        # Attention runs for every (batch, head) in one go. Keep the weights (B, h, T, T) for inspection.
        out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # Concatenate the heads and apply the output projection -> (B, T, d_model).
        return self.merge_heads(out) @ self.W_o
