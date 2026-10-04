"""Attention usage example and sanity checks.

    python example.py            # check solution.py
    python example.py practice   # check your practice.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loader import load_impl  # noqa: E402

impl = load_impl(__file__)
attention, causal_mask, MultiHeadAttention = (
    impl.scaled_dot_product_attention, impl.causal_mask, impl.MultiHeadAttention)

np.set_printoptions(precision=3, suppress=True)

# 1) Attention as a soft dictionary lookup.
#    Three keys pointing in different directions, each with a recognisable value.
K = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
V = np.array([[1.0, 0], [0, 1.0], [5.0, 5.0]])
Q = np.array([[0, 10.0, 0]])  # matches key 1
out, w = attention(Q, K, V)
print("lookup weights:", w, " output:", out, " (expected about value 1 = [0, 1])")

# 2) Causal mask: position i must not look at positions after i.
print("causal mask for T=4:\n", causal_mask(4).astype(int))

# 3) Multi-head self-attention on a batch of sequences.
B, T, d_model, n_heads = 2, 5, 16, 4
rng = np.random.default_rng(0)
x = rng.normal(size=(B, T, d_model))
mha = MultiHeadAttention(d_model, n_heads, seed=0)
y = mha.forward(x, mask=causal_mask(T))
print("input", x.shape, "-> output", y.shape, "| attention weights", mha.attn_weights.shape)
print("head 0 weights for sequence 0 (upper triangle should be 0):\n", mha.attn_weights[0, 0])

# 4) Causality check: changing the LAST token must not change any earlier output.
x2 = x.copy()
x2[:, -1] = rng.normal(size=(B, d_model))
y2 = mha.forward(x2, mask=causal_mask(T))
print("earlier outputs unchanged after editing the last token:", np.allclose(y[:, :-1], y2[:, :-1]))

# 5) Multi-head = per-head attention + concat. Recompute head 0 by hand and compare.
#    Head i uses columns [i*d_head, (i+1)*d_head) of each projection matrix.
d_head = d_model // n_heads
manual = np.concatenate(
    [attention(x @ mha.W_q[:, i * d_head:(i + 1) * d_head],
               x @ mha.W_k[:, i * d_head:(i + 1) * d_head],
               x @ mha.W_v[:, i * d_head:(i + 1) * d_head], causal_mask(T))[0]
     for i in range(n_heads)], axis=-1) @ mha.W_o
print("matches manual per-head computation:", np.allclose(manual, y))

assert np.allclose(out, [[0, 1]], atol=1e-3)
assert y.shape == (B, T, d_model)
assert mha.attn_weights.shape == (B, n_heads, T, T)
assert np.allclose(mha.attn_weights.sum(axis=-1), 1.0)
assert np.allclose(np.triu(mha.attn_weights, k=1), 0.0)
assert np.allclose(y[:, :-1], y2[:, :-1])
assert np.allclose(manual, y)
print("All checks passed.")
