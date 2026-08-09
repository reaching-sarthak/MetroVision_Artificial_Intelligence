"""
fast_layers.py

Numerically-identical, MUCH faster replacements for the position-loop
convolutions in conv1.py/conv2.py/conv3.py and the loop-based
maxpool.py. Same math, same shapes, same gradients — just no
Python-level loop over every one of the ~500k output spatial positions.

WHY THIS WAS THE BOTTLENECK
----------------------------------------------------------------------
conv1_forward loops `for i in range(716): for j in range(716):` and
does one small numpy call per position — ~512,656 Python-level
iterations, each with call overhead that dwarfs the actual math. conv2
(126,736 positions), conv3 (125,316), and maxpool (up to ~4.1M
inner iterations, since it loops over every channel AND every 2x2
window AND every element inside the window) have the same problem.

THE FIX: im2col + one matmul
----------------------------------------------------------------------
Convolution is just: for every output position, dot the local patch
against every filter. Instead of doing that dot product one position
at a time in Python, extract every patch at once into one big matrix
(im2col, via numpy's sliding_window_view — no copy until the final
reshape) and do ONE matrix multiply for the entire image. NumPy's
matmul calls into BLAS, which is orders of magnitude faster per FLOP
than a Python for-loop.

d_input is computed with a different trick: instead of scattering each
output position's gradient back one at a time (K*K writes per
position, order 500k+ iterations), sum over the K*K kernel OFFSETS
instead (only 9 or 25 iterations for a 3x3/5x5 kernel), each one a
single big matmul. Same total math, ~50,000x fewer Python iterations.

maxpool is a plain reshape+max since stride == pool size (the 2x2
windows tile the input with no overlap) — no loop needed at all.

VALIDATED against the original conv1.py/conv2.py/conv3.py/maxpool.py
on small synthetic inputs — see validate_fast_layers.py. Forward
outputs match to float32 precision; backward matches numerically via
independent finite-difference gradient checks (not just "same as the
slow version", so a shared bug wouldn't slip through both).
"""

from __future__ import annotations

import numpy as np


# ============================================================
# GENERIC VECTORIZED CONVOLUTION (valid, stride 1)
# Works for conv1 (32,5,5,3), conv2 (64,3,3,32), conv3 (128,3,3,64) —
# the kernel size and channel counts come from filters.shape.
# ============================================================

def conv_forward(input_tensor: np.ndarray, filters: np.ndarray, biases: np.ndarray) -> tuple:
    """
    input_tensor : (H, W, C_in)
    filters      : (F, K, K, C_in)
    biases       : (F,)
    Returns output (H-K+1, W-K+1, F) and a cache for conv_backward.
    """
    h, w, c_in = input_tensor.shape
    f, k, _, c_in_f = filters.shape
    if c_in_f != c_in:
        raise ValueError(f"filters expect {c_in_f} input channels, got {c_in}")

    out_h, out_w = h - k + 1, w - k + 1

    x = input_tensor.astype(np.float32, copy=False)

    # im2col: every (K,K,C_in) patch, for every output position, at once.
    patches = np.lib.stride_tricks.sliding_window_view(x, (k, k, c_in))[:, :, 0, :, :, :]
    # patches shape: (out_h, out_w, K, K, C_in)
    flat_patches = patches.reshape(out_h * out_w, k * k * c_in)

    flat_filters = filters.reshape(f, k * k * c_in).astype(np.float32, copy=False)

    output_flat = flat_patches @ flat_filters.T + biases.astype(np.float32, copy=False)
    output = output_flat.reshape(out_h, out_w, f).astype(np.float32, copy=False)

    cache = {
        "input_shape": x.shape,
        "filters": filters,
        "flat_patches": flat_patches,  # (out_h*out_w, K*K*C_in) — reused in backward
        "output_shape": output.shape,
        "kernel_size": k,
    }
    return output, cache


def conv_backward(d_output: np.ndarray, cache: dict) -> tuple:
    in_h, in_w, c_in = cache["input_shape"]
    filters = cache["filters"]
    flat_patches = cache["flat_patches"]
    out_h, out_w, f = cache["output_shape"]
    k = cache["kernel_size"]

    d_output = d_output.astype(np.float32, copy=False)
    d_output_flat = d_output.reshape(out_h * out_w, f)

    # ---- d_biases: sum over every spatial position ----
    d_biases = d_output.sum(axis=(0, 1)).astype(np.float32)

    # ---- d_filters: one matmul over ALL positions at once ----
    d_filters_flat = d_output_flat.T @ flat_patches  # (F, K*K*C_in)
    d_filters = d_filters_flat.reshape(f, k, k, c_in).astype(np.float32)

    # ---- d_input: sum over K*K kernel OFFSETS (not over positions) ----
    # For offset (u, v): every output position (i,j) contributed
    # d_output[i,j,:] @ W[:,u,v,:] to d_input[i+u, j+v, :].
    # Looping over the (small) K*K offsets instead of the (huge)
    # out_h*out_w positions turns ~500k Python iterations into ~9-25.
    d_input = np.zeros((in_h, in_w, c_in), dtype=np.float32)
    for u in range(k):
        for v in range(k):
            contribution = d_output @ filters[:, u, v, :].astype(np.float32, copy=False)  # (out_h,out_w,C_in)
            d_input[u:u + out_h, v:v + out_w, :] += contribution

    return d_input, d_filters, d_biases


# ============================================================
# VECTORIZED MAXPOOL (2x2, stride 2 — matches maxpool.py exactly)
# ============================================================

def maxpool_forward(input_tensor: np.ndarray) -> tuple:
    h, w, c = input_tensor.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"maxpool_forward (fast) requires even H,W for exact 2x2 tiling, got ({h},{w})")

    x = input_tensor.astype(np.float32, copy=False)
    out_h, out_w = h // 2, w // 2

    # Non-overlapping 2x2 tiling (stride == pool size) is a pure reshape.
    reshaped = x.reshape(out_h, 2, out_w, 2, c)
    output = reshaped.max(axis=(1, 3)).astype(np.float32)

    cache = {
        "input_shape": x.shape,
        "output_shape": output.shape,
        "reshaped_input": reshaped,
        "output": output,
    }
    return output, cache


def maxpool_backward(d_output: np.ndarray, cache: dict) -> np.ndarray:
    in_h, in_w, c = cache["input_shape"]
    out_h, out_w, _ = cache["output_shape"]
    reshaped = cache["reshaped_input"]
    output = cache["output"]

    d_output = d_output.astype(np.float32, copy=False)

    # mask: which of the (up to 4) positions in each window equal the max.
    # NOTE: on an exact tie (extremely unlikely with real-valued features)
    # this splits the gradient evenly across every tied position, whereas
    # the reference maxpool.py always gives 100% of it to whichever tied
    # position it scanned first. Forward outputs are identical either way;
    # only tie-breaking in backward differs, which is immaterial in
    # practice with float32 activations.
    mask = (reshaped == output[:, None, :, None, :]).astype(np.float32)
    mask_count = mask.sum(axis=(1, 3), keepdims=True)
    d_reshaped = (d_output[:, None, :, None, :] * mask) / mask_count

    d_input = d_reshaped.reshape(in_h, in_w, c).astype(np.float32)
    return d_input
