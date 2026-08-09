"""
Code_Block: MaxPool - 2x2 max pooling (stride 2, valid), from scratch (NumPy only).

No TensorFlow / PyTorch / OpenCV / SciPy / deep-learning-framework pooling
helpers are used. Every 2x2 window scan and every max reduction is performed
manually with explicit NumPy indexing and loops.

This file exposes exactly two functions meant to be called from trainflow.py:

    maxpool_forward(input_tensor)      -> output, cache
    maxpool_backward(d_output, cache)  -> d_input

MaxPool has NO trainable parameters - nothing for trainflow.py to load from a
checkpoint or hand to the optimizer for this layer. It sits between silu1 and
conv2 in the real pipeline:

    silu1_output (716,716,32) -> maxpool_forward -> conv2   (output: 358,358,32)

------------------------------------------------------------------------------
MATH: MAX POOLING (pool F x F, stride S, valid / no padding)
------------------------------------------------------------------------------
For each channel c, at each output position (p, q):

    output[p, q, c] = max over (u, v) in {0..F-1}^2 of  input[p*S + u, q*S + v, c]

For this layer (F = 2, S = 2), stride equals pool size, so the 2x2 windows
tile the input with NO overlap: every input element belongs to exactly one
window, and each window contributes exactly one number to the output.

------------------------------------------------------------------------------
WHY NO TRAINABLE PARAMETERS
------------------------------------------------------------------------------
"Take the maximum of each 2x2 window" is fully specified by the architecture -
no weight to multiply by, no bias to add. Backward only has to ROUTE
gradients (to the argmax locations); it never updates anything.

------------------------------------------------------------------------------
WHY EACH FEATURE MAP IS POOLED INDEPENDENTLY (channels unchanged)
------------------------------------------------------------------------------
Pooling is a purely SPATIAL operation - channel c's output depends ONLY on
channel c's input. Channels are neither created, destroyed, nor combined:

    (716, 716, 32) -> (358, 358, 32)      # C in == C out

Output size, computed dynamically from whatever input is passed in:

    H_out = floor((H_in - F) / S) + 1

------------------------------------------------------------------------------
MATH: BACKWARD PASS (gradient routing / max mask)
------------------------------------------------------------------------------
Within a 2x2 window the output depends ONLY on the single largest element:

    ∂output/∂input[window element] = 1  at the argmax position
                                    = 0  at the other three positions

So ∂L/∂input is built by taking each upstream gradient d_output[p,q,c] and
depositing it, unscaled, at the input location of that window's maximum,
while the window's other three input positions receive 0. This is why
forward() records WHERE each max came from (max_indices) - backward() just
scatters gradients straight there without recomputing any argmax.
"""

import numpy as np

# ==============================
# CONFIG (fixed architecture choices - NOT input-size dependent)
# ==============================
POOL_SIZE = 2   # square pooling window: 2 x 2
STRIDE = 2      # pooling stride (non-overlapping windows: stride == pool size)


# ==============================
# VALIDATION
# ==============================
def _validate_forward_inputs(input_tensor: np.ndarray) -> None:
    if not isinstance(input_tensor, np.ndarray):
        raise TypeError(f"input_tensor must be a NumPy array, got {type(input_tensor)}")
    if input_tensor.ndim != 3:
        raise ValueError(f"input_tensor must be 3-D (H, W, C), got shape {input_tensor.shape}")
    h, w, _ = input_tensor.shape
    if h < POOL_SIZE or w < POOL_SIZE:
        raise ValueError(f"input spatial size ({h},{w}) smaller than pool size {POOL_SIZE}")


def _validate_backward_inputs(d_output: np.ndarray, cache: dict) -> None:
    for key in ("input_shape", "max_indices", "output_shape"):
        if key not in cache:
            raise KeyError(f"cache must contain '{key}' (as returned by maxpool_forward)")
    if d_output.shape != cache["output_shape"]:
        raise ValueError(f"d_output shape {d_output.shape} != forward output shape {cache['output_shape']}")


# ==============================
# FORWARD PASS
# ==============================
def maxpool_forward(input_tensor: np.ndarray, report_progress: bool = False) -> tuple:
    """
    Run 2x2 / stride-2 max pooling over every feature map with explicit loops.

    For each channel c and each non-overlapping 2x2 window:
        1. extract the window        input[2p:2p+2, 2q:2q+2, c]
        2. find its largest value    -> output[p, q, c]
        3. record WHICH (row, col) inside the window supplied that max
                                       -> max_indices[p, q, c]

    Args:
        input_tensor: (H, W, C) float array. H, W >= 2. (Real pipeline: 716x716x32.)
        report_progress: print a % line every ~10% of channels.

    Returns:
        output: (H//2, W//2, C) float32 volume.
        cache: dict with everything maxpool_backward() needs:
            cache["input_shape"]  : the (H, W, C) shape of the forward input
            cache["max_indices"]  : (H//2, W//2, C, 2) int64 - (row, col) of
                                     the max in every window, i.e. WHERE the
                                     gradient goes in backward()
            cache["output_shape"] : the output's shape, for backward's own validation
    """
    _validate_forward_inputs(input_tensor)

    input_tensor = input_tensor.astype(np.float32, copy=False)
    in_h, in_w, in_c = input_tensor.shape
    out_h = (in_h - POOL_SIZE) // STRIDE + 1
    out_w = (in_w - POOL_SIZE) // STRIDE + 1

    output = np.zeros((out_h, out_w, in_c), dtype=np.float32)
    # [...,0] -> input row of the window max, [...,1] -> input col of the window max
    max_indices = np.zeros((out_h, out_w, in_c, 2), dtype=np.int64)

    progress_step = max(1, in_c // 10)
    for c in range(in_c):                    # pool each feature map independently
        for p in range(out_h):
            row_start = p * STRIDE
            for q in range(out_w):
                col_start = q * STRIDE

                window = input_tensor[row_start:row_start + POOL_SIZE,
                                      col_start:col_start + POOL_SIZE, c]

                # scan the 4 elements; keep the largest value AND its (row, col)
                # so backward() can route the gradient to it.
                best_value = window[0, 0]
                best_row, best_col = 0, 0
                for u in range(POOL_SIZE):
                    for v in range(POOL_SIZE):
                        if window[u, v] > best_value:
                            best_value = window[u, v]
                            best_row, best_col = u, v

                output[p, q, c] = best_value
                max_indices[p, q, c, 0] = row_start + best_row
                max_indices[p, q, c, 1] = col_start + best_col

        if report_progress and (c % progress_step == 0 or c == in_c - 1):
            print(f"  maxpool forward: channel {c + 1}/{in_c} ({(c + 1) / in_c * 100:.0f}%)")

    cache = {
        "input_shape": input_tensor.shape,
        "max_indices": max_indices,
        "output_shape": output.shape,
    }
    return output, cache


# ==============================
# BACKWARD PASS
# ==============================
def maxpool_backward(d_output: np.ndarray, cache: dict) -> np.ndarray:
    """
    Manual backward pass for MaxPool: scatter each upstream gradient to the
    single input position that produced its window's max; everywhere else
    in the window gets 0.

    Args:
        d_output: (H//2, W//2, C) float array - ∂L/∂output, handed down from
            the next layer's backward() (conv2, in the real pipeline).
        cache: the dict returned by maxpool_forward() for this same input.

    Returns:
        d_input: (H, W, C) float32 - ∂L/∂input, to hand to the PREVIOUS
            layer's backward() (silu1, in the real pipeline).
    """
    _validate_backward_inputs(d_output, cache)

    in_h, in_w, in_c = cache["input_shape"]
    max_indices = cache["max_indices"]
    out_h, out_w, _ = cache["output_shape"]
    d_output = d_output.astype(np.float32, copy=False)

    d_input = np.zeros((in_h, in_w, in_c), dtype=np.float32)

    for c in range(in_c):
        for p in range(out_h):
            for q in range(out_w):
                row, col = max_indices[p, q, c]
                # deposit the upstream gradient, UNSCALED, at the argmax
                # location only - the other 3 positions in the window get 0
                # (they never influenced the forward output at all).
                d_input[row, col, c] += d_output[p, q, c]

    return d_input


# ==============================
# SELF-TEST: shape check + numerical gradient check
# ==============================
if __name__ == "__main__":
    print("=" * 55)
    print("CODE_BLOCK: MAXPOOL - FORWARD/BACKWARD SELF-TEST")
    print("=" * 55)

    rng = np.random.default_rng(0)

    # ---- 1. Shape check on a real-size-shaped (but synthetic) input ----
    real_input = rng.standard_normal((716, 716, 32)).astype(np.float32)
    output, cache = maxpool_forward(real_input, report_progress=False)
    assert output.shape == (358, 358, 32), f"unexpected output shape {output.shape}"
    print(f"shape check on 716x716x32 input -> output {output.shape}  OK")

    # ---- 2. Manual spot-check: window (0:2, 0:2) of channel 0 ----
    window_0 = real_input[0:2, 0:2, 0]
    manual_max = np.max(window_0)
    assert np.isclose(manual_max, output[0, 0, 0]), "spot-check max mismatch"
    r0, c0 = cache["max_indices"][0, 0, 0]
    assert real_input[r0, c0, 0] == manual_max, "stored argmax must locate the true max"
    print(f"spot-check window(0:2,0:2,ch0): max={manual_max:.4f}, "
          f"stored index=({r0},{c0})  OK")

    # ---- 3. Numerical gradient check on a small synthetic input ----
    print("\nRunning numerical gradient check on a small (6, 6, 3) input...")
    small_input = rng.standard_normal((6, 6, 3)).astype(np.float32)
    out, cache = maxpool_forward(small_input)
    assert out.shape == (3, 3, 3), f"unexpected small output shape {out.shape}"

    # Arbitrary fixed upstream gradient defines a valid scalar loss
    # L = sum(output * d_out_fixed), whose true gradient wrt any input
    # element is exactly what maxpool_backward should compute.
    d_out_fixed = rng.standard_normal(out.shape).astype(np.float32)
    d_input = maxpool_backward(d_out_fixed, cache)
    assert d_input.shape == small_input.shape, "d_input shape must match forward input shape"

    def scalar_loss(inp):
        o, _ = maxpool_forward(inp)
        return np.sum(o * d_out_fixed)

    eps = 1e-3
    n_samples = 10
    print("\n--- d_input gradient check (10 random elements) ---")
    for _ in range(n_samples):
        idx = tuple(rng.integers(0, s) for s in small_input.shape)
        base = small_input.copy()
        orig = base[idx]

        base[idx] = orig + eps
        loss_plus = scalar_loss(base)
        base[idx] = orig - eps
        loss_minus = scalar_loss(base)
        base[idx] = orig

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = d_input[idx]
        # Note: elements that are NEVER the max of their window correctly
        # get analytic gradient 0 - numeric should also land near 0 for those,
        # since nudging a non-max element doesn't change any window's max
        # (unless the nudge is large enough to overtake the current max,
        # which eps=1e-3 is far too small to do here).
        rel_err = abs(numeric - analytic) / max(1e-6, abs(numeric) + abs(analytic))

        print(f"  idx={idx}  numeric={numeric:+.6f}  analytic={analytic:+.6f}  rel_err={rel_err:.2e}")
        assert rel_err < 1e-2, f"d_input mismatch at {idx}: rel_err={rel_err:.2e}"

    print("\nAll gradient checks passed: maxpool_backward matches numerical differentiation.")
    print("MaxPool forward/backward self-test PASSED.")