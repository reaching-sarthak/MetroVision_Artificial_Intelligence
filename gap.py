"""
Code_Block: GlobalAveragePooling (GAP) - spatial mean per channel, from scratch.

NumPy only. No TensorFlow / PyTorch / deep-learning-framework pooling helpers.

This file exposes exactly two functions meant to be called from trainflow.py:

    gap_forward(input_tensor)      -> output, cache
    gap_backward(d_output, cache)  -> d_input

GAP has NO trainable parameters - nothing for trainflow.py to load from a
checkpoint or hand to the optimizer for this layer. It sits between conv3 and
dense1 in the real pipeline:

    conv3_output (128,354,354) -> gap_forward -> dense1   (output: 128,)

Channel-first (C, H, W) input, matching the original block_gap.py convention -
conv3's raw output is channel-last (H, W, C), so trainflow.py transposes it
before calling gap_forward (see the __main__ example below).

------------------------------------------------------------------------------
WHY GLOBAL AVERAGE POOLING
------------------------------------------------------------------------------
GAP reduces the SPATIAL dimensions (354x354 -> 1x1) while preserving the
CHANNEL-level feature information (128 -> 128). Each of the 128 Conv3 filters
learned to detect one pattern across the whole image; GAP reports, per filter,
the average strength of that pattern over the full spatial extent - a compact
128-dim descriptor a classifier head can consume directly, without the huge
parameter count a flatten + dense layer would need (354*354*128 ≈ 16 million
inputs).

------------------------------------------------------------------------------
MATH: GLOBAL AVERAGE POOLING
------------------------------------------------------------------------------
For each channel c (feature map f_c of shape H x W):

    output[c] = (1 / (H * W)) * Σ_i Σ_j  f_c[i, j]

There is exactly ONE output value per channel; channels are never mixed -
output[c] depends ONLY on feature map c. Computed dynamically from whatever
input_tensor is passed in, rather than hardcoded to 354x354.

------------------------------------------------------------------------------
WHY NO TRAINABLE PARAMETERS
------------------------------------------------------------------------------
GAP is a FIXED, parameter-free reduction (like max pooling, but averaging
instead of taking a maximum) - "mean over the spatial window" is fully
specified by the architecture. Weights: 0. Biases: 0.

------------------------------------------------------------------------------
MATH: BACKWARD PASS
------------------------------------------------------------------------------
Each output[c] is the mean of H*W input elements, so EVERY element in feature
map c contributed equally (weight 1/(H*W)) to output[c]:

    ∂output[c]/∂input[c,i,j] = 1 / (H*W)     for every (i, j)

By the chain rule, given the upstream gradient d_output = ∂L/∂output (shape
(C,)):

    d_input[c,i,j] = d_output[c] / (H*W)     -- same value at EVERY spatial
                                                 position of channel c

Unlike max pooling (which routes the whole gradient to ONE winning position),
average pooling SPREADS the gradient evenly across every position - this is
why no argmax/index bookkeeping is needed here, only H and W.
"""

import numpy as np


# ==============================
# VALIDATION
# ==============================
def _validate_forward_inputs(input_tensor: np.ndarray) -> None:
    if not isinstance(input_tensor, np.ndarray):
        raise TypeError(f"input_tensor must be a NumPy array, got {type(input_tensor)}")
    if input_tensor.ndim != 3:
        raise ValueError(f"input_tensor must be 3-D (C, H, W), got shape {input_tensor.shape}")


def _validate_backward_inputs(d_output: np.ndarray, cache: dict) -> None:
    for key in ("input_shape", "num_pixels"):
        if key not in cache:
            raise KeyError(f"cache must contain '{key}' (as returned by gap_forward)")
    num_channels = cache["input_shape"][0]
    if d_output.shape != (num_channels,):
        raise ValueError(f"d_output shape {d_output.shape} != ({num_channels},)")


# ==============================
# FORWARD PASS
# ==============================
def gap_forward(input_tensor: np.ndarray, report_progress: bool = False) -> tuple:
    """
    Compute the Global Average Pooling forward pass, channel by channel.

    For each channel c:
        1. take its full (H, W) feature map  ->  input_tensor[c]
        2. sum all H*W pixel values in that map
        3. divide by H*W                       ->  average
        4. store the single scalar at output[c]

    Args:
        input_tensor: (C, H, W) float array, channel-first. (Real pipeline: 128x354x354.)
        report_progress: print a % line every ~10% of channels.

    Returns:
        output: (C,) float32 vector; output[c] = mean of feature map c.
        cache: dict with everything gap_backward() needs:
            cache["input_shape"] : the (C, H, W) shape of the forward input
            cache["num_pixels"]  : H * W, the divisor used by both directions
    """
    _validate_forward_inputs(input_tensor)

    input_tensor = input_tensor.astype(np.float32, copy=False)
    in_c, in_h, in_w = input_tensor.shape
    num_pixels = in_h * in_w

    output = np.zeros(in_c, dtype=np.float32)
    progress_step = max(1, in_c // 10)
    for c in range(in_c):
        feature_map_c = input_tensor[c]                    # (H, W) map for channel c only
        output[c] = np.float32(np.sum(feature_map_c) / num_pixels)
        if report_progress and (c % progress_step == 0 or c == in_c - 1):
            print(f"  gap forward: channel {c + 1}/{in_c} ({(c + 1) / in_c * 100:.0f}%)")

    cache = {
        "input_shape": input_tensor.shape,
        "num_pixels": num_pixels,
    }
    return output, cache


# ==============================
# BACKWARD PASS
# ==============================
def gap_backward(d_output: np.ndarray, cache: dict) -> np.ndarray:
    """
    Manual backward pass for GAP: spread each channel's upstream gradient
    evenly across every spatial position of that channel.

        d_input[c, i, j] = d_output[c] / (H * W)     for every (i, j)

    Args:
        d_output: (C,) float array - ∂L/∂output, handed down from the next
            layer's backward() (dense1, in the real pipeline).
        cache: the dict returned by gap_forward() for this same input.

    Returns:
        d_input: (C, H, W) float32 - ∂L/∂input, to hand to the PREVIOUS
            layer's backward() (conv3, in the real pipeline - remember to
            transpose back to (H, W, C) before calling conv3_backward, since
            conv3 works channel-last).
    """
    _validate_backward_inputs(d_output, cache)

    in_c, in_h, in_w = cache["input_shape"]
    num_pixels = cache["num_pixels"]
    d_output = d_output.astype(np.float32, copy=False)

    # Every spatial position of channel c gets the SAME value: d_output[c] / (H*W).
    # Broadcasting handles this without any explicit loop: (C,) -> (C,1,1) -> (C,H,W).
    per_pixel_grad = (d_output / np.float32(num_pixels)).reshape(in_c, 1, 1)
    d_input = np.broadcast_to(per_pixel_grad, (in_c, in_h, in_w)).astype(np.float32)

    return d_input


# ==============================
# SELF-TEST: shape check + numerical gradient check
# ==============================
if __name__ == "__main__":
    print("=" * 55)
    print("CODE_BLOCK: GAP - FORWARD/BACKWARD SELF-TEST")
    print("=" * 55)

    rng = np.random.default_rng(0)

    # ---- 1. Shape check on a real-size-shaped (but synthetic) input ----
    real_input = rng.standard_normal((128, 354, 354)).astype(np.float32)
    output, cache = gap_forward(real_input, report_progress=False)
    assert output.shape == (128,), f"unexpected output shape {output.shape}"
    print(f"shape check on 128x354x354 input -> output {output.shape}  OK")

    # ---- 2. Manual spot-check: channel 0 ----
    manual_avg_0 = np.sum(real_input[0]) / (354 * 354)
    assert np.isclose(manual_avg_0, output[0], atol=1e-4), "spot-check channel-0 average mismatch"
    print(f"spot-check channel 0: sum/num_pixels={manual_avg_0:.8f}, output[0]={output[0]:.8f}  OK")

    # ---- 3. Numerical gradient check on a small synthetic input ----
    print("\nRunning numerical gradient check on a small (4, 5, 5) input...")
    small_input = rng.standard_normal((4, 5, 5)).astype(np.float32)
    out, cache = gap_forward(small_input)
    assert out.shape == (4,), f"unexpected small output shape {out.shape}"

    d_out_fixed = rng.standard_normal(out.shape).astype(np.float32)
    d_input = gap_backward(d_out_fixed, cache)
    assert d_input.shape == small_input.shape, "d_input shape must match forward input shape"

    # Do the finite-difference math in float64 for a clean, non-noisy check
    # (this layer is small enough that float32 would likely be fine too, but
    # float64 costs nothing here and removes any doubt).
    def scalar_loss64(inp64):
        c, h, w = inp64.shape
        num_pixels = h * w
        o = np.array([np.sum(inp64[ch]) / num_pixels for ch in range(c)])
        return np.sum(o * d_out_fixed.astype(np.float64))

    eps = 1e-4
    n_samples = 8
    print("\n--- d_input gradient check (8 random elements) ---")
    for _ in range(n_samples):
        idx = tuple(rng.integers(0, s) for s in small_input.shape)
        base = small_input.astype(np.float64)
        orig = base[idx]

        base[idx] = orig + eps
        loss_plus = scalar_loss64(base)
        base[idx] = orig - eps
        loss_minus = scalar_loss64(base)
        base[idx] = orig

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = float(d_input[idx])
        rel_err = abs(numeric - analytic) / max(1e-6, abs(numeric) + abs(analytic))

        print(f"  idx={idx}  numeric={numeric:+.6f}  analytic={analytic:+.6f}  rel_err={rel_err:.2e}")
        assert rel_err < 1e-4, f"d_input mismatch at {idx}: rel_err={rel_err:.2e}"

    # ---- 4. Sanity property: gradient is UNIFORM across each channel ----
    # (average pooling spreads gradient evenly - unlike max pooling's routing)
    for c in range(small_input.shape[0]):
        assert np.allclose(d_input[c], d_input[c, 0, 0]), \
            f"channel {c}: d_input should be uniform across all spatial positions"
    print("\nuniformity check passed: d_input is constant across every spatial position, per channel.")

    print("\nAll gradient checks passed: gap_backward matches numerical differentiation.")
    print("GAP forward/backward self-test PASSED.")