"""
Code_Block: SiLU2 (Sigmoid Linear Unit / Swish) activation, from scratch (NumPy only).

No TensorFlow / PyTorch / SciPy / deep-learning-framework activation helpers are used.
The sigmoid and the SiLU product are computed manually with raw NumPy math.

This file exposes exactly two functions meant to be called from trainflow.py:

    silu2_forward(input_tensor)      -> output, cache
    silu2_backward(d_output, cache)  -> d_input

SiLU has NO trainable parameters (no weights, no biases) - there is nothing
for trainflow.py to load from a checkpoint or hand to the optimizer for this
layer. It sits between conv2 and conv3 in the real pipeline:

    conv2_output (356,356,64) -> silu2_forward -> conv3

Math is IDENTICAL to silu1_forward/silu1_backward (block_silu.py) - this is
a separate file only because it sits at a different point in the pipeline
with a different tensor shape (64 channels instead of 32), not because the
function itself behaves any differently.

------------------------------------------------------------------------------
MATH: THE SIGMOID FUNCTION
------------------------------------------------------------------------------
For any scalar x,

    σ(x) = 1 / (1 + exp(-x))

Squashes any real number into the open interval (0, 1); σ(0) = 0.5,
σ(x) -> 1 as x -> +inf, σ(x) -> 0 as x -> -inf. Smooth (infinitely
differentiable), which is exactly what gradient-based training needs.

------------------------------------------------------------------------------
MATH: THE SILU ACTIVATION
------------------------------------------------------------------------------
For any scalar x,

    SiLU(x) = x * σ(x) = x / (1 + exp(-x))

SiLU multiplies each activation by its own sigmoid "gate":
  - large positive x -> σ(x) ≈ 1 -> SiLU(x) ≈ x       (passes signal through)
  - large negative x -> σ(x) ≈ 0 -> SiLU(x) ≈ 0        (suppresses signal)
  - near zero it bends smoothly (small negative dip for x < 0) - no kink,
    no dead region, unlike ReLU.

------------------------------------------------------------------------------
WHY ELEMENT-WISE, AND WHY THE SHAPE NEVER CHANGES
------------------------------------------------------------------------------
output[i,j,k] depends ONLY on input[i,j,k] - no reduction/summation crosses
positions or channels, so SiLU accepts a tensor of ANY shape and returns that
same shape unchanged: (356, 356, 64) -> (356, 356, 64).

------------------------------------------------------------------------------
MATH: BACKWARD PASS
------------------------------------------------------------------------------
Derivative of SiLU (product rule on x·σ(x), using σ'(x) = σ(x)·(1 - σ(x))):

    d(SiLU)/dx = σ(x) + x·σ(x)·(1 - σ(x))
               = σ(x) · (1 + x·(1 - σ(x)))

By the chain rule, given the upstream gradient d_output = ∂L/∂output:

    d_input = d_output * d(SiLU)/dx        (element-wise multiply, same shape)

Entire derivative is built from x (cached input) and σ(x) (cached sigmoid) -
no exp() is ever recomputed in backward().
"""

import numpy as np


# ==============================
# SIGMOID (manual)
# ==============================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute σ(x) = 1 / (1 + exp(-x)) manually, element-wise, in float32.

    Steps (each a raw NumPy op - no framework helpers):
        1. negate:        -x
        2. exponentiate:  exp(-x)
        3. shift:         1 + exp(-x)
        4. reciprocate:   1 / (1 + exp(-x))
    """
    exp_neg_x = np.exp(-x)
    denom = np.float32(1.0) + exp_neg_x
    return np.float32(1.0) / denom


# ==============================
# VALIDATION
# ==============================
def _validate_forward_inputs(input_tensor: np.ndarray) -> None:
    if not isinstance(input_tensor, np.ndarray):
        raise TypeError(f"input_tensor must be a NumPy array, got {type(input_tensor)}")


def _validate_backward_inputs(d_output: np.ndarray, cache: dict) -> None:
    if "input" not in cache or "sigmoid" not in cache:
        raise KeyError("cache must contain 'input' and 'sigmoid' (as returned by silu2_forward)")
    if d_output.shape != cache["input"].shape:
        raise ValueError(f"d_output shape {d_output.shape} != forward input shape {cache['input'].shape}")


# ==============================
# FORWARD PASS
# ==============================
def silu2_forward(input_tensor: np.ndarray) -> tuple:
    """
    Apply SiLU(x) = x · σ(x) to every scalar of the input tensor.

    Args:
        input_tensor: any-shape float array (real pipeline: (356,356,64),
            the conv2 output).

    Returns:
        output: float32, same shape as input_tensor.
        cache: dict with everything silu2_backward() needs:
            cache["input"]   : x, exactly as received (float32)
            cache["sigmoid"] : σ(x), reused directly in the derivative
    """
    _validate_forward_inputs(input_tensor)

    x = input_tensor.astype(np.float32, copy=False)
    sigma = sigmoid(x)
    output = x * sigma

    cache = {
        "input": x,
        "sigmoid": sigma,
    }
    return output, cache


# ==============================
# BACKWARD PASS
# ==============================
def silu2_backward(d_output: np.ndarray, cache: dict) -> np.ndarray:
    """
    Manual backward pass for SiLU2.

        d_input = d_output * [ σ(x) · (1 + x·(1 - σ(x))) ]

    Args:
        d_output: ∂L/∂output, same shape as the forward input/output.
        cache: the dict returned by silu2_forward() for this same input.

    Returns:
        d_input: ∂L/∂input, same shape as d_output, to hand to the PREVIOUS
            layer's backward() (conv2, in the real pipeline).
    """
    _validate_backward_inputs(d_output, cache)

    x = cache["input"]
    sigma = cache["sigmoid"]
    d_output = d_output.astype(np.float32, copy=False)

    # d(SiLU)/dx = σ(x) * (1 + x * (1 - σ(x)))
    local_grad = sigma * (np.float32(1.0) + x * (np.float32(1.0) - sigma))

    d_input = d_output * local_grad
    return d_input.astype(np.float32)


# ==============================
# SELF-TEST: shape check + numerical gradient check
# ==============================
if __name__ == "__main__":
    print("=" * 55)
    print("CODE_BLOCK: SILU2 - FORWARD/BACKWARD SELF-TEST")
    print("=" * 55)

    rng = np.random.default_rng(0)

    # ---- 1. Shape check on a real-size-shaped (but synthetic) input ----
    real_input = rng.standard_normal((356, 356, 64)).astype(np.float32)
    output, cache = silu2_forward(real_input)
    assert output.shape == real_input.shape, f"unexpected output shape {output.shape}"
    print(f"shape check on 356x356x64 input -> output {output.shape}  OK")

    # ---- 2. Known-value sanity check ----
    test_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    test_out, _ = silu2_forward(test_values)
    expected = np.array([-0.238406, -0.268941, 0.0, 0.731059, 1.761594], dtype=np.float32)
    assert np.allclose(test_out, expected, atol=1e-5), "known-value check failed"
    print("known-value check (x in [-2..2]) matches hand-computed SiLU  OK")

    # ---- 3. Numerical gradient check on a small synthetic input ----
    print("\nRunning numerical gradient check on a small (6, 6, 4) input...")
    small_input = rng.standard_normal((6, 6, 4)).astype(np.float32)
    out, cache = silu2_forward(small_input)

    d_out_fixed = rng.standard_normal(out.shape).astype(np.float32)
    d_input = silu2_backward(d_out_fixed, cache)

    def scalar_loss(inp):
        o, _ = silu2_forward(inp)
        return np.sum(o * d_out_fixed)

    eps = 1e-3
    n_samples = 8
    print("\n--- d_input gradient check (8 random elements) ---")
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
        rel_err = abs(numeric - analytic) / max(1e-6, abs(numeric) + abs(analytic))

        print(f"  idx={idx}  numeric={numeric:+.6f}  analytic={analytic:+.6f}  rel_err={rel_err:.2e}")
        assert rel_err < 1e-2, f"d_input mismatch at {idx}: rel_err={rel_err:.2e}"

    print("\nAll gradient checks passed: silu2_backward matches numerical differentiation.")
    print("SiLU2 forward/backward self-test PASSED.")