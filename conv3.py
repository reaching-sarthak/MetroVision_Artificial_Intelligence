"""
Code_Block: Conv3 - Third convolutional layer of the CNN, from scratch (NumPy only).

No TensorFlow / PyTorch / OpenCV / scipy convolution helpers are used.
Every multiply and add of the convolution is performed manually.

Structurally identical to Conv1/Conv2 - same math, same backward derivation -
just different filter size / channel counts, since it sits one layer deeper:

    silu2_output (356,356,64) -> conv3_forward -> gap   (output: 354,354,128)

This file exposes exactly two functions meant to be called from trainflow.py:

    conv3_forward(input_tensor, filters, biases)  -> output, cache
    conv3_backward(d_output, cache)               -> d_input, d_filters, d_biases

`filters` and `biases` are NEVER created or stored inside this file during
training - trainflow.py owns them (loaded from a checkpoint, or freshly
created via initialize_filters()/initialize_biases() the very first time
there is no checkpoint yet) and passes them in on every call.

------------------------------------------------------------------------------
MATH: DISCRETE 3-D CONVOLUTION (valid mode, stride = 1)
------------------------------------------------------------------------------
For filter k, at each spatial output position (i, j):

    output[i, j, k] = b[k] + Σ_c Σ_u Σ_v  input[i+u, j+v, c] * W[k, u, v, c]

where
    c    ∈ {0 ... 63}  indexes the input channel (SiLU2's 64 feature maps)
    u, v ∈ {0, 1, 2}   indexes inside the 3x3 kernel window
    W = filters        shape (128, 3, 3, 64)
    b = biases         shape (128,)

Each filter produces exactly ONE feature map: the 3x3x64 (576-element) patch
is dotted against the 3x3x64 filter -> one scalar per spatial position per filter.

------------------------------------------------------------------------------
MATH: OUTPUT SIZE (valid convolution, stride 1, kernel F=3)
------------------------------------------------------------------------------
    H_out = H_in - F + 1     (e.g. 356 - 3 + 1 = 354)
    W_out = W_in - F + 1

Computed dynamically from whatever input_tensor is passed in, rather than
hardcoded to 356 - this lets the SAME functions be exercised on small
synthetic inputs for gradient-checking (see __main__) without touching the
real 356x356x64 pipeline shape at all.

------------------------------------------------------------------------------
MATH: BACKWARD PASS (identical derivation to Conv1/Conv2, different sizes)
------------------------------------------------------------------------------
Given the upstream gradient d_output = ∂L/∂output (same shape as output):

  ∂L/∂b[k] = Σ_i Σ_j  d_output[i,j,k]

  ∂L/∂W[k,u,v,c] = Σ_i Σ_j  d_output[i,j,k] * input[i+u, j+v, c]
      -> at every spatial position, outer product of d_output[i,j,:] (128,)
         with the flattened 3x3x64 patch (576,), summed over all positions.

  ∂L/∂input[x,y,c] = Σ_k Σ_u Σ_v  d_output[x-u, y-v, k] * W[k,u,v,c]
      -> at every output position, "smear" d_output[i,j,:] back across the
         3x3x64 patch of input it came from, weighted by every filter:
         d_input[i:i+3, j:j+3, :] += d_output[i,j,:] @ flat_filters.
"""

import numpy as np

# ==============================
# CONFIG (fixed architecture choices - NOT input-size dependent)
# ==============================
NUM_FILTERS = 128          # number of convolution filters -> 128 output feature maps
FILTER_SIZE = 3            # square kernel: 3 x 3
IN_CHANNELS = 64            # one channel per Conv2/SiLU2 feature map
STRIDE = 1                  # convolution stride (this layer only supports stride 1)
WEIGHT_SEED = 42            # fixed seed for reproducible FIRST-EVER weight init


# ==============================
# WEIGHT / BIAS INITIALIZATION
# (called by trainflow.py exactly ONCE, only when no checkpoint exists yet)
# ==============================
def initialize_filters(seed: int = WEIGHT_SEED) -> np.ndarray:
    """
    Create the trainable filter bank, shape (128, 3, 3, 64), dtype float32.

    Small random-normal (std = 0.05) break-symmetry init. Called once, ever,
    per training run that starts from scratch - after that, trainflow.py
    keeps reusing (and updating) this same array in memory / checkpoint.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, IN_CHANNELS))
            * np.float32(0.05)).astype(np.float32)


def initialize_biases() -> np.ndarray:
    """Create one trainable bias per filter, shape (128,), dtype float32, starting at 0."""
    return np.zeros(NUM_FILTERS, dtype=np.float32)


# ==============================
# VALIDATION
# ==============================
def _validate_forward_inputs(input_tensor: np.ndarray, filters: np.ndarray, biases: np.ndarray) -> None:
    if not isinstance(input_tensor, np.ndarray):
        raise TypeError(f"input_tensor must be a NumPy array, got {type(input_tensor)}")
    if input_tensor.ndim != 3:
        raise ValueError(f"input_tensor must be 3-D (H, W, C), got shape {input_tensor.shape}")
    h, w, c = input_tensor.shape
    if c != IN_CHANNELS:
        raise ValueError(f"input_tensor must have {IN_CHANNELS} channels, got {c}")
    if h < FILTER_SIZE or w < FILTER_SIZE:
        raise ValueError(f"input spatial size ({h},{w}) smaller than filter size {FILTER_SIZE}")
    if filters.shape != (NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, IN_CHANNELS):
        raise ValueError(f"filters must have shape {(NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, IN_CHANNELS)}, "
                         f"got {filters.shape}")
    if biases.shape != (NUM_FILTERS,):
        raise ValueError(f"biases must have shape ({NUM_FILTERS},), got {biases.shape}")


def _validate_backward_inputs(d_output: np.ndarray, cache: dict) -> None:
    if "input" not in cache or "filters" not in cache:
        raise KeyError("cache must contain 'input' and 'filters' (as returned by conv3_forward)")
    expected_shape = cache["output_shape"]
    if d_output.shape != expected_shape:
        raise ValueError(f"d_output shape {d_output.shape} != forward output shape {expected_shape}")


# ==============================
# FORWARD PASS
# ==============================
def conv3_forward(input_tensor: np.ndarray,
                   filters: np.ndarray,
                   biases: np.ndarray,
                   report_progress: bool = False) -> tuple:
    """
    Manual forward convolution: 3x3x64 filters, stride 1, valid padding.

    At every spatial position, extract the 3x3x64 patch and dot it against
    all 128 flattened filters at once (mathematically identical to computing
    sum(patch * filter_k) + bias_k separately for each of the 128 filters -
    batching them into one NumPy call is purely a speed optimization).

    Args:
        input_tensor: (H, W, 64) float array. H, W >= 3. (Real pipeline: 356x356x64.)
        filters: (128, 3, 3, 64) float array - trainable weights, owned by trainflow.py.
        biases: (128,) float array - trainable biases, owned by trainflow.py.
        report_progress: print a % line every ~10% of rows.

    Returns:
        output: (H-2, W-2, 128) float32 volume.
        cache: dict with everything conv3_backward() needs:
            cache["input"]        : the (H, W, 64) input, exactly as received
            cache["filters"]      : the (128, 3, 3, 64) filters, exactly as received
            cache["output_shape"] : the output's shape, for backward's own validation
    """
    _validate_forward_inputs(input_tensor, filters, biases)

    input_tensor = input_tensor.astype(np.float32, copy=False)
    filters = filters.astype(np.float32, copy=False)
    biases = biases.astype(np.float32, copy=False)

    in_h, in_w, _ = input_tensor.shape
    out_h = in_h - FILTER_SIZE + 1
    out_w = in_w - FILTER_SIZE + 1

    output = np.zeros((out_h, out_w, NUM_FILTERS), dtype=np.float32)

    # Flatten each filter to a (576,) row ONCE: flat_filters is (128, 576).
    flat_filters = filters.reshape(NUM_FILTERS, FILTER_SIZE * FILTER_SIZE * IN_CHANNELS)

    progress_step = max(1, out_h // 10)
    for i in range(out_h):
        for j in range(out_w):
            patch = input_tensor[i:i + FILTER_SIZE, j:j + FILTER_SIZE, :]
            flat_patch = patch.reshape(FILTER_SIZE * FILTER_SIZE * IN_CHANNELS)
            # dot against all 128 filters at once, then add all 128 biases
            output[i, j, :] = flat_filters.dot(flat_patch) + biases
        if report_progress and (i % progress_step == 0 or i == out_h - 1):
            print(f"  conv3 forward: row {i + 1}/{out_h} ({(i + 1) / out_h * 100:.0f}%)")

    cache = {
        "input": input_tensor,
        "filters": filters,
        "output_shape": output.shape,
    }
    return output, cache


# ==============================
# BACKWARD PASS
# ==============================
def conv3_backward(d_output: np.ndarray,
                    cache: dict,
                    report_progress: bool = False) -> tuple:
    """
    Manual backward pass for Conv3. See the module docstring for the three
    gradient formulas this implements.

    Args:
        d_output: (H-2, W-2, 128) float array - ∂L/∂output, handed down from
            the next layer's backward() (GAP, in the real pipeline).
        cache: the dict returned by conv3_forward() for this same input.
        report_progress: print a % line every ~10% of rows.

    Returns:
        d_input   : (H, W, 64)     float32 - ∂L/∂input, to hand to the PREVIOUS
                     layer's backward() (SiLU2, in the real pipeline).
        d_filters : (128, 3, 3, 64) float32 - ∂L/∂filters, for the optimizer.
        d_biases  : (128,)          float32 - ∂L/∂biases,  for the optimizer.
    """
    _validate_backward_inputs(d_output, cache)

    input_tensor = cache["input"]
    filters = cache["filters"]
    d_output = d_output.astype(np.float32, copy=False)

    in_h, in_w, _ = input_tensor.shape
    out_h, out_w, _ = cache["output_shape"]

    # ---- ∂L/∂biases: sum the upstream gradient over all spatial positions ----
    d_biases = d_output.sum(axis=(0, 1)).astype(np.float32)

    # ---- ∂L/∂filters and ∂L/∂input: computed together in one pass ----
    flat_filters = filters.reshape(NUM_FILTERS, FILTER_SIZE * FILTER_SIZE * IN_CHANNELS)  # (128, 576)
    flat_d_filters = np.zeros((NUM_FILTERS, FILTER_SIZE * FILTER_SIZE * IN_CHANNELS), dtype=np.float32)
    d_input = np.zeros((in_h, in_w, IN_CHANNELS), dtype=np.float32)

    progress_step = max(1, out_h // 10)
    for i in range(out_h):
        for j in range(out_w):
            d_out_vec = d_output[i, j, :]                       # (128,) - gradient at this position

            # ∂L/∂filters contribution: outer product of d_out_vec (128,) and
            # the flattened patch (576,) this position used -> (128, 576).
            patch = input_tensor[i:i + FILTER_SIZE, j:j + FILTER_SIZE, :]
            flat_patch = patch.reshape(FILTER_SIZE * FILTER_SIZE * IN_CHANNELS)
            flat_d_filters += np.outer(d_out_vec, flat_patch)

            # ∂L/∂input contribution: "smear" d_out_vec back across the same
            # 3x3x64 patch, weighted by every filter -> (576,) -> (3,3,64).
            flat_d_patch = d_out_vec.dot(flat_filters)           # (128,) @ (128,576) -> (576,)
            d_input[i:i + FILTER_SIZE, j:j + FILTER_SIZE, :] += flat_d_patch.reshape(
                FILTER_SIZE, FILTER_SIZE, IN_CHANNELS
            )
        if report_progress and (i % progress_step == 0 or i == out_h - 1):
            print(f"  conv3 backward: row {i + 1}/{out_h} ({(i + 1) / out_h * 100:.0f}%)")

    d_filters = flat_d_filters.reshape(NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, IN_CHANNELS)

    return d_input.astype(np.float32), d_filters.astype(np.float32), d_biases


# ==============================
# SELF-TEST: shape check + numerical gradient check
# ==============================
if __name__ == "__main__":
    print("=" * 55)
    print("CODE_BLOCK: CONV3 - FORWARD/BACKWARD SELF-TEST")
    print("=" * 55)

    rng = np.random.default_rng(0)

    # ---- 1. Shape sanity check on a real-size-shaped (but synthetic) input ----
    real_input = rng.standard_normal((356, 356, 64)).astype(np.float32)
    filters = initialize_filters()
    biases = initialize_biases()
    output, cache = conv3_forward(real_input, filters, biases, report_progress=False)
    assert output.shape == (354, 354, 128), f"unexpected output shape {output.shape}"
    print(f"shape check on 356x356x64 input -> output {output.shape}  OK")

    # ---- 2. Numerical gradient check on a SMALL synthetic input ----
    print("\nRunning numerical gradient check on a small (7, 7, 64) input...")
    small_input = rng.standard_normal((7, 7, IN_CHANNELS)).astype(np.float32)
    small_filters = (rng.standard_normal((NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, IN_CHANNELS))
                      * 0.1).astype(np.float32)
    small_biases = rng.standard_normal(NUM_FILTERS).astype(np.float32) * 0.1

    out, cache = conv3_forward(small_input, small_filters, small_biases)
    d_out_fixed = rng.standard_normal(out.shape).astype(np.float32)

    d_input, d_filters, d_biases = conv3_backward(d_out_fixed, cache)

    # ---- float64 shadow forward pass, used ONLY for finite differencing ----
    # conv3_forward always downcasts to float32 internally, which is correct
    # for the real (float32) pipeline - but the loss subtraction inside a
    # finite-difference check (loss_plus - loss_minus, divided by 2*eps)
    # amplifies float32 rounding by up to 1/(2*eps) = 500x. At this layer's
    # size (128 filters * 576-dim patches), that occasionally produces a
    # spurious few-percent "error" even when the analytic gradient is exactly
    # right. Running the perturbation math in float64 removes that artifact;
    # the ANALYTIC values being checked still come from the real float32
    # conv3_backward() above - only the numerical reference is higher-precision.
    def forward64(inp, filt, bia):
        in_h, in_w, _ = inp.shape
        out_h, out_w = in_h - FILTER_SIZE + 1, in_w - FILTER_SIZE + 1
        flat_filt = filt.reshape(NUM_FILTERS, FILTER_SIZE * FILTER_SIZE * IN_CHANNELS)
        o = np.zeros((out_h, out_w, NUM_FILTERS), dtype=np.float64)
        for i in range(out_h):
            for j in range(out_w):
                patch = inp[i:i + FILTER_SIZE, j:j + FILTER_SIZE, :].reshape(-1)
                o[i, j, :] = flat_filt.dot(patch) + bia
        return o

    d_out_fixed64 = d_out_fixed.astype(np.float64)

    def scalar_loss64(inp, filt, bia):
        return np.sum(forward64(inp, filt, bia) * d_out_fixed64)

    def numerical_grad(get_array, set_array, shape, n_samples=6, eps=1e-4):
        errors = []
        for _ in range(n_samples):
            idx = tuple(rng.integers(0, s) for s in shape)
            base_input = small_input.astype(np.float64)
            base_filters = small_filters.astype(np.float64)
            base_biases = small_biases.astype(np.float64)
            arr = get_array(base_input, base_filters, base_biases)
            orig = arr[idx]

            arr[idx] = orig + eps
            loss_plus = scalar_loss64(base_input, base_filters, base_biases)
            arr[idx] = orig - eps
            loss_minus = scalar_loss64(base_input, base_filters, base_biases)
            arr[idx] = orig

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = float(set_array[idx])
            rel_err = abs(numeric - analytic) / max(1e-6, abs(numeric) + abs(analytic))
            errors.append((idx, numeric, analytic, rel_err))
        return errors

    print("\n--- d_input gradient check (6 random elements) ---")
    for idx, num, ana, err in numerical_grad(lambda i, f, b: i, d_input, small_input.shape):
        print(f"  idx={idx}  numeric={num:+.6f}  analytic={ana:+.6f}  rel_err={err:.2e}")
        assert err < 1e-2, f"d_input mismatch at {idx}: rel_err={err:.2e}"

    print("\n--- d_filters gradient check (6 random elements) ---")
    for idx, num, ana, err in numerical_grad(lambda i, f, b: f, d_filters, small_filters.shape):
        print(f"  idx={idx}  numeric={num:+.6f}  analytic={ana:+.6f}  rel_err={err:.2e}")
        assert err < 1e-2, f"d_filters mismatch at {idx}: rel_err={err:.2e}"

    print("\n--- d_biases gradient check (6 random elements) ---")
    for idx, num, ana, err in numerical_grad(lambda i, f, b: b, d_biases, small_biases.shape):
        print(f"  idx={idx}  numeric={num:+.6f}  analytic={ana:+.6f}  rel_err={err:.2e}")
        assert err < 1e-2, f"d_biases mismatch at {idx}: rel_err={err:.2e}"

    print("\nAll gradient checks passed: conv3_backward matches numerical differentiation.")
    print("Conv3 forward/backward self-test PASSED.")