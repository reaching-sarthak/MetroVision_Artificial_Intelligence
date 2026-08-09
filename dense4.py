"""
CODE_BLOCK: DENSE4
Final fully connected classification layer: 128 -> 4

NumPy only.

Architecture:
    Input  : (128,)   dropout output
    Output : (4,)     raw class logits

Class order:
    [0] Pothole
    [1] Crack
    [2] Both
    [3] Normal

This block is designed for the new trainflow.py architecture.

The layer does NOT:
    - load files
    - save files
    - create its own persistent layer object
    - perform optimization
    - apply softmax
    - calculate cross-entropy

Instead:

    trainflow.py
          |
          |-- W, b
          |
          |-- dense4_forward()
          |
          |-- softmax()
          |
          |-- cross_entropy()
          |
          |-- dense4_backward()
          |
          |-- optimizer
          |
          └-- update W, b
"""


import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 128
NUM_CLASSES = 4

WEIGHT_SEED = 42

CLASS_NAMES = (
    "Pothole",
    "Crack",
    "Both",
    "Normal"
)

INPUT_SHAPE = (INPUT_SIZE,)
OUTPUT_SHAPE = (NUM_CLASSES,)
WEIGHT_SHAPE = (INPUT_SIZE, NUM_CLASSES)
BIAS_SHAPE = (NUM_CLASSES,)


# ============================================================
# PARAMETER INITIALIZATION
# ============================================================

def initialize_dense4_parameters(
    seed: int = WEIGHT_SEED
) -> dict:
    """
    Initialize the trainable parameters of Dense4.

    Weight initialization:
        Xavier / Glorot

        std = sqrt(2 / (fan_in + fan_out))
            = sqrt(2 / (128 + 4))
            = sqrt(2 / 132)

    Parameters:
        W : (128, 4)
        b : (4,)

    Returns:
        parameters dictionary containing:
            "weights"
            "bias"
    """

    rng = np.random.default_rng(seed)

    xavier_std = np.float32(
        np.sqrt(
            2.0 / (INPUT_SIZE + NUM_CLASSES)
        )
    )

    weights = (
        rng.standard_normal(
            WEIGHT_SHAPE,
            dtype=np.float32
        )
        * xavier_std
    ).astype(np.float32)

    bias = np.zeros(
        BIAS_SHAPE,
        dtype=np.float32
    )

    return {
        "weights": weights,
        "bias": bias
    }


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_dense4_input(
    input_vector: np.ndarray
) -> None:
    """
    Validate the input entering Dense4.
    """

    if not isinstance(input_vector, np.ndarray):
        raise TypeError(
            f"input_vector must be a NumPy array, "
            f"got {type(input_vector)}"
        )

    if input_vector.ndim != 1:
        raise ValueError(
            f"input_vector must be 1-D, "
            f"got shape {input_vector.shape}"
        )

    if input_vector.shape != INPUT_SHAPE:
        raise ValueError(
            f"input_vector must have shape {INPUT_SHAPE}, "
            f"got {input_vector.shape}"
        )


# ============================================================
# FORWARD PASS
# ============================================================

def dense4_forward(
    input_vector: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray
):
    """
    Forward pass through Dense4.

    Mathematical operation:

        Z[k] = sum_i X[i] * W[i,k] + b[k]

    or equivalently:

        Z = X @ W + b

    Input:
        X : (128,)

    Parameters:
        W : (128, 4)
        b : (4,)

    Output:
        Z : (4,)

    No activation is applied.

    These are raw logits and are passed directly to the
    softmax / cross-entropy stage.

    Returns:
        logits
        cache
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_dense4_input(input_vector)

    if weights.shape != WEIGHT_SHAPE:
        raise ValueError(
            f"weights must have shape {WEIGHT_SHAPE}, "
            f"got {weights.shape}"
        )

    if bias.shape != BIAS_SHAPE:
        raise ValueError(
            f"bias must have shape {BIAS_SHAPE}, "
            f"got {bias.shape}"
        )

    # Keep everything float32.
    X = input_vector.astype(
        np.float32,
        copy=False
    )

    W = weights.astype(
        np.float32,
        copy=False
    )

    b = bias.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------------
    # Manual affine transformation
    # --------------------------------------------------------

    logits = np.zeros(
        OUTPUT_SHAPE,
        dtype=np.float32
    )

    for k in range(NUM_CLASSES):

        weighted_sum = np.float32(0.0)

        for i in range(INPUT_SIZE):

            weighted_sum += (
                X[i] * W[i, k]
            )

        logits[k] = (
            weighted_sum + b[k]
        )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    # We need X and W during backward().
    #
    # We do NOT need to cache the logits because Dense4's
    # backward derivative only requires:
    #
    #     X
    #     W
    #     upstream gradient dL/dZ

    cache = {
        "input": X,
        "weights": W
    }

    return logits, cache


# ============================================================
# BACKWARD PASS
# ============================================================

def dense4_backward(
    grad_logits: np.ndarray,
    cache: dict
):
    """
    Backward pass through Dense4.

    The upstream gradient is:

        grad_logits = dL/dZ

    where Z represents the four raw class logits.

    We calculate:

        dL/dW
        dL/db
        dL/dX

    --------------------------------------------------------
    Mathematical derivatives
    --------------------------------------------------------

    Forward:

        Z[k] = sum_i X[i] * W[i,k] + b[k]

    Therefore:

        dL/dW[i,k]
            = X[i] * dL/dZ[k]

    And:

        dL/db[k]
            = dL/dZ[k]

    And:

        dL/dX[i]
            = sum_k dL/dZ[k] * W[i,k]

    --------------------------------------------------------
    Shapes
    --------------------------------------------------------

        grad_logits : (4,)

        grad_weights : (128, 4)

        grad_bias : (4,)

        grad_input : (128,)
    """

    # --------------------------------------------------------
    # Retrieve cached values
    # --------------------------------------------------------

    X = cache["input"]
    W = cache["weights"]

    # --------------------------------------------------------
    # Validate upstream gradient
    # --------------------------------------------------------

    if not isinstance(
        grad_logits,
        np.ndarray
    ):
        raise TypeError(
            "grad_logits must be a NumPy array"
        )

    if grad_logits.shape != OUTPUT_SHAPE:
        raise ValueError(
            f"grad_logits must have shape "
            f"{OUTPUT_SHAPE}, "
            f"got {grad_logits.shape}"
        )

    grad_logits = grad_logits.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------------
    # Allocate gradients
    # --------------------------------------------------------

    grad_weights = np.zeros(
        WEIGHT_SHAPE,
        dtype=np.float32
    )

    grad_bias = np.zeros(
        BIAS_SHAPE,
        dtype=np.float32
    )

    grad_input = np.zeros(
        INPUT_SHAPE,
        dtype=np.float32
    )

    # ========================================================
    # dL / db
    # ========================================================

    # From:
    #
    #     Z[k] = ... + b[k]
    #
    # derivative with respect to b[k] is 1.
    #
    # Therefore:
    #
    #     dL/db[k] = dL/dZ[k]

    for k in range(NUM_CLASSES):

        grad_bias[k] = grad_logits[k]

    # ========================================================
    # dL / dW
    # ========================================================

    # From:
    #
    #     Z[k] = sum_i X[i] * W[i,k] + b[k]
    #
    # therefore:
    #
    #     dZ[k]/dW[i,k] = X[i]
    #
    # and:
    #
    #     dL/dW[i,k] = X[i] * dL/dZ[k]

    for k in range(NUM_CLASSES):

        for i in range(INPUT_SIZE):

            grad_weights[i, k] = (
                X[i] * grad_logits[k]
            )

    # ========================================================
    # dL / dX
    # ========================================================

    # Every input X[i] contributes to ALL four logits.
    #
    # Therefore:
    #
    # dL/dX[i]
    #     = Σ_k dL/dZ[k] * W[i,k]

    for i in range(INPUT_SIZE):

        gradient_sum = np.float32(0.0)

        for k in range(NUM_CLASSES):

            gradient_sum += (
                grad_logits[k] * W[i, k]
            )

        grad_input[i] = gradient_sum

    # --------------------------------------------------------
    # Return gradients
    # --------------------------------------------------------

    gradients = {
        "weights": grad_weights,
        "bias": grad_bias
    }

    return grad_input, gradients


# ============================================================
# OPTIONAL PARAMETER COUNT HELPER
# ============================================================

def dense4_parameter_count() -> int:
    """
    Return the total number of trainable Dense4 parameters.

        weights = 128 * 4 = 512
        biases  = 4

        total = 516
    """

    return (
        INPUT_SIZE * NUM_CLASSES
        + NUM_CLASSES
    )


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CODE_BLOCK: DENSE4 - FORWARD + BACKWARD")
    print("=" * 60)

    # --------------------------------------------------------
    # Initialize parameters
    # --------------------------------------------------------

    parameters = initialize_dense4_parameters()

    W = parameters["weights"]
    b = parameters["bias"]

    print(
        f"\nWeights shape : {W.shape}"
    )

    print(
        f"Bias shape    : {b.shape}"
    )

    print(
        f"Parameters    : "
        f"{dense4_parameter_count()}"
    )

    # --------------------------------------------------------
    # Create test input
    # --------------------------------------------------------

    rng = np.random.default_rng(0)

    test_input = rng.standard_normal(
        INPUT_SHAPE,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, cache = dense4_forward(
        test_input,
        W,
        b
    )

    print("\n--- Forward ---")

    print(
        f"Input shape   : {test_input.shape}"
    )

    print(
        f"Logits shape  : {logits.shape}"
    )

    print(
        f"Logits        : {logits}"
    )

    # --------------------------------------------------------
    # Simulated upstream gradient
    #
    # In the real trainflow this will come from:
    #
    # cross_entropy_backward()
    #
    # and will be dL/dZ.
    # --------------------------------------------------------

    grad_logits = np.array(
        [0.1, -0.2, 0.3, -0.2],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    grad_input, gradients = dense4_backward(
        grad_logits,
        cache
    )

    grad_W = gradients["weights"]
    grad_b = gradients["bias"]

    print("\n--- Backward ---")

    print(
        f"dL/dZ shape   : {grad_logits.shape}"
    )

    print(
        f"dL/dW shape   : {grad_W.shape}"
    )

    print(
        f"dL/db shape   : {grad_b.shape}"
    )

    print(
        f"dL/dX shape   : {grad_input.shape}"
    )

    # --------------------------------------------------------
    # Shape assertions
    # --------------------------------------------------------

    assert logits.shape == (4,)
    assert grad_input.shape == (128,)
    assert grad_W.shape == (128, 4)
    assert grad_b.shape == (4,)

    assert logits.dtype == np.float32
    assert grad_input.dtype == np.float32
    assert grad_W.dtype == np.float32
    assert grad_b.dtype == np.float32

    # --------------------------------------------------------
    # Manual spot checks
    # --------------------------------------------------------

    # Check gradient for W[0,0]:

    expected_grad_W00 = (
        test_input[0] * grad_logits[0]
    )

    assert np.isclose(
        grad_W[0, 0],
        expected_grad_W00
    )

    # Check gradient for bias 0:

    assert np.isclose(
        grad_b[0],
        grad_logits[0]
    )

    # Check gradient for input 0:

    expected_grad_X0 = np.float32(0.0)

    for k in range(NUM_CLASSES):

        expected_grad_X0 += (
            grad_logits[k] * W[0, k]
        )

    assert np.isclose(
        grad_input[0],
        expected_grad_X0
    )

    print(
        "\nGradient spot-checks : PASSED"
    )

    print(
        "Dense4 forward/backward block executed successfully."
    )