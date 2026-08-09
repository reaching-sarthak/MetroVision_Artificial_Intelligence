import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 128
OUTPUT_NEURONS = 128

INPUT_SHAPE = (INPUT_SIZE,)
OUTPUT_SHAPE = (OUTPUT_NEURONS,)

WEIGHT_SHAPE = (INPUT_SIZE, OUTPUT_NEURONS)
BIAS_SHAPE = (OUTPUT_NEURONS,)


# ============================================================
# PARAMETER INITIALIZATION
# ============================================================

def initialize_parameters(seed: int = 42) -> tuple:
    """
    Initialize the trainable parameters of Dense128.

    Architecture:
        128 inputs -> 128 outputs

    Weight initialization:
        He initialization

            W ~ N(0, sqrt(2 / fan_in))

    Bias initialization:
        b = 0

    Returns:
        W : shape (128, 128), float32
        b : shape (128,), float32
    """

    rng = np.random.default_rng(seed)

    fan_in = INPUT_SIZE

    he_std = np.float32(
        np.sqrt(2.0 / fan_in)
    )

    W = (
        rng.standard_normal(
            WEIGHT_SHAPE,
            dtype=np.float32
        ) * he_std
    )

    b = np.zeros(
        BIAS_SHAPE,
        dtype=np.float32
    )

    return W, b


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(input_vector: np.ndarray) -> None:
    """
    Validate the input received from Global Average Pooling.
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

def dense128_forward(
    input_vector: np.ndarray,
    W: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    Dense1 forward pass.

    Mathematical operation:

        Z = XW + b

    where:

        X : (128,)
        W : (128, 128)
        b : (128,)
        Z : (128,)

    Every output neuron receives all 128 input features.

    The implementation uses an explicit loop so the mathematics
    remains transparent.

    Parameters:
        input_vector : (128,)
        W            : (128, 128)
        b            : (128,)

    Returns:
        Z : (128,)
    """

    validate_input(input_vector)

    if W.shape != WEIGHT_SHAPE:
        raise ValueError(
            f"W must have shape {WEIGHT_SHAPE}, "
            f"got {W.shape}"
        )

    if b.shape != BIAS_SHAPE:
        raise ValueError(
            f"b must have shape {BIAS_SHAPE}, "
            f"got {b.shape}"
        )

    X = input_vector.astype(
        np.float32,
        copy=False
    )

    W = W.astype(
        np.float32,
        copy=False
    )

    b = b.astype(
        np.float32,
        copy=False
    )

    Z = np.zeros(
        OUTPUT_SHAPE,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Calculate each output neuron.
    #
    # Z[j] = sum_i X[i] * W[i,j] + b[j]
    # --------------------------------------------------------

    for j in range(OUTPUT_NEURONS):

        weighted_sum = np.float32(0.0)

        for i in range(INPUT_SIZE):

            weighted_sum += (
                X[i] * W[i, j]
            )

        Z[j] = (
            weighted_sum + b[j]
        )

    return Z


# ============================================================
# BACKWARD PASS
# ============================================================

def dense128_backward(
    upstream_gradient: np.ndarray,
    input_vector: np.ndarray,
    W: np.ndarray
) -> tuple:
    """
    Dense1 backward pass.

    Forward equation:

        Z = XW + b

    Given:

        dL/dZ

    calculate:

        dL/dX
        dL/dW
        dL/db

    Mathematical derivatives:

        dL/dW[i,j]
            = X[i] * dL/dZ[j]

        dL/db[j]
            = dL/dZ[j]

        dL/dX[i]
            = sum_j dL/dZ[j] * W[i,j]

    Parameters:
        upstream_gradient : dL/dZ, shape (128,)
        input_vector      : X, shape (128,)
        W                 : W, shape (128,128)

    Returns:
        d_input   : dL/dX, shape (128,)
        dW        : dL/dW, shape (128,128)
        db        : dL/db, shape (128,)
    """

    validate_input(input_vector)

    if upstream_gradient.shape != OUTPUT_SHAPE:
        raise ValueError(
            f"upstream_gradient must have shape "
            f"{OUTPUT_SHAPE}, "
            f"got {upstream_gradient.shape}"
        )

    if W.shape != WEIGHT_SHAPE:
        raise ValueError(
            f"W must have shape {WEIGHT_SHAPE}, "
            f"got {W.shape}"
        )

    X = input_vector.astype(
        np.float32,
        copy=False
    )

    dZ = upstream_gradient.astype(
        np.float32,
        copy=False
    )

    W = W.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------------
    # Allocate gradients
    # --------------------------------------------------------

    d_input = np.zeros(
        INPUT_SHAPE,
        dtype=np.float32
    )

    dW = np.zeros(
        WEIGHT_SHAPE,
        dtype=np.float32
    )

    db = np.zeros(
        BIAS_SHAPE,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # dL/db
    #
    # Z[j] contains + b[j].
    #
    # Therefore:
    #
    # dL/db[j] = dL/dZ[j]
    # --------------------------------------------------------

    for j in range(OUTPUT_NEURONS):

        db[j] = dZ[j]

    # --------------------------------------------------------
    # dL/dW
    #
    # Z[j] = sum_i X[i]W[i,j] + b[j]
    #
    # Therefore:
    #
    # dL/dW[i,j] = X[i] * dL/dZ[j]
    # --------------------------------------------------------

    for i in range(INPUT_SIZE):

        for j in range(OUTPUT_NEURONS):

            dW[i, j] = (
                X[i] * dZ[j]
            )

    # --------------------------------------------------------
    # dL/dX
    #
    # Each input X[i] connects to ALL 128 output neurons.
    #
    # Therefore:
    #
    # dL/dX[i]
    #     = sum_j dL/dZ[j] * W[i,j]
    # --------------------------------------------------------

    for i in range(INPUT_SIZE):

        gradient_sum = np.float32(0.0)

        for j in range(OUTPUT_NEURONS):

            gradient_sum += (
                dZ[j] * W[i, j]
            )

        d_input[i] = gradient_sum

    return d_input, dW, db