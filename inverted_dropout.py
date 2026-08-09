import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_SIZE = 128
DROPOUT_RATE = 0.40


# ============================================================
# VALIDATION
# ============================================================

def validate_input(input_tensor: np.ndarray) -> None:
    """
    Validate the Dense128 activation entering Dropout.

    Expected shape:
        (128,)
    """

    if not isinstance(input_tensor, np.ndarray):
        raise TypeError(
            f"input_tensor must be a NumPy array, "
            f"got {type(input_tensor)}"
        )

    if input_tensor.ndim != 1:
        raise ValueError(
            f"input_tensor must be 1-D, "
            f"got shape {input_tensor.shape}"
        )

    if input_tensor.shape != (INPUT_SIZE,):
        raise ValueError(
            f"input_tensor must have shape "
            f"({INPUT_SIZE},), got {input_tensor.shape}"
        )


# ============================================================
# FORWARD PASS
# ============================================================

def forward(
    input_tensor: np.ndarray,
    dropout_rate: float = DROPOUT_RATE,
    training: bool = True,
    rng: np.random.Generator | None = None
) -> tuple:
    """
    Apply inverted dropout.

    During training:

        keep_probability = 1 - dropout_rate

        mask ~ Bernoulli(keep_probability)

        output = (mask * input) / keep_probability

    During inference:

        output = input

    Returns:

        output : Dropout output, shape (128,)
        mask   : Binary dropout mask, shape (128,)

    The mask is returned because it is required during
    backpropagation.
    """

    validate_input(input_tensor)

    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(
            f"dropout_rate must be in [0, 1), "
            f"got {dropout_rate}"
        )

    X = input_tensor.astype(
        np.float32,
        copy=False
    )

    keep_probability = np.float32(
        1.0 - dropout_rate
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    if not training:

        # Dropout is disabled during inference.
        #
        # Every activation passes through unchanged.

        output = X.copy()

        # A mask of ones is convenient for backward compatibility,
        # although backward() should normally only be called during
        # training.

        mask = np.ones(
            INPUT_SIZE,
            dtype=np.float32
        )

        return output, mask

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    if rng is None:
        rng = np.random.default_rng()

    # Generate Bernoulli mask.
    #
    # 1 -> neuron survives
    # 0 -> neuron is dropped
    #
    # With dropout_rate = 0.40:
    #
    # P(mask = 1) = 0.60
    # P(mask = 0) = 0.40

    mask = (
        rng.random(INPUT_SIZE) < keep_probability
    ).astype(np.float32)

    # Inverted dropout:
    #
    # surviving activation:
    #
    #     X / 0.60
    #
    # dropped activation:
    #
    #     0 / 0.60 = 0

    output = (
        mask * X
    ) / keep_probability

    output = output.astype(
        np.float32,
        copy=False
    )

    return output, mask


# ============================================================
# BACKWARD PASS
# ============================================================

def backward(
    upstream_gradient: np.ndarray,
    mask: np.ndarray,
    dropout_rate: float = DROPOUT_RATE
) -> np.ndarray:
    """
    Backpropagate through inverted dropout.

    Forward equation:

        output = (mask * input) / keep_probability

    Therefore:

        dL/dinput
            =
        dL/doutput * mask / keep_probability

    If a neuron was dropped:

        mask = 0

        therefore:

        dL/dinput = 0

    If a neuron survived:

        mask = 1

        therefore:

        dL/dinput =
            dL/doutput / keep_probability

    Parameters:

        upstream_gradient:
            dL/d(output), shape (128,)

        mask:
            dropout mask generated during forward(),
            shape (128,)

        dropout_rate:
            normally 0.40

    Returns:

        d_input:
            dL/d(input), shape (128,)
    """

    if not isinstance(
        upstream_gradient,
        np.ndarray
    ):
        raise TypeError(
            "upstream_gradient must be a NumPy array"
        )

    if not isinstance(
        mask,
        np.ndarray
    ):
        raise TypeError(
            "mask must be a NumPy array"
        )

    if upstream_gradient.shape != (INPUT_SIZE,):
        raise ValueError(
            f"upstream_gradient must have shape "
            f"({INPUT_SIZE},), "
            f"got {upstream_gradient.shape}"
        )

    if mask.shape != (INPUT_SIZE,):
        raise ValueError(
            f"mask must have shape "
            f"({INPUT_SIZE},), "
            f"got {mask.shape}"
        )

    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(
            f"dropout_rate must be in [0, 1), "
            f"got {dropout_rate}"
        )

    keep_probability = np.float32(
        1.0 - dropout_rate
    )

    d_output = upstream_gradient.astype(
        np.float32,
        copy=False
    )

    # Chain rule:
    #
    # dL/dinput =
    #
    # dL/doutput
    #       *
    # mask
    #       /
    # keep_probability

    d_input = (
        d_output * mask
    ) / keep_probability

    return d_input.astype(
        np.float32,
        copy=False
    )