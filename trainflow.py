"""
trainflow.py

Full training flow for the road-damage CNN.

Responsibilities:
    1. Run the complete forward pass.
    2. Calculate Softmax probabilities.
    3. Calculate numerically stable Cross-Entropy loss.
    4. Start the backward pass with dL/dlogits = probabilities - target.
    5. Pass gradients backward through every trainable layer.
    6. Collect all trainable parameters and gradients.
    7. Update every parameter using ONE shared AdamW optimizer.

NumPy only.
"""

import numpy as np

# ============================================================
# IMPORT MODEL COMPONENTS
# ============================================================

# Replace these imports with the actual filenames/classes
# in your project.

from conv1 import conv1_forward, conv1_backward
from conv2 import conv2_forward, conv2_backward
from conv3 import conv3_forward, conv3_backward
from dense128 import dense128_forward, dense128_backward
from dense4 import dense4_forward, dense4_backward
from gap import gap_backward, gap_forward
from inverted_dropout import forward


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 4

CLASS_NAMES = (
    "Pothole",
    "Crack",
    "Both",
    "Normal",
)
CLASS_WEIGHTS = np.array(
    [2.505195, 1.078859, 2.505195, 0.439608],
    dtype=np.float32
)
# AdamW settings
LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.01


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw class logits into probabilities.

    Mathematical definition:

        P_k = exp(Z_k) / sum_j exp(Z_j)

    A numerically stable implementation subtracts the maximum
    logit before exponentiation:

        shifted_Z = Z - max(Z)

        P_k = exp(shifted_Z_k)
              ------------------
              sum_j exp(shifted_Z_j)

    Subtracting the same constant from every logit does NOT
    change the resulting probabilities.

    Args:
        logits: shape (4,)

    Returns:
        probabilities: shape (4,), float32
    """

    logits = np.asarray(logits, dtype=np.float32)

    if logits.shape != (NUM_CLASSES,):
        raise ValueError(
            f"Expected logits shape ({NUM_CLASSES},), "
            f"got {logits.shape}"
        )

    # Numerical stability.
    shifted_logits = logits - np.max(logits)

    exponentials = np.exp(shifted_logits)

    probabilities = (
        exponentials / np.sum(exponentials)
    )

    return probabilities.astype(np.float32)


# ============================================================
# CROSS-ENTROPY LOSS
# ============================================================

# ============================================================
# CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(
    logits: np.ndarray,
    target_class: int
):
    """
    Calculate softmax probabilities and CLASS-WEIGHTED
    cross-entropy loss.

    For target class y:

        L = -w_y * log(P_y)

    where w_y is the weight assigned to the target class.

    The corresponding gradient is:

        dL/dZ = w_y * (P - Y)

    Class indices:

        0 = Pothole
        1 = Crack
        2 = Both
        3 = Normal

    Args:
        logits:
            Raw class logits, shape (4,)

        target_class:
            Integer class index.

    Returns:
        loss:
            Scalar float32

        probabilities:
            Softmax probabilities, shape (4,)

        grad_logits:
            dL/dlogits, shape (4,)
    """

    logits = np.asarray(logits, dtype=np.float32)

    if logits.shape != (NUM_CLASSES,):
        raise ValueError(
            f"Expected logits shape ({NUM_CLASSES},), "
            f"got {logits.shape}"
        )

    if not isinstance(target_class, (int, np.integer)):
        raise TypeError(
            "target_class must be an integer class index"
        )

    if not 0 <= target_class < NUM_CLASSES:
        raise ValueError(
            f"target_class must be between 0 and "
            f"{NUM_CLASSES - 1}, got {target_class}"
        )

    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    probabilities = softmax(logits)

    # --------------------------------------------------------
    # GET WEIGHT FOR THE TRUE CLASS
    # --------------------------------------------------------

    target_weight = CLASS_WEIGHTS[target_class]

    # --------------------------------------------------------
    # NUMERICALLY STABLE CROSS-ENTROPY
    # --------------------------------------------------------

    shifted_logits = logits - np.max(logits)

    log_sum_exp = (
        np.log(np.sum(np.exp(shifted_logits)))
        + np.max(logits)
    )

    # Ordinary cross-entropy:
    #
    #     -log(P_y)
    #
    # becomes:
    #
    #     -w_y * log(P_y)

    loss = (
        target_weight
        * (
            -logits[target_class]
            + log_sum_exp
        )
    )

    loss = np.float32(loss)

    # --------------------------------------------------------
    # GRADIENT WITH RESPECT TO LOGITS
    # --------------------------------------------------------

    # Ordinary softmax + cross-entropy:
    #
    #     dL/dZ = P - Y
    #
    # Class-weighted version:
    #
    #     dL/dZ = w_y * (P - Y)

    grad_logits = probabilities.copy()

    grad_logits[target_class] -= np.float32(1.0)

    grad_logits *= target_weight

    grad_logits = grad_logits.astype(np.float32)

    return loss, probabilities, grad_logits

# ============================================================
# ONE-HOT LABEL HELPER
# ============================================================

def class_name(target_class: int) -> str:
    """Convert numerical class index into human-readable name."""

    return CLASS_NAMES[target_class]


# ============================================================
# TRAINING FLOW
# ============================================================

class TrainFlow:

    def __init__(
        self,
        conv1,
        conv2,
        conv3,
        dense128,
        dense4,
        dropout_rate=0.40,
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        weight_decay=WEIGHT_DECAY,
    ):
        """
        Create the complete model training flow.

        IMPORTANT:

        The layer objects passed here already own their trainable
        parameters.

        TrainFlow does NOT create duplicate weights.

        It simply connects the layers together.
        """

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        self.dense128 = dense128
        self.dense4 = dense4

        self.dropout_rate = dropout_rate

        # ----------------------------------------------------
        # ONE SHARED ADAMW OPTIMIZER
        # ----------------------------------------------------

        self.optimizer = AdamW(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )

        # Dropout mask from the most recent training pass.
        self.dropout_mask = None

        # Cached forward values.
        self.cache = {}

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        input_tensor: np.ndarray,
        training: bool = True,
        dropout_seed: int | None = None,
    ):
        """
        Run the complete forward pass.

        Flow:

            input
              ↓
            Conv1
              ↓
            SiLU
              ↓
            Conv2
              ↓
            ...
              ↓
            Conv3
              ↓
            GAP
              ↓
            Dense1
              ↓
            Dropout
              ↓
            Dense4
              ↓
            logits

        IMPORTANT:

        Softmax is intentionally NOT inside Dense4.

        It is calculated afterward by the loss function.
        """

        # ----------------------------------------------------
        # CONV1
        # ----------------------------------------------------

        x = self.conv1.conv1_forward(input_tensor)

        # ----------------------------------------------------
        # CONV2
        # ----------------------------------------------------

        x = self.conv2.conv2_forward(x)

        # ----------------------------------------------------
        # CONV3
        # ----------------------------------------------------

        x = self.conv3.conv3_forward(x)

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        # Replace this with your actual GAP function/class.
        x = gap_forward(x)

        # ----------------------------------------------------
        # DENSE1
        # ----------------------------------------------------

        x = self.dense128.dense128_forward(x)

        # ----------------------------------------------------
        # DROPOUT
        # ----------------------------------------------------

        output, mask = forward(
            x,
            dropout_rate=self.dropout_rate,
            training=training
        )

        x = output
        self.dropout_mask = mask
        self.cache["dropout_mask"] = mask
        self.cache["dropout_keep_prob"] = np.float32(1.0 - self.dropout_rate)


            # ----------------------------------------------------
        # DENSE4
        # ----------------------------------------------------

        logits = self.dense4.dense4_forward(x)
        return logits

    # ========================================================
    # LOSS
    # ========================================================

    def compute_loss(
        self,
        logits: np.ndarray,
        target_class: int,
    ):
        """
        Calculate:

            logits
              ↓
            softmax
              ↓
            cross entropy

        Returns:
            loss
            probabilities
            grad_logits
        """

        return cross_entropy_loss(
            logits,
            target_class
        )

    # ========================================================
    # BACKWARD
    # ========================================================

    def backward(
        self,
        grad_logits: np.ndarray,
    ):
        """
        Propagate dL/dlogits backward through the network.

        Reverse order:

            Dense4
              ↓
            Dropout
              ↓
            Dense1
              ↓
            GAP
              ↓
            Conv3
              ↓
            Conv2
              ↓
            Conv1
        """

        # ----------------------------------------------------
        # DENSE4 BACKWARD
        # ----------------------------------------------------

        grad = self.dense4.dense4_backward(grad_logits)

        # ----------------------------------------------------
        # DROPOUT BACKWARD
        # ----------------------------------------------------

        # Inverted dropout derivative:

        #
        # dL/dx = dL/dy * mask / keep_probability
        #

        grad = grad * self.cache["dropout_mask"] / self.cache["dropout_keep_prob"]
        # ----------------------------------------------------
        # DENSE1 BACKWARD
        # ----------------------------------------------------

        grad = self.dense128.dense128_backward(grad)

        # ----------------------------------------------------
        # GAP BACKWARD
        # ----------------------------------------------------

        grad = gap_backward(grad)

        # ----------------------------------------------------
        # CONV3 BACKWARD
        # ----------------------------------------------------

        grad = self.conv3.conv3_backward(grad)

        # ----------------------------------------------------
        # CONV2 BACKWARD
        # ----------------------------------------------------

        grad = self.conv2.conv2_backward(grad)

        # ----------------------------------------------------
        # CONV1 BACKWARD
        # ----------------------------------------------------

        grad = self.conv1.conv1_backward(grad)

        return grad

    # ========================================================
    # PARAMETER COLLECTION
    # ========================================================

    def get_parameters(self):
        """
        Collect every trainable parameter in the network.

        All names are globally unique.

        These arrays are the ACTUAL arrays owned by the layers.
        AdamW therefore modifies the model parameters directly.
        """

        return {
            "conv1.filters": self.conv1.filters,
            "conv1.bias": self.conv1.bias,

            "conv2.filters": self.conv2.filters,
            "conv2.bias": self.conv2.bias,

            "conv3.filters": self.conv3.filters,
            "conv3.bias": self.conv3.bias,

            "dense128.W": self.dense128.W,
            "dense128.b": self.dense128.b,

            "dense4.W": self.dense4.W,
            "dense4.b": self.dense4.b,
        }

    # ========================================================
    # GRADIENT COLLECTION
    # ========================================================

    def get_gradients(self):
        """
        Collect gradients generated by each layer's backward().

        The names MUST exactly match get_parameters().
        """

        return {
            "conv1.filters": self.conv1.d_filters,
            "conv1.bias": self.conv1.d_bias,

            "conv2.filters": self.conv2.d_filters,
            "conv2.bias": self.conv2.d_bias,

            "conv3.filters": self.conv3.d_filters,
            "conv3.bias": self.conv3.d_bias,

            "dense128.W": self.dense128.d_W,
            "dense128.b": self.dense128.d_b,

            "dense4.W": self.dense4.d_W,
            "dense4.b": self.dense4.d_b,
        }

    # ========================================================
    # ONE TRAINING STEP
    # ========================================================

    def train_step(
        self,
        input_tensor: np.ndarray,
        target_class: int,
        dropout_seed: int | None = None,
    ):
        """
        Execute ONE complete training step.

        The complete process is:

            FORWARD
                ↓
            logits
                ↓
            softmax
                ↓
            cross entropy
                ↓
            dL/dlogits
                ↓
            BACKWARD
                ↓
            gradients
                ↓
            AdamW
                ↓
            UPDATED PARAMETERS
        """

        # ----------------------------------------------------
        # 1. FORWARD
        # ----------------------------------------------------

        logits = self.forward(
            input_tensor,
            training=True,
            dropout_seed=dropout_seed,
        )

        # ----------------------------------------------------
        # 2. LOSS + SOFTMAX
        # ----------------------------------------------------

        loss, probabilities, grad_logits = (
            self.compute_loss(
                logits,
                target_class
            )
        )

        # ----------------------------------------------------
        # 3. BACKWARD
        # ----------------------------------------------------

        self.backward(grad_logits)

        # ----------------------------------------------------
        # 4. COLLECT PARAMETERS
        # ----------------------------------------------------

        params = self.get_parameters()

        # ----------------------------------------------------
        # 5. COLLECT GRADIENTS
        # ----------------------------------------------------

        grads = self.get_gradients()

        # ----------------------------------------------------
        # 6. ADAMW UPDATE
        # ----------------------------------------------------

        self.optimizer.step(
            params,
            grads
        )

        # ----------------------------------------------------
        # RETURN TRAINING INFORMATION
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(logits)
        )

        accuracy = float(
            predicted_class == target_class
        )

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "target_class": target_class,
            "predicted_class": predicted_class,
            "accuracy": accuracy,
            "optimizer_step": self.optimizer.t,
        }


# ============================================================
# ADAMW
# ============================================================

class AdamW:

    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    ):

        self.lr = np.float32(learning_rate)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.epsilon = np.float32(epsilon)
        self.weight_decay = np.float32(weight_decay)

        # Each parameter gets its own moment estimates.
        self.m = {}
        self.v = {}

        # ONE global training-step counter.
        self.t = 0

    def step(
        self,
        params: dict,
        grads: dict,
    ):
        """
        Perform one AdamW update on every parameter.

        All updates happen IN PLACE.
        """

        missing = set(params) - set(grads)

        if missing:
            raise KeyError(
                f"No gradient provided for: {sorted(missing)}"
            )

        self.t += 1

        for name, param in params.items():

            grad = grads[name].astype(
                np.float32,
                copy=False
            )

            if name not in self.m:

                self.m[name] = np.zeros_like(
                    param,
                    dtype=np.float32
                )

                self.v[name] = np.zeros_like(
                    param,
                    dtype=np.float32
                )

            # ------------------------------------------------
            # FIRST MOMENT
            # ------------------------------------------------

            self.m[name] = (
                self.beta1 * self.m[name]
                + (1.0 - self.beta1) * grad
            )

            # ------------------------------------------------
            # SECOND MOMENT
            # ------------------------------------------------

            self.v[name] = (
                self.beta2 * self.v[name]
                + (1.0 - self.beta2)
                * (grad * grad)
            )

            # ------------------------------------------------
            # BIAS CORRECTION
            # ------------------------------------------------

            m_hat = (
                self.m[name]
                / (1.0 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[name]
                / (1.0 - self.beta2 ** self.t)
            )

            # ------------------------------------------------
            # DECOUPLED WEIGHT DECAY
            # ------------------------------------------------

            param -= (
                self.lr
                * self.weight_decay
                * param
            )

            # ------------------------------------------------
            # ADAM GRADIENT UPDATE
            # ------------------------------------------------

            param -= (
                self.lr
                * m_hat
                / (
                    np.sqrt(v_hat)
                    + self.epsilon
                )
            )

    def state_dict(self):
        """Return optimizer state for checkpointing."""

        return {
            "t": self.t,
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state):
        """Restore optimizer state."""

        self.t = state["t"]
        self.m = state["m"]
        self.v = state["v"]


