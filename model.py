"""
model.py

This is the missing "glue" layer between your stateless from-scratch
blocks (conv1.py, conv2.py, conv3.py, dense128.py, dense4.py, silu1.py,
silu2.py, maxpool.py, gap.py, inverted_dropout.py) and trainflow.py.

WHAT WAS WRONG IN THE ORIGINAL trainflow.py
--------------------------------------------------------------------------
1. trainflow.py's TrainFlow.forward()/backward() called things like
   `self.conv1.conv1_forward(x)` and `self.conv1.filters` — i.e. it
   assumed `conv1` was an OBJECT that owns its own weights and caches.
   But conv1.py/conv2.py/conv3.py/dense128.py/dense4.py only expose
   plain functions (`conv1_forward(input, filters, biases)`), they own
   nothing. There was no class anywhere that actually created those
   objects. This file adds that missing class layer (Conv1Layer,
   Conv2Layer, Conv3Layer, Dense128Layer, Dense4Layer).

2. silu1, maxpool and silu2 were never called at all. The forward pass
   went conv1 -> conv2 -> conv3 directly, skipping three layers of the
   architecture described in your own docstrings
   (conv1 -> silu1 -> maxpool -> conv2 -> silu2 -> conv3 -> gap -> ...).

3. gap.py explicitly documents that it expects channel-FIRST input
   (C, H, W), but conv3's output is channel-LAST (H, W, C). trainflow.py
   never transposed between them, so gap_forward would have received
   the wrong axis order (h,w confused with channels) and produced
   garbage, silently (wrong shape errors, or worse, wrong numbers if
   H==W==C never held).

4. trainflow.py imported `gap_forward`/`gap_backward` from gap.py, but
   then *redefined* two placeholder functions with the same names lower
   down in the same file. In Python, that later definition silently
   shadows the import — so the real gap.py code was never actually
   being used, and gap_backward() would have thrown
   NotImplementedError the first time backward() ran.

5. inverted_dropout.py's real `forward()` signature is
   `forward(input_tensor, dropout_rate, training, rng) -> (output, mask)`
   (a tuple). trainflow.py called it as
   `forward(x, dropout_rate=..., training=..., seed=dropout_seed)` and
   expected a **dict** back (`cache["output"]`, `cache["mask"]`,
   `cache["keep_probability"]`) — none of which exist. This would have
   raised a TypeError on the very first forward pass.

6. dense4_backward() needs the `cache` dict returned by dense4_forward();
   trainflow.py called `self.dense4.dense4_backward(grad_logits)` with
   no cache at all.

This file fixes all six issues. trainflow.py itself is left untouched
(so you can diff against it) — TrainFlowFixed below is the corrected
version. Swap your training script to import `TrainFlowFixed` from here
instead of `TrainFlow` from trainflow.py.
"""

from __future__ import annotations

import numpy as np

import conv1 as _conv1
import conv2 as _conv2
import conv3 as _conv3
import silu1 as _silu1
import silu2 as _silu2
import maxpool as _maxpool          # kept only for initialize_*() helpers where used
import gap as _gap
import dense128 as _dense128
import dense4 as _dense4
import inverted_dropout as _dropout
import fast_layers as _fast          # vectorized conv1/conv2/conv3/maxpool — see fast_layers.py

from trainflow import AdamW, cross_entropy_loss, CLASS_NAMES, NUM_CLASSES  # reuse, unchanged


# ============================================================
# LAYER WRAPPERS
#
# Each wrapper OWNS its trainable parameters (so a checkpoint just has
# to save `.filters`/`.bias` or `.W`/`.b`), calls the real stateless
# function during forward(), stashes whatever cache backward() needs,
# and exposes d_filters/d_bias (or d_W/d_b) after backward() so
# TrainFlow.get_gradients() can find them by the same attribute names
# get_parameters() uses.
# ============================================================

class Conv1Layer:
    def __init__(self, filters: np.ndarray | None = None, bias: np.ndarray | None = None, seed: int = 42):
        self.filters = filters if filters is not None else _conv1.initialize_filters(seed)
        self.bias = bias if bias is not None else _conv1.initialize_biases()
        self._cache = None
        self.d_filters = None
        self.d_bias = None

    def conv1_forward(self, x: np.ndarray) -> np.ndarray:
        # Vectorized (fast_layers) — numerically validated against conv1.py,
        # see validate_fast_layers.py. Swap back to _conv1.conv1_forward if
        # you ever need the literal from-scratch loop version.
        out, cache = _fast.conv_forward(x, self.filters, self.bias)
        self._cache = cache
        return out

    def conv1_backward(self, d_out: np.ndarray) -> np.ndarray:
        d_input, d_filters, d_bias = _fast.conv_backward(d_out, self._cache)
        self.d_filters, self.d_bias = d_filters, d_bias
        return d_input


class Conv2Layer:
    def __init__(self, filters: np.ndarray | None = None, bias: np.ndarray | None = None, seed: int = 42):
        self.filters = filters if filters is not None else _conv2.initialize_filters(seed)
        self.bias = bias if bias is not None else _conv2.initialize_biases()
        self._cache = None
        self.d_filters = None
        self.d_bias = None

    def conv2_forward(self, x: np.ndarray) -> np.ndarray:
        out, cache = _fast.conv_forward(x, self.filters, self.bias)
        self._cache = cache
        return out

    def conv2_backward(self, d_out: np.ndarray) -> np.ndarray:
        d_input, d_filters, d_bias = _fast.conv_backward(d_out, self._cache)
        self.d_filters, self.d_bias = d_filters, d_bias
        return d_input


class Conv3Layer:
    def __init__(self, filters: np.ndarray | None = None, bias: np.ndarray | None = None, seed: int = 42):
        self.filters = filters if filters is not None else _conv3.initialize_filters(seed)
        self.bias = bias if bias is not None else _conv3.initialize_biases()
        self._cache = None
        self.d_filters = None
        self.d_bias = None

    def conv3_forward(self, x: np.ndarray) -> np.ndarray:
        out, cache = _fast.conv_forward(x, self.filters, self.bias)
        self._cache = cache
        return out

    def conv3_backward(self, d_out: np.ndarray) -> np.ndarray:
        d_input, d_filters, d_bias = _fast.conv_backward(d_out, self._cache)
        self.d_filters, self.d_bias = d_filters, d_bias
        return d_input


class Dense128Layer:
    def __init__(self, W: np.ndarray | None = None, b: np.ndarray | None = None, seed: int = 42):
        if W is None or b is None:
            init_W, init_b = _dense128.initialize_parameters(seed)
            W = W if W is not None else init_W
            b = b if b is not None else init_b
        self.W = W
        self.b = b
        self._input_cache = None
        self.d_W = None
        self.d_b = None

    def dense128_forward(self, x: np.ndarray) -> np.ndarray:
        z = _dense128.dense128_forward(x, self.W, self.b)
        self._input_cache = x
        return z

    def dense128_backward(self, d_out: np.ndarray) -> np.ndarray:
        # dense128.py's backward takes (upstream_grad, input, W) directly —
        # it doesn't use a cache dict the way the other blocks do.
        d_input, dW, db = _dense128.dense128_backward(d_out, self._input_cache, self.W)
        self.d_W, self.d_b = dW, db
        return d_input


class Dense4Layer:
    def __init__(self, W: np.ndarray | None = None, b: np.ndarray | None = None, seed: int = 42):
        if W is None or b is None:
            params = _dense4.initialize_dense4_parameters(seed)
            W = W if W is not None else params["weights"]
            b = b if b is not None else params["bias"]
        self.W = W
        self.b = b
        self._cache = None
        self.d_W = None
        self.d_b = None

    def dense4_forward(self, x: np.ndarray) -> np.ndarray:
        logits, cache = _dense4.dense4_forward(x, self.W, self.b)
        self._cache = cache
        return logits

    def dense4_backward(self, grad_logits: np.ndarray) -> np.ndarray:
        d_input, grads = _dense4.dense4_backward(grad_logits, self._cache)
        self.d_W, self.d_b = grads["weights"], grads["bias"]
        return d_input


# ============================================================
# CORRECTED TRAINFLOW
# ============================================================

class TrainFlowFixed:
    """
    Same public interface as trainflow.TrainFlow (forward, backward,
    compute_loss, get_parameters, get_gradients, train_step) but with
    the six bugs described at the top of this file fixed:

        conv1 -> silu1 -> maxpool -> conv2 -> silu2 -> conv3
              -> (transpose HWC->CHW) -> gap -> (128,)
              -> dense128 -> dropout -> dense4 -> logits
    """

    def __init__(
        self,
        conv1: Conv1Layer,
        conv2: Conv2Layer,
        conv3: Conv3Layer,
        dense128: Dense128Layer,
        dense4: Dense4Layer,
        dropout_rate: float = 0.40,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.dense128 = dense128
        self.dense4 = dense4
        self.dropout_rate = dropout_rate

        self.optimizer = AdamW(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        )

        self.cache = {}

    # ========================================================
    # FORWARD
    # ========================================================
    def forward(self, input_tensor: np.ndarray, training: bool = True, dropout_seed: int | None = None):
        x = self.conv1.conv1_forward(input_tensor)

        x, silu1_cache = _silu1.silu1_forward(x)
        self.cache["silu1"] = silu1_cache

        x, maxpool_cache = _fast.maxpool_forward(x)  # vectorized — see fast_layers.py
        self.cache["maxpool"] = maxpool_cache

        x = self.conv2.conv2_forward(x)

        x, silu2_cache = _silu2.silu2_forward(x)
        self.cache["silu2"] = silu2_cache

        x = self.conv3.conv3_forward(x)  # channel-last (H, W, 128)

        # gap.py expects channel-FIRST (C, H, W) — conv3 hands us
        # channel-last, so transpose before/after every GAP call.
        x_chw = np.transpose(x, (2, 0, 1))
        x, gap_cache = _gap.gap_forward(x_chw)
        self.cache["gap"] = gap_cache

        x = self.dense128.dense128_forward(x)

        rng = np.random.default_rng(dropout_seed) if dropout_seed is not None else None
        dropout_out, dropout_mask = _dropout.forward(
            x, dropout_rate=self.dropout_rate, training=training, rng=rng
        )
        self.cache["dropout_mask"] = dropout_mask

        logits = self.dense4.dense4_forward(dropout_out)
        return logits

    # ========================================================
    # LOSS
    # ========================================================
    def compute_loss(self, logits: np.ndarray, target_class: int):
        return cross_entropy_loss(logits, target_class)

    # ========================================================
    # BACKWARD
    # ========================================================
    def backward(self, grad_logits: np.ndarray):
        grad = self.dense4.dense4_backward(grad_logits)

        grad = _dropout.backward(grad, self.cache["dropout_mask"], dropout_rate=self.dropout_rate)

        grad = self.dense128.dense128_backward(grad)

        grad = _gap.gap_backward(grad, self.cache["gap"])       # (128, H, W)
        grad = np.transpose(grad, (1, 2, 0))                     # back to (H, W, 128)

        grad = self.conv3.conv3_backward(grad)

        grad = _silu2.silu2_backward(grad, self.cache["silu2"])

        grad = self.conv2.conv2_backward(grad)

        grad = _fast.maxpool_backward(grad, self.cache["maxpool"])

        grad = _silu1.silu1_backward(grad, self.cache["silu1"])

        grad = self.conv1.conv1_backward(grad)

        return grad

    # ========================================================
    # PARAMETER / GRADIENT COLLECTION
    # ========================================================
    def get_parameters(self):
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

    def get_gradients(self):
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
    def train_step(self, input_tensor: np.ndarray, target_class: int, dropout_seed: int | None = None):
        logits = self.forward(input_tensor, training=True, dropout_seed=dropout_seed)
        loss, probabilities, grad_logits = self.compute_loss(logits, target_class)
        self.backward(grad_logits)

        params = self.get_parameters()
        grads = self.get_gradients()
        self.optimizer.step(params, grads)

        predicted_class = int(np.argmax(logits))
        accuracy = float(predicted_class == target_class)

        return {
            "loss": float(loss),
            "logits": logits,
            "probabilities": probabilities,
            "target_class": target_class,
            "predicted_class": predicted_class,
            "accuracy": accuracy,
            "optimizer_step": self.optimizer.t,
        }

    def predict(self, input_tensor: np.ndarray):
        """Inference only — no dropout, no cache needed for backward."""
        logits = self.forward(input_tensor, training=False)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        return int(np.argmax(logits)), probs


# ============================================================
# FACTORY: build a fresh model, or one from checkpointed arrays
# ============================================================

def build_trainflow(state: dict | None = None, **trainflow_kwargs) -> TrainFlowFixed:
    """
    state: optional dict of numpy arrays (as produced by checkpoint.py's
    load_checkpoint) keyed like "conv1.filters", "conv1.bias", ...
    If None, every layer is freshly initialized (Xavier/He, seed=42).
    """
    state = state or {}

    conv1 = Conv1Layer(state.get("conv1.filters"), state.get("conv1.bias"))
    conv2 = Conv2Layer(state.get("conv2.filters"), state.get("conv2.bias"))
    conv3 = Conv3Layer(state.get("conv3.filters"), state.get("conv3.bias"))
    dense128 = Dense128Layer(state.get("dense128.W"), state.get("dense128.b"))
    dense4 = Dense4Layer(state.get("dense4.W"), state.get("dense4.b"))

    return TrainFlowFixed(conv1, conv2, conv3, dense128, dense4, **trainflow_kwargs)
