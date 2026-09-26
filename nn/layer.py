"""
layer.py — A Single Fully-Connected (Dense) Layer
===================================================

WHAT IS A LAYER?
-----------------
One layer does exactly two things:

  1. LINEAR TRANSFORM:   z = x @ W + b
       x = input  (shape: [batch_size, input_size])
       W = weights (shape: [input_size, output_size])
       b = biases  (shape: [1, output_size])
       z = pre-activation output (shape: [batch_size, output_size])

  2. ACTIVATION:         a = f(z)
       f = activation function (ReLU, sigmoid, etc.)
       a = activated output — this is passed to the NEXT layer

WHAT ARE WEIGHTS AND BIASES?
------------------------------
  - WEIGHTS (W): Each weight is a "connection strength" between a neuron
    in this layer and a neuron in the previous layer. The network LEARNS
    by adjusting these numbers during training.
  
  - BIASES (b): A bias lets a neuron "shift" its activation threshold.
    Without bias, all neurons would produce 0 output when input is 0.
    Bias gives the neuron a head start.

WEIGHT INITIALIZATION — WHY IT MATTERS
----------------------------------------
Starting weights at 0 is a disaster — every neuron computes the same thing
and learns the same thing (the "symmetry problem"). We must break symmetry.

We use "He initialization" (for ReLU) and "Xavier initialization" (for tanh/sigmoid):
  - He:     W ~ N(0, sqrt(2 / input_size))    ← scales for ReLU's half-zeroing
  - Xavier: W ~ N(0, sqrt(1 / input_size))    ← scales for tanh/sigmoid's range

BACKPROPAGATION THROUGH A LAYER
---------------------------------
During backward pass, we receive `grad_output` (dLoss/d_a from the NEXT layer).
We need to compute:
  - dLoss/dW  → how much to update our weights
  - dLoss/db  → how much to update our biases
  - dLoss/d_x → gradient to pass BACK to the PREVIOUS layer

Chain rule:
  dLoss/dz  = grad_output * f'(z)    [element-wise: activation derivative]
  dLoss/dW  = x.T @ dLoss/dz        [how much each weight contributed to loss]
  dLoss/db  = sum(dLoss/dz, axis=0) [sum over the batch]
  dLoss/d_x = dLoss/dz @ W.T        [pass gradient backward]
"""

from typing import Optional

import numpy as np
from .activations import get_activation


class Layer:
    """
    A single dense (fully-connected) layer with learnable weights and biases.

    Parameters
    ----------
    input_size : int
        Number of neurons feeding INTO this layer.
    output_size : int
        Number of neurons in THIS layer (its output dimension).
    activation : str
        Name of the activation function. Default is 'relu'.
    layer_id : int
        Human-readable index (e.g., layer 1, 2, ... 150).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        layer_id: int = 0,
    ):
        self.input_size  = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.layer_id = layer_id

        # Get the forward and derivative functions for this activation
        self.act_fn, self.act_deriv = get_activation(activation)

        # -- Weight Initialization ------------------------------------------
        # He initialization for ReLU-family, Xavier for others.
        # The "scale" factor prevents gradients from exploding or vanishing
        # right at the start of training.
        if activation in ("relu", "leaky_relu"):
            scale = np.sqrt(2.0 / input_size)   # He
        else:
            scale = np.sqrt(1.0 / input_size)   # Xavier

        # W shape: [input_size, output_size]
        # Each column is the weight vector for one output neuron.
        self.W: np.ndarray = np.random.randn(input_size, output_size) * scale

        # b shape: [1, output_size] — one bias per output neuron
        # Initialize to zero (bias symmetry is not a problem like weight symmetry)
        self.b: np.ndarray = np.zeros((1, output_size))

        # -- Gradient Storage ----------------------------------------------
        # These get populated during backward pass and consumed by the optimizer.
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)

        # -- Cache for Backprop --------------------------------------------
        # During forward pass, we save intermediate values we'll need later
        # during the backward pass (the chain rule needs them).
        self._input_cache: Optional[np.ndarray] = None  # x (input to this layer)
        self._z_cache:     Optional[np.ndarray] = None  # z = x @ W + b (pre-activation)

    # --- Forward Pass --------------------------------------------------------

    def forward(self, x: np.ndarray, apply_activation: bool = True) -> np.ndarray:
        """
        Compute the layer's output given input x.
        """
        self._input_cache = x
        self._applied_activation = apply_activation

        z = x @ self.W + self.b
        self._z_cache = z

        if apply_activation:
            return self.act_fn(z)
        return z

    # --- Backward Pass -------------------------------------------------------

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients for weights, biases, and input.

        This is where the CHAIN RULE is applied:

          dLoss/dz  = grad_output * f'(z)   ← element-wise multiplication
          dLoss/dW  = x.T  @  dLoss/dz
          dLoss/db  = sum(dLoss/dz, axis=0, keepdims=True)
          dLoss/dx  = dLoss/dz  @  W.T      ← this goes to the previous layer

        Parameters
        ----------
        grad_output : np.ndarray, shape [batch_size, output_size]
            The gradient of the loss with respect to this layer's OUTPUT (a).
            Passed backward from the next layer.

        Returns
        -------
        np.ndarray, shape [batch_size, input_size]
            The gradient of the loss with respect to this layer's INPUT (x).
            This is passed to the PREVIOUS layer's backward().
        """
        # -- Guard: ensure forward() has been called ------------------------
        if self._input_cache is None or self._z_cache is None:
            raise RuntimeError(
                f"Layer {self.layer_id}: backward() called before forward(). "
                "Run a forward pass first."
            )

        # -- Step 1: activation derivative ----------------------------------
        # We need dLoss/dz.  By chain rule:
        #   dLoss/dz = dLoss/da * da/dz = grad_output * f'(z)
        if getattr(self, '_applied_activation', True):
            dz = grad_output * self.act_deriv(self._z_cache)
        else:
            dz = grad_output

        batch_size = dz.shape[0]

        # -- Step 2: gradient for weights -----------------------------------
        # dLoss/dW = x.T @ dz
        # We average over the batch (divide by batch_size) to keep updates stable.
        self.dW = self._input_cache.T @ dz / batch_size

        # -- Step 3: gradient for biases ------------------------------------
        # Sum across the batch (each sample contributes to the bias gradient)
        self.db = np.sum(dz, axis=0, keepdims=True) / batch_size

        # -- Step 4: gradient to pass to the previous layer -----------------
        # dLoss/dx = dz @ W.T
        grad_input = dz @ self.W.T
        return grad_input

    # --- Utility -------------------------------------------------------------

    def param_count(self) -> int:
        """Return total number of learnable parameters in this layer."""
        # Each weight + each bias = input_size * output_size + output_size
        return self.W.size + self.b.size

    def __repr__(self) -> str:
        return (
            f"Layer(id={self.layer_id}, "
            f"{self.input_size}→{self.output_size}, "
            f"act={self.activation_name}, "
            f"params={self.param_count():,})"
        )
