"""
dropout.py -- Dropout Regularization Layer
============================================

WHAT IS DROPOUT?
-----------------
Dropout is one of the most powerful and widely-used techniques to prevent
OVERFITTING in neural networks. It was introduced by Hinton et al. (2012).

OVERFITTING RECAP:
  The network memorizes training data instead of learning general patterns.
  It gets 99% accuracy on training data but only 60% on new data.
  Dropout forces the network to learn more ROBUST, REDUNDANT features.

HOW DROPOUT WORKS:
  During TRAINING, at each forward pass:
    - Randomly "drop" (zero out) each neuron with probability `p`.
    - Scale surviving neurons by 1/(1-p) to keep the expected output the same.
      (This is called "inverted dropout" -- the modern standard.)

  During INFERENCE (prediction), dropout is turned OFF completely.
    - All neurons are active.
    - No scaling needed (already handled during training).

WHY DOES RANDOMLY ZEROING NEURONS HELP?
-----------------------------------------
Think of it like training a team of specialists:

  WITHOUT dropout: Each neuron can always rely on its neighbors. Neurons
  start to co-adapt -- neuron A always corrects for neuron B's mistakes.
  They become fragile: remove one and the whole thing breaks.

  WITH dropout: Every neuron knows it might be dropped. So each one MUST
  learn a useful, independent feature on its own. The network ends up
  with many overlapping representations -- much harder to overfit.

ANALOGY: A basketball team where random players sit out each practice.
Every player must learn ALL positions, not just their specialty. The team
becomes more versatile and robust.

TYPICAL DROPOUT RATES:
  - Hidden layers: p = 0.1 to 0.5 (drop 10-50% of neurons)
  - Input layer:   p = 0.1 to 0.2 (be gentle with raw features)
  - Output layer:  p = 0         (NEVER drop output neurons!)

  In our 150-layer network we use p = 0.1 (light dropout) because
  very deep networks with heavy dropout can struggle to learn anything.

THE MATH:
  Forward (train):
    mask = random(shape) > p        --> binary mask (0 or 1)
    output = input * mask / (1 - p) --> apply and scale

  Backward (train):
    grad_input = grad_output * mask / (1 - p)  --> same mask, same scale

  Forward (inference):
    output = input   --> pass through unchanged (all neurons active)
"""

import numpy as np


class Dropout:
    """
    Dropout regularization layer.

    This is NOT a layer with learnable weights. It is a "pass-through"
    layer that only does something during training.

    Parameters
    ----------
    rate : float
        Fraction of neurons to DROP during training.
        rate=0.1 means 10% of neurons are zeroed each forward pass.
        rate=0.0 means no dropout (disabled).
    layer_id : int
        Human-readable identifier for this dropout position.
    """

    def __init__(self, rate: float = 0.1, layer_id: int = 0):
        if not (0.0 <= rate < 1.0):
            raise ValueError(f"Dropout rate must be in [0, 1). Got {rate}.")
        self.rate     = rate          # fraction to DROP
        self.keep     = 1.0 - rate   # fraction to KEEP
        self.layer_id = layer_id
        self.training = True          # flip to False during inference

        # The mask is saved during forward pass for use in backward pass
        self._mask: np.ndarray | None = None

    # ---- Forward Pass -------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout during training; pass through unchanged during inference.

        Parameters
        ----------
        x : np.ndarray, shape [batch_size, features]
            Input activations from the previous layer.

        Returns
        -------
        np.ndarray, same shape as x.
        """
        if not self.training or self.rate == 0.0:
            # INFERENCE MODE: no dropout, full network active
            self._mask = None
            return x

        # TRAINING MODE: generate random binary mask
        # np.random.rand gives values in [0, 1). Values > rate are KEPT (1),
        # values <= rate are DROPPED (0).
        self._mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)

        # Apply mask and scale by 1/keep so the expected sum stays constant.
        # Without scaling: if 50% neurons are dropped, outputs are halved,
        # which would mismatch inference (where all neurons are active).
        return x * self._mask / self.keep

    # ---- Backward Pass ------------------------------------------------------

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Pass gradients back through dropout -- same mask, same scaling.

        Neurons that were dropped (mask=0) get ZERO gradient -- they
        didn't contribute to the output, so they don't get updated.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient from the next layer.

        Returns
        -------
        np.ndarray
            Gradient for the previous layer.
        """
        if self._mask is None:
            # Inference mode -- pass gradient unchanged
            return grad_output

        return grad_output * self._mask / self.keep

    def set_training(self, training: bool) -> None:
        """Switch between training (dropout active) and inference (dropout off)."""
        self.training = training

    def param_count(self) -> int:
        """Dropout has NO learnable parameters."""
        return 0

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate}, layer_id={self.layer_id})"
