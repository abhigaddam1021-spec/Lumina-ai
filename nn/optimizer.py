"""
optimizer.py — Weight Update Algorithms
=========================================

WHAT IS AN OPTIMIZER?
----------------------
After backpropagation gives us gradients (dLoss/dW for every weight), we
need a RULE for HOW to update the weights. That rule is the optimizer.

The simplest rule: move each weight in the direction that reduces loss.
  W_new = W_old - learning_rate * dLoss/dW

But simple SGD has problems (slow, gets stuck). Modern optimizers use
tricks like momentum and adaptive learning rates to train faster and better.

LEARNING RATE (lr) — THE MOST IMPORTANT HYPERPARAMETER
---------------------------------------------------------
  Too large: updates overshoot the minimum, loss bounces or explodes
  Too small: training is painfully slow, may get stuck in local minima
  Just right: smooth, steady descent toward the minimum

We implement three optimizers, from simple to state-of-the-art:
  1. SGD          — classic Stochastic Gradient Descent
  2. SGD+Momentum — SGD with velocity to "roll through" shallow valleys
  3. Adam         — adaptive learning rate + momentum (default)
"""

import numpy as np
from .layer import Layer


# -----------------------------------------------------------------------------
# SGD — Stochastic Gradient Descent
# -----------------------------------------------------------------------------
# The simplest optimizer. For each weight:
#   W = W - lr * dW
#
# "Stochastic" because we compute gradients on mini-batches (random subsets)
# of data, not the entire dataset. This adds noise but is much faster.
# -----------------------------------------------------------------------------

class SGD:
    """Plain Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01):
        """
        Parameters
        ----------
        learning_rate : float
            Step size for each weight update. Typical range: 0.0001 – 0.1.
        """
        self.lr = learning_rate

    def update(self, layers: list[Layer]) -> None:
        """Update every layer's weights and biases using computed gradients."""
        for layer in layers:
            # Move weights opposite to the gradient (downhill on the loss surface)
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


# -----------------------------------------------------------------------------
# SGD with Momentum
# -----------------------------------------------------------------------------
# Adds a "velocity" term — like a ball rolling downhill:
#   velocity = beta * velocity + (1 - beta) * gradient
#   W = W - lr * velocity
#
# The velocity accumulates past gradients, so the ball speeds up in consistent
# directions and slows down when gradients flip sign. Escapes shallow valleys.
#
# beta ≈ 0.9 means velocity = 90% old + 10% new gradient.
# -----------------------------------------------------------------------------

class SGDMomentum:
    """SGD with Momentum — faster convergence than plain SGD."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Parameters
        ----------
        learning_rate : float
            Base step size.
        momentum : float
            How much of the previous velocity to retain (0–1). Typical: 0.9.
        """
        self.lr = learning_rate
        self.beta = momentum
        # Velocity dictionaries keyed by layer id
        self._vW: dict[int, np.ndarray] = {}
        self._vb: dict[int, np.ndarray] = {}

    def update(self, layers: list[Layer]) -> None:
        for layer in layers:
            lid = layer.layer_id

            # Initialize velocity to zeros on first call for this layer
            if lid not in self._vW:
                self._vW[lid] = np.zeros_like(layer.W)
                self._vb[lid] = np.zeros_like(layer.b)

            # Update velocity: blend old velocity with new gradient
            self._vW[lid] = self.beta * self._vW[lid] + (1 - self.beta) * layer.dW
            self._vb[lid] = self.beta * self._vb[lid] + (1 - self.beta) * layer.db

            # Update weights using the smoothed velocity
            layer.W -= self.lr * self._vW[lid]
            layer.b -= self.lr * self._vb[lid]


# -----------------------------------------------------------------------------
# Adam — Adaptive Moment Estimation
# -----------------------------------------------------------------------------
# The standard optimizer for deep learning (used by most state-of-the-art models).
# Combines two ideas:
#
#   1st moment (m): Exponential moving average of gradients (like momentum)
#   2nd moment (v): Exponential moving average of SQUARED gradients
#                   → measures how "noisy" each weight's gradient is
#
# Update rule:
#   m = beta1 * m + (1 - beta1) * g          ← smoothed gradient
#   v = beta2 * v + (1 - beta2) * g^2        ← smoothed gradient variance
#   m_hat = m / (1 - beta1^t)                ← bias correction (important early on)
#   v_hat = v / (1 - beta2^t)
#   W = W - lr * m_hat / (sqrt(v_hat) + eps)
#
# WHY DOES THIS WORK? Weights with large, noisy gradients get SMALLER steps
# (v is large → denominator is large). Weights with small, consistent gradients
# get BIGGER steps. Each weight gets its own effective learning rate!
#
# Typical hyperparameters: lr=0.001, beta1=0.9, beta2=0.999
# -----------------------------------------------------------------------------

class Adam:
    """Adam optimizer — adaptive learning rates + momentum. Best default choice."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Global step size. Typical: 0.001.
        beta1 : float
            Decay rate for 1st moment (gradient moving average). Typical: 0.9.
        beta2 : float
            Decay rate for 2nd moment (squared gradient moving average). Typical: 0.999.
        epsilon : float
            Small constant to prevent division by zero. Typical: 1e-8.
        """
        self.lr    = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = epsilon
        self.t     = 0          # timestep (for bias correction)

        # 1st and 2nd moment estimates, keyed by layer id
        self._mW: dict[int, np.ndarray] = {}
        self._vW: dict[int, np.ndarray] = {}
        self._mb: dict[int, np.ndarray] = {}
        self._vb: dict[int, np.ndarray] = {}

    def update(self, layers: list[Layer]) -> None:
        """Apply one Adam step to all layers."""
        self.t += 1  # increment global timestep

        # Bias correction factors (important in early timesteps where
        # moments are still warming up from 0)
        bc1 = 1 - self.beta1 ** self.t   # gets closer to 1 as t grows
        bc2 = 1 - self.beta2 ** self.t

        for layer in layers:
            lid = layer.layer_id

            # Initialize moments to zero on first encounter
            if lid not in self._mW:
                self._mW[lid] = np.zeros_like(layer.W)
                self._vW[lid] = np.zeros_like(layer.W)
                self._mb[lid] = np.zeros_like(layer.b)
                self._vb[lid] = np.zeros_like(layer.b)

            # -- Weights ----------------------------------------------------
            # Update 1st moment (smoothed gradient)
            self._mW[lid] = self.beta1 * self._mW[lid] + (1 - self.beta1) * layer.dW
            # Update 2nd moment (smoothed squared gradient)
            self._vW[lid] = self.beta2 * self._vW[lid] + (1 - self.beta2) * layer.dW**2

            # Bias-corrected moments
            m_hat_W = self._mW[lid] / bc1
            v_hat_W = self._vW[lid] / bc2

            # Update: step size shrinks when gradient variance is high
            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)

            # -- Biases -----------------------------------------------------
            self._mb[lid] = self.beta1 * self._mb[lid] + (1 - self.beta1) * layer.db
            self._vb[lid] = self.beta2 * self._vb[lid] + (1 - self.beta2) * layer.db**2

            m_hat_b = self._mb[lid] / bc1
            v_hat_b = self._vb[lid] / bc2

            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


# -----------------------------------------------------------------------------
# Optimizer Registry
# -----------------------------------------------------------------------------

def get_optimizer(name: str, **kwargs):
    """
    Factory function. Returns an optimizer instance by name.
    
    Usage: get_optimizer("adam", learning_rate=0.001)
    """
    registry = {
        "sgd":          SGD,
        "sgd_momentum": SGDMomentum,
        "adam":         Adam,
    }
    if name not in registry:
        raise KeyError(f"Unknown optimizer '{name}'. Choose from: {list(registry.keys())}")
    return registry[name](**kwargs)
