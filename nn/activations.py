"""
activations.py — Activation Functions
======================================

WHY DO WE NEED ACTIVATION FUNCTIONS?
--------------------------------------
A neural network without activations is just matrix multiplication stacked
on top of more matrix multiplication. No matter how many layers you add,
it collapses to a single linear transformation — useless for learning
complex patterns.

Activation functions introduce NON-LINEARITY, which is what lets a deep
network approximate ANY function (this is the Universal Approximation Theorem).

Each activation here gives:
  forward(x)  → the output value (used during forward pass)
  backward(x) → the DERIVATIVE (used during backpropagation to compute gradients)

LEARNING TIP: The derivative tells the network "how sensitive is the output
to a tiny change in the input?". If the derivative is 0, no learning happens
(this is called the "vanishing gradient" problem — a key challenge in deep nets).
"""

import numpy as np


# -----------------------------------------------------------------------------
# ReLU — Rectified Linear Unit
# -----------------------------------------------------------------------------
# Formula:   f(x) = max(0, x)
# Derivative: 1 if x > 0, else 0
#
# WHY USE IT? Simple, fast, doesn't saturate for positive values.
# Used in most modern networks. Default for hidden layers here.
# PROBLEM: "Dying ReLU" — if a neuron always gets negative input, its
# gradient is always 0 and it never updates ("dies").
# -----------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """Forward pass: set all negative values to zero."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Backward pass: gradient is 1 where x > 0, else 0."""
    return (x > 0).astype(float)


# -----------------------------------------------------------------------------
# Leaky ReLU
# -----------------------------------------------------------------------------
# Formula:   f(x) = x if x > 0, else alpha * x  (alpha is a small number like 0.01)
# Derivative: 1 if x > 0, else alpha
#
# WHY USE IT? Fixes the "dying ReLU" problem — negative inputs still get a
# tiny gradient (alpha * x) so the neuron never truly dies.
# -----------------------------------------------------------------------------

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Forward pass: small negative slope for x < 0."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Backward pass: 1 for positive, alpha for negative."""
    return np.where(x > 0, 1.0, alpha)


# -----------------------------------------------------------------------------
# Sigmoid
# -----------------------------------------------------------------------------
# Formula:   f(x) = 1 / (1 + e^(-x))    → squashes output to (0, 1)
# Derivative: f(x) * (1 - f(x))
#
# WHY USE IT? Great for OUTPUT layers doing binary classification (is it cat
# or dog?). Outputs a probability between 0 and 1.
# PROBLEM: Saturates at extremes — gradients become nearly 0 for very
# large/small inputs → vanishing gradient problem in deep networks.
# -----------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Forward pass: map any real number to (0, 1)."""
    # Clip to prevent overflow in exp for very negative values
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Backward pass: uses the fact that sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


# -----------------------------------------------------------------------------
# Tanh — Hyperbolic Tangent
# -----------------------------------------------------------------------------
# Formula:   f(x) = (e^x - e^(-x)) / (e^x + e^(-x))  → output range (-1, 1)
# Derivative: 1 - f(x)^2
#
# WHY USE IT? Like sigmoid but centered at 0 → gradients flow better.
# Often used in RNNs. Same saturation problem as sigmoid.
# -----------------------------------------------------------------------------

def tanh(x: np.ndarray) -> np.ndarray:
    """Forward pass: map any real number to (-1, 1)."""
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Backward pass: 1 - tanh(x)^2."""
    return 1.0 - np.tanh(x) ** 2


# -----------------------------------------------------------------------------
# Linear (Identity)
# -----------------------------------------------------------------------------
# Formula:   f(x) = x
# Derivative: 1
#
# WHY USE IT? Used on the OUTPUT layer for regression tasks where the
# output can be any real number (not just 0-1).
# -----------------------------------------------------------------------------

def linear(x: np.ndarray) -> np.ndarray:
    """Forward pass: pass through unchanged."""
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    """Backward pass: gradient is always 1."""
    return np.ones_like(x)


# -----------------------------------------------------------------------------
# Softmax
# -----------------------------------------------------------------------------
# Formula:   f(x_i) = e^(x_i) / sum(e^(x_j) for all j)
#
# WHY USE IT? Converts a vector of raw scores into PROBABILITIES that sum to 1.
# Used on the output layer for multi-class classification.
# We handle its derivative inside the loss function (combined with cross-entropy
# it simplifies beautifully to: predicted - actual).
# -----------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Forward pass: convert logits to probabilities. Subtract max for numerical stability."""
    # Subtracting max(x) prevents e^x from exploding to infinity
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# Activation Registry — maps string names to (forward_fn, backward_fn) pairs
# -----------------------------------------------------------------------------
# This lets us choose activations by name (e.g., "relu") instead of hardcoding.

ACTIVATIONS: dict[str, tuple] = {
    "relu":       (relu,       relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "sigmoid":    (sigmoid,    sigmoid_derivative),
    "tanh":       (tanh,       tanh_derivative),
    "linear":     (linear,     linear_derivative),
}

def get_activation(name: str) -> tuple:
    """
    Return (forward_fn, backward_fn) for the named activation.
    Raises KeyError with a helpful message if the name is unknown.
    """
    if name not in ACTIVATIONS:
        available = list(ACTIVATIONS.keys())
        raise KeyError(f"Unknown activation '{name}'. Choose from: {available}")
    return ACTIVATIONS[name]
