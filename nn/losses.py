"""
losses.py — Loss Functions
===========================

WHAT IS A LOSS FUNCTION?
--------------------------
The loss (also called "cost" or "error") is a SINGLE NUMBER that measures
how wrong the network's predictions are. The goal of training is to make
this number as small as possible.

Training loop summary:
  1. Forward pass  → get predictions
  2. Compute loss  → measure how wrong we are
  3. Backward pass → compute gradients (direction to improve)
  4. Update weights → move weights in the direction that reduces loss

WHY DIFFERENT LOSS FUNCTIONS?
-------------------------------
  - Binary cross-entropy → binary classification (yes/no)
  - Categorical cross-entropy → multi-class classification (cat/dog/bird)
  - Mean Squared Error (MSE) → regression (predict a number)
  - Mean Absolute Error (MAE) → regression (more robust to outliers)

Each function returns:
  loss_value : float   — the scalar loss (for monitoring training)
  grad       : ndarray — dLoss/d_predictions (to start backpropagation)
"""

import numpy as np


# -----------------------------------------------------------------------------
# Binary Cross-Entropy (BCE)
# -----------------------------------------------------------------------------
# Used when: output is ONE neuron with sigmoid activation (binary: 0 or 1)
#
# Formula:  L = -mean( y*log(p) + (1-y)*log(1-p) )
#   y = true label (0 or 1)
#   p = predicted probability (0 to 1, from sigmoid)
#
# INTUITION: If y=1 and p≈1, log(1)≈0 → loss near 0 ✓
#            If y=1 and p≈0, log(0)→-∞ → huge penalty ✗
#
# Gradient: dL/dp = (p - y) / (p*(1-p)) * batch_size
#   But combined with sigmoid output, simplifies to: (p - y)
# -----------------------------------------------------------------------------

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute binary cross-entropy loss and its gradient.

    Parameters
    ----------
    y_true : np.ndarray, shape [batch_size, 1]
        Ground truth labels, values are 0 or 1.
    y_pred : np.ndarray, shape [batch_size, 1]
        Predicted probabilities from sigmoid output layer (values in 0–1).

    Returns
    -------
    loss : float
        Scalar loss value.
    grad : np.ndarray, shape [batch_size, 1]
        Gradient of loss w.r.t. y_pred.
    """
    # Clip predictions to avoid log(0) = -infinity
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)

    # Loss: average negative log-likelihood over the batch
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    # Gradient: dL/dp (combined with sigmoid derivative, simplifies to p - y)
    batch_size = y_true.shape[0]
    grad = (p - y_true) / batch_size
    return float(loss), grad


# -----------------------------------------------------------------------------
# Categorical Cross-Entropy (with Softmax)
# -----------------------------------------------------------------------------
# Used when: output is N neurons with softmax (predict one of N classes)
#
# Formula:  L = -mean( sum_j( y_j * log(p_j) ) )
#   y = one-hot vector (e.g. [0,1,0] for class 1 of 3)
#   p = softmax probabilities
#
# Gradient (softmax + cross-entropy combined): p - y
#   This beautiful simplification is why we pair softmax with cross-entropy!
# -----------------------------------------------------------------------------

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute categorical cross-entropy loss with softmax output.

    Parameters
    ----------
    y_true : np.ndarray, shape [batch_size, num_classes]
        One-hot encoded ground truth labels.
    y_pred : np.ndarray, shape [batch_size, num_classes]
        Softmax probabilities from the output layer.

    Returns
    -------
    loss : float
        Scalar loss value.
    grad : np.ndarray, shape [batch_size, num_classes]
        Combined softmax+cross-entropy gradient (p - y) / batch_size.
    """
    eps = 1e-12
    p = np.clip(y_pred, eps, 1.0)

    # Loss: for each sample, only the true class's log-probability matters
    loss = -np.mean(np.sum(y_true * np.log(p), axis=1))

    # Combined gradient (softmax + cross-entropy): p - y
    batch_size = y_true.shape[0]
    grad = (p - y_true) / batch_size
    return float(loss), grad


# -----------------------------------------------------------------------------
# Mean Squared Error (MSE)
# -----------------------------------------------------------------------------
# Used when: predicting a continuous number (regression)
#
# Formula:  L = mean( (y - p)^2 )
# Gradient: dL/dp = 2*(p - y) / batch_size  → simplified to (p-y)/batch_size
#
# INTUITION: Squaring the error punishes large mistakes much more than small
# ones (a prediction 10 away is penalized 100x more than one 1 away).
# -----------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute Mean Squared Error loss and gradient.

    Parameters
    ----------
    y_true : np.ndarray — ground truth values.
    y_pred : np.ndarray — predicted values from the output layer.

    Returns
    -------
    loss : float
    grad : np.ndarray
    """
    diff = y_pred - y_true
    loss = float(np.mean(diff ** 2))

    batch_size = y_true.shape[0]
    grad = 2 * diff / batch_size
    return loss, grad


# -----------------------------------------------------------------------------
# Loss Registry
# -----------------------------------------------------------------------------

LOSSES: dict[str, callable] = {
    "bce":   binary_cross_entropy,
    "cce":   categorical_cross_entropy,
    "mse":   mse,
}

def get_loss(name: str) -> callable:
    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Choose from: {list(LOSSES.keys())}")
    return LOSSES[name]
