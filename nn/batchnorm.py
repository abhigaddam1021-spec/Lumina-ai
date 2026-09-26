"""
batchnorm.py -- Batch Normalization Layer
=========================================

WHAT IS BATCH NORMALIZATION?
----------------------------
Invented in 2015 by Ioffe and Szegedy, Batch Normalization (BatchNorm) is arguably 
the most important breakthrough for training deep neural networks. 

THE PROBLEM IT SOLVES:
When a network has 150 layers, the input to layer 150 depends on the outputs of 
the 149 layers before it. As weights update during training, the scale and distribution 
of the activations keep shifting wildly (called "Internal Covariate Shift"). 
This causes gradients to explode into infinity or vanish into zero (the TV static problem).

HOW IT WORKS:
During training, BatchNorm looks at the current "mini-batch" of data passing through it:
  1. It calculates the mean (average) and variance (spread) of the batch.
  2. It normalizes the data to have a mean of 0 and a variance of 1.
  3. It scales and shifts the normalized data using two LEARNABLE parameters:
     - Gamma (scale)
     - Beta (shift)
     (We store these as self.W and self.b so our Optimizer can update them automatically!)

During this process, it keeps a "running average" of the mean and variance.
During inference (testing/prediction), we don't calculate the batch mean (because we 
might only be predicting one single sample). Instead, we use the running averages.
"""

import numpy as np


class BatchNorm:
    """
    Batch Normalization layer.
    Normalizes the activations of the previous layer to stabilize training.
    """

    def __init__(self, size: int, layer_id: int, momentum: float = 0.9, epsilon: float = 1e-5):
        self.size = size
        self.layer_id = f"bn_{layer_id}"
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = True

        # Learnable parameters
        # We name them W (gamma) and b (beta) so the existing Optimizer works perfectly!
        self.W = np.ones((1, size), dtype=np.float32)   # Gamma (Scale)
        self.b = np.zeros((1, size), dtype=np.float32)  # Beta (Shift)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Running statistics for inference mode
        self.running_mean = np.zeros((1, size), dtype=np.float32)
        self.running_var = np.ones((1, size), dtype=np.float32)

        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the batch during training, or apply running stats during inference.
        """
        if self.training:
            # 1. Calculate batch mean and variance
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            # 2. Update running stats for inference
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var

            # 3. Normalize
            x_norm = (x - mean) / np.sqrt(var + self.epsilon)
            
            # Cache for backward pass
            self._cache = (x, x_norm, mean, var)
        else:
            # Inference mode: use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # 4. Scale and shift (gamma * x_norm + beta)
        return self.W * x_norm + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Complex calculus to flow gradients backward through the normalization process.
        """
        if not self.training or self._cache is None:
            return grad_output

        x, x_norm, mean, var = self._cache
        N = x.shape[0]

        # 1. Gradients for learnable parameters gamma (W) and beta (b)
        self.dW = np.sum(grad_output * x_norm, axis=0, keepdims=True)
        self.db = np.sum(grad_output, axis=0, keepdims=True)

        # 2. Gradient flowing back to the previous layer
        grad_x_norm = grad_output * self.W
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        
        # Highly optimized analytical derivative for BatchNorm
        dx = (1.0 / N) * std_inv * (
            N * grad_x_norm - np.sum(grad_x_norm, axis=0) - x_norm * np.sum(grad_x_norm * x_norm, axis=0)
        )
        return dx

    def set_training(self, training: bool) -> None:
        self.training = training

    def param_count(self) -> int:
        return self.size * 2
