"""
network.py — The 150-Layer Neural Network
==========================================

HOW THE FULL NETWORK WORKS
----------------------------
We stack 150 Layer objects end-to-end. Data flows through them like this:

  Input (raw data)
    → Layer 1   (learns simple patterns: edges, basic features)
    → Layer 2   (combines layer 1 features into slightly more complex ones)
    → ...
    → Layer 50  (abstract mid-level representations)
    → ...
    → Layer 150 (highly abstract, task-specific representations)
    → Output layer (maps to prediction)

Each layer in early positions learns LOW-LEVEL features.
Deeper layers learn HIGH-LEVEL, ABSTRACT features.
This is the same principle as in image recognition networks where:
  Early layers → detect edges
  Mid layers   → detect shapes
  Deep layers  → detect faces / objects

THE MODES — HOW THEY DIFFER
-----------------------------
The "mode" changes WHICH LAYERS are active during a forward pass:

  [QUICK] QUICK (depth 30):      Only layers 1–30   — fast, shallow thinking
  [DATA] ANALYTIC (depth 90):   Layers 1–90        — mid-depth reasoning
  [BRAIN] DEEP THINK (depth 150): All 150 layers    — full depth, slowest
  💻 CODE (depth 120):       Layers 1–120      — deep, precise
  🎨 CREATIVE (depth 60):    Layers 1–60       — mid-range, varied
  🔬 RESEARCH (depth 90):    Layers 1–90       — analytic
  🐛 DEBUG (depth 120):      Layers 1–120      — systematic
  📚 TUTOR (depth 60):       Layers 1–60       — explanatory

The OUTPUT LAYER always connects to whichever is the last active layer.

SKIP CONNECTIONS (Residual)
-----------------------------
At every 10th layer, we add a SKIP CONNECTION: the input to that block of
10 layers is ADDED to the block's output. This is called a "residual connection".

WHY? In very deep networks (150 layers), gradients tend to vanish (become
tiny) as they travel backward through many layers. A skip connection gives
the gradient a "shortcut highway" to flow through unchanged. This is the
key innovation behind ResNet (which allows training networks 1000+ layers deep).

  Without skip: output = f(input)
  With skip:    output = f(input) + input   ← identity shortcut
"""

import numpy as np
from .layer import Layer
from .dropout import Dropout
from .batchnorm import BatchNorm
from .losses import get_loss
from .activations import sigmoid, softmax


# -- Mode configurations -------------------------------------------------------
# Each mode specifies:
# - depth: How many layers are active.
# - temp: Softmax/Sigmoid temperature (lower = stricter/sharper, higher = softer/creative).
# - noise: Gaussian noise injected into hidden layers (forces varied thinking).
# - dropout_scale: Multiplier for the base dropout rate.

MODE_CONFIGS = {
    "quick":      {"depth": 30,  "temp": 1.0, "noise": 0.0,   "dropout_scale": 1.0},
    "analytic":   {"depth": 90,  "temp": 0.8, "noise": 0.0,   "dropout_scale": 0.5},  # Sharp, focused predictions
    "deep_think": {"depth": 150, "temp": 1.0, "noise": 0.02,  "dropout_scale": 1.0},  # Jack of all trades: balanced, slight noise for robustness
    "code":       {"depth": 120, "temp": 0.5, "noise": 0.0,   "dropout_scale": 0.1},  # Extremely strict, deterministic, logical
    "creative":   {"depth": 60,  "temp": 2.0, "noise": 0.15,  "dropout_scale": 1.5},  # High variance, soft outputs, exploratory
    "research":   {"depth": 90,  "temp": 1.0, "noise": 0.0,   "dropout_scale": 1.0},  # Methodical, standard parameters
    "debug":      {"depth": 120, "temp": 0.2, "noise": 0.0,   "dropout_scale": 0.0},  # Absolute precision, zero variance
    "tutor":      {"depth": 60,  "temp": 1.2, "noise": 0.05,  "dropout_scale": 1.0},  # Smooth, approachable, slight variance
}

MODE_INFO: dict[str, str] = {
    "quick":      "⚡ Quick Answer — fast, sharp, 30 layers",
    "analytic":   "📊 Deep Analysis — structured reasoning, 90 layers",
    "deep_think": "🧠 Deep Think — full power, all 150 layers",
    "code":       "💻 Precision Mode — exact & logical, 120 layers",
    "creative":   "🎨 Creative Mode — exploratory & imaginative, 60 layers",
    "research":   "🔬 Research Mode — methodical & thorough, 90 layers",
    "debug":      "🐛 Debug Mode — step-by-step, zero variance, 120 layers",
    "tutor":      "📚 Explain Mode — patient & clear, 60 layers",
}


class NeuralNetwork:
    """
    A deep neural network with up to 150 hidden layers.

    Supports multiple modes that activate different subsets of layers,
    residual (skip) connections every 10 layers, and batch training.

    Parameters
    ----------
    input_size : int
        Dimensionality of each input sample (number of features).
    hidden_size : int
        Number of neurons in each hidden layer.
    output_size : int
        Number of output neurons (= number of classes for classification,
        or 1 for binary/regression).
    num_hidden_layers : int
        Total hidden layers to build (default 150).
    hidden_activation : str
        Activation for hidden layers. Default 'leaky_relu'.
    output_activation : str
        Activation for the output layer. 'sigmoid', 'linear', or 'softmax'.
    task : str
        'binary', 'multiclass', or 'regression'. Determines loss function.
    mode : str
        Starting inference mode.
    use_residual : bool
        If True, add skip connections every 10 layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int = 150,
        hidden_activation: str = "leaky_relu",
        output_activation: str = "sigmoid",
        task: str = "binary",
        mode: str = "quick",
        use_residual: bool = True,
        dropout_rate: float = 0.1,
    ):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden  = num_hidden_layers
        self.task        = task
        self.use_residual = use_residual

        # Build loss function based on task type
        loss_map = {
            "binary":     "bce",   # binary cross-entropy
            "multiclass": "cce",   # categorical cross-entropy
            "regression": "mse",   # mean squared error
        }
        self._loss_fn = get_loss(loss_map[task])

        # -- Build Layers ---------------------------------------------------
        # Layer 0: input → hidden (first hidden layer)
        self.hidden_layers: list[Layer] = []
        self.batch_norms: list[BatchNorm] = []
        self.dropouts: list[Dropout] = []

        for i in range(num_hidden_layers):
            # First layer connects input_size → hidden_size
            # All subsequent layers connect hidden_size → hidden_size
            in_size = input_size if i == 0 else hidden_size

            self.hidden_layers.append(
                Layer(
                    input_size=in_size,
                    output_size=hidden_size,
                    activation=hidden_activation,
                    layer_id=i + 1,
                )
            )
            # Add Batch Normalization (temporarily disabled)
            # self.batch_norms.append(BatchNorm(size=hidden_size, layer_id=i + 1))
            
            # Add a dropout layer matching this hidden layer
            self.dropouts.append(Dropout(rate=dropout_rate, layer_id=i + 1))

        # Output layer: hidden → output
        # Uses 'linear' activation here; loss functions handle final non-linearity
        self.output_layer = Layer(
            input_size=hidden_size,
            output_size=output_size,
            activation="linear",   # raw logits; softmax/sigmoid applied in loss
            layer_id=num_hidden_layers + 1,
        )

        self._output_activation_name = output_activation

        # Track training history for plotting
        self.loss_history:     list[float] = []
        self.acc_history:      list[float] = []
        self.val_loss_history: list[float] = []
        self.val_acc_history:  list[float] = []
        
        # Set initial mode (applies depth, noise, temperature, and dropout scaling)
        self._base_dropout = dropout_rate
        self.set_mode(mode)

    # --- Mode control --------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """
        Switch inference mode, controlling active layers and AI skills.

        Parameters
        ----------
        mode : str
            One of: quick, analytic, deep_think, code, creative, research, debug, tutor.
        """
        if mode not in MODE_CONFIGS:
            available = list(MODE_CONFIGS.keys())
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {available}")
        
        self.mode = mode
        config = MODE_CONFIGS[mode]
        
        # Apply depth
        self._active_layers = min(config["depth"], self.num_hidden)
        
        # Apply temperature & noise
        self._temperature = config["temp"]
        self._noise       = config["noise"]
        
        # Apply dropout scaling based on the mode's skill requirement
        base_dropout = getattr(self, "_base_dropout", None)
        if base_dropout is None:
            # First time setup: find the current dropout rate
            base_dropout = self.dropouts[0].rate if self.dropouts else 0.0
            self._base_dropout = base_dropout
            
        for dropout in self.dropouts:
            dropout.rate = min(0.99, base_dropout * config["dropout_scale"])
            dropout.keep = 1.0 - dropout.rate

    def mode_info(self) -> str:
        return MODE_INFO.get(self.mode, self.mode)

    # --- Forward Pass --------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Pass input through the active layers and output layer.

        Parameters
        ----------
        x : np.ndarray, shape [batch_size, input_size]

        Returns
        -------
        np.ndarray, shape [batch_size, output_size]
            Raw predictions (probabilities for classification, values for regression).
        """
        current = x

        # -- Pass through active hidden layers ------------------------------
        for i, layer in enumerate(self.hidden_layers[:self._active_layers]):

            # Store input for residual connection (every 10 layers)
            residual = current

            # Apply the layer's linear transform + activation
            current = layer.forward(current)
            
            # Apply Batch Normalization (only if BN layers are active)
            if self.batch_norms:
                current = self.batch_norms[i].forward(current)
            
            # Apply dropout
            current = self.dropouts[i].forward(current)
            
            # Apply AI skill noise (if configured by the current mode) ONLY during training
            if self._noise > 0.0 and self.dropouts[0].training:
                # User's brilliant insight: scale noise by network depth so it doesn't explode!
                # We use sqrt(active_layers) because random variance adds up quadratically.
                scaled_noise = self._noise / np.sqrt(self._active_layers)
                current += np.random.randn(*current.shape) * scaled_noise

            # -- Residual (Skip) Connection ---------------------------------
            # Every 10 layers, add the input of this block back to the output.
            # This is only possible when dimensions match.
            if (
                self.use_residual
                and (i + 1) % 10 == 0       # every 10th layer
                and i > 0                    # not the very first layer
                and residual.shape == current.shape  # dimensions must match
            ):
                # Add the shortcut: gradient can flow directly through this path
                current = current + residual

        # -- Output layer --------------------------------------------------
        logits = self.output_layer.forward(current)
        
        # Apply temperature scaling (temperature > 1 softens, < 1 sharpens)
        logits = logits / max(self._temperature, 1e-6)

        # Apply final activation based on task type
        if self.task == "binary":
            return sigmoid(logits)          # → probabilities in (0, 1)
        elif self.task == "multiclass":
            return softmax(logits)          # → probability distribution summing to 1
        else:  # regression
            return logits * self._temperature # Scale raw values by temp instead of logits

    # --- Loss Computation ----------------------------------------------------

    def compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Compute loss value and gradient for the output."""
        return self._loss_fn(y_true, y_pred)

    # --- Backward Pass -------------------------------------------------------

    def backward(self, grad: np.ndarray) -> None:
        """
        Propagate gradients backward through the network.

        Starting from the output layer, each layer receives the gradient
        from the layer ABOVE it and passes a gradient to the layer BELOW it.

        Parameters
        ----------
        grad : np.ndarray
            dLoss/d_predictions — the initial gradient from the loss function.
        """
        # -- Output layer backward -----------------------------------------
        grad = self.output_layer.backward(grad)

        # -- Hidden layers backward (reverse order!) -----------------------
        # We go from the LAST active layer back to the FIRST.
        # Each layer's backward() consumes the gradient from the layer ahead
        # and produces a gradient for the layer behind.
        active = list(enumerate(self.hidden_layers[:self._active_layers]))
        
        for i, layer in reversed(active):
            # If this is a residual layer, the gradient splits!
            # The grad flows BOTH into the block and directly past it.
            is_residual = self.use_residual and (i + 1) % 10 == 0 and i > 0
            
            if is_residual:
                shortcut_grad = grad.copy()
                
            # Pass gradient through dropout first
            grad = self.dropouts[i].backward(grad)
            # Then through Batch Normalization (only if BN layers are active)
            if self.batch_norms:
                grad = self.batch_norms[i].backward(grad)
            # Then through the dense layer
            grad = layer.backward(grad)
            
            # Recombine the shortcut gradient
            if is_residual:
                grad = grad + shortcut_grad

    def set_training(self, training: bool) -> None:
        """Switch dropout and batchnorm layers between training and inference modes."""
        for dropout in self.dropouts:
            dropout.set_training(training)
        for bn in self.batch_norms:
            bn.set_training(training)

    # --- Prediction Utilities ------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return raw predictions (probabilities or values)."""
        self.set_training(False)
        return self.forward(x)

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """
        Return integer class labels.
        
        For binary: 0 or 1 based on 0.5 threshold.
        For multiclass: argmax of probability vector.
        """
        preds = self.forward(x)
        if self.task == "binary":
            return (preds >= 0.5).astype(int).flatten()
        elif self.task == "multiclass":
            return np.argmax(preds, axis=1)
        else:
            return preds  # regression: return raw values

    def accuracy(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """Compute classification accuracy (not meaningful for regression)."""
        if self.task == "regression":
            return 0.0
        preds = self.predict_classes(x)
        if self.task == "binary":
            labels = y_true.flatten().astype(int)
        else:
            labels = np.argmax(y_true, axis=1)
        return float(np.mean(preds == labels))

    # --- Model Persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        """Save all layer weights to a .npz file."""
        data = {}
        for i, layer in enumerate(self.hidden_layers):
            data[f"W_{i}"] = layer.W
            data[f"b_{i}"] = layer.b
        # Save BatchNorm params
        for i, bn in enumerate(self.batch_norms):
            data[f"bn_gamma_{i}"] = bn.W
            data[f"bn_beta_{i}"] = bn.b
            data[f"bn_run_mean_{i}"] = bn.running_mean
            data[f"bn_run_var_{i}"] = bn.running_var
            
        data["W_out"] = self.output_layer.W
        data["b_out"] = self.output_layer.b
        np.savez(path, **data)
        print(f"  [SAVE] Model saved to '{path}.npz'")

    def load(self, path: str) -> None:
        """Load weights from a saved .npz file."""
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        for i, layer in enumerate(self.hidden_layers):
            layer.W = data[f"W_{i}"]
            layer.b = data[f"b_{i}"]
        # Load BatchNorm params
        for i, bn in enumerate(self.batch_norms):
            if f"bn_gamma_{i}" in data:
                bn.W = data[f"bn_gamma_{i}"]
                bn.b = data[f"bn_beta_{i}"]
                bn.running_mean = data[f"bn_run_mean_{i}"]
                bn.running_var = data[f"bn_run_var_{i}"]
                
        self.output_layer.W = data["W_out"]
        self.output_layer.b = data["b_out"]
        print(f"  [LOAD] Model loaded from '{path}'")

    # --- Summary -------------------------------------------------------------

    def param_count(self) -> int:
        """Return total number of learnable parameters."""
        total = sum(l.param_count() for l in self.hidden_layers)
        # total += sum(bn.param_count() for bn in self.batch_norms)
        return total + self.output_layer.param_count()

    def summary(self) -> None:
        """Print the network architecture."""
        total_params = self.param_count()

        print("\n" + "=" * 60)
        print(f"  [BRAIN] DeepMind AI — Neural Network Summary")
        print("=" * 60)
        print(f"  Total hidden layers : {self.num_hidden}")
        print(f"  Active mode         : {self.mode_info()}")
        print(f"  Active layers       : {self._active_layers} / {self.num_hidden}")
        print(f"  Input size          : {self.input_size}")
        print(f"  Hidden size         : {self.hidden_size} neurons/layer")
        print(f"  Output size         : {self.output_size}")
        print(f"  Task                : {self.task}")
        print(f"  Residual connections: {self.use_residual}")
        print(f"  Total parameters    : {total_params:,}")
        print("=" * 60 + "\n")
