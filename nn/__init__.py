"""nn package — expose the main classes."""

from .network   import NeuralNetwork, MODE_CONFIGS, MODE_INFO
from .trainer   import Trainer
from .optimizer import Adam, SGD, SGDMomentum, get_optimizer
from .losses    import get_loss
from .activations import get_activation

__all__ = [
    "NeuralNetwork", "Trainer",
    "Adam", "SGD", "SGDMomentum", "get_optimizer",
    "get_loss", "get_activation",
    "MODE_DEPTHS", "MODE_INFO",
]
