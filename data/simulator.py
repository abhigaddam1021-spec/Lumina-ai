"""
simulator.py — Simulated Training Datasets
============================================

WHY SIMULATED DATA?
--------------------
We don't need real-world data to train and study a neural network.
Synthetic datasets let us:
  - Control the difficulty (add noise, change class overlap)
  - Know the GROUND TRUTH (perfect labels)
  - Train instantly on any machine
  - Understand what the network is actually learning

We provide four datasets, each exercising a different aspect of the network:

  1. MOONS       — Two interleaved crescent shapes (binary classification)
                   A linear model CANNOT separate this. The network must
                   learn a curved decision boundary. Good test of depth.

  2. CIRCLES     — Concentric circles (binary classification)
                   Even harder than moons — the inner ring is one class,
                   the outer ring is the other. Linear models fail completely.

  3. XOR         — Classic XOR pattern (binary classification)
                   The oldest non-linear benchmark. A single-layer network
                   cannot learn XOR. Requires at least 1 hidden layer.

  4. MULTICLASS  — Multiple Gaussian blobs (multiclass classification)
                   N clusters, each a different class. Tests softmax output.

  5. REGRESSION  — A noisy sine wave (regression)
                   Network must learn a continuous function, not just 0/1.

Each function returns:
  X : np.ndarray, shape [n_samples, n_features]  — input features
  y : np.ndarray, shape [n_samples, n_outputs]   — labels (one-hot if multiclass)
"""

import numpy as np


def make_moons(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> tuple:
    """
    Two interleaved crescent (moon) shapes.
    Classic non-linear binary classification benchmark.

    Parameters
    ----------
    n_samples : int   — total data points (split evenly between two moons)
    noise     : float — standard deviation of Gaussian noise added to points
    seed      : int   — random seed for reproducibility

    Returns
    -------
    X : shape [n_samples, 2]  — 2D coordinates
    y : shape [n_samples, 1]  — labels: 0 or 1
    """
    np.random.seed(seed)
    n_each = n_samples // 2

    # Moon 1: upper crescent — angles 0 to π
    theta1 = np.linspace(0, np.pi, n_each)
    moon1 = np.column_stack([np.cos(theta1), np.sin(theta1)])

    # Moon 2: lower crescent — angles π to 2π, shifted right and down
    theta2 = np.linspace(0, np.pi, n_each)
    moon2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])

    # Combine and add Gaussian noise (makes classification harder/realistic)
    X = np.vstack([moon1, moon2]) + np.random.randn(n_samples, 2) * noise
    y = np.array([0] * n_each + [1] * n_each).reshape(-1, 1)

    # Shuffle so classes are mixed throughout the dataset
    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


def make_circles(n_samples: int = 1000, noise: float = 0.08, seed: int = 42) -> tuple:
    """
    Two concentric circles — inner circle is class 0, outer is class 1.
    Requires the network to learn radial distance as a feature.

    Returns
    -------
    X : shape [n_samples, 2]
    y : shape [n_samples, 1]  — 0 (inner) or 1 (outer)
    """
    np.random.seed(seed)
    n_each = n_samples // 2

    # Inner circle: radius ~0.3
    angles1 = np.random.uniform(0, 2 * np.pi, n_each)
    r1 = np.random.uniform(0.2, 0.4, n_each)
    X1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])

    # Outer circle: radius ~0.8
    angles2 = np.random.uniform(0, 2 * np.pi, n_each)
    r2 = np.random.uniform(0.7, 0.9, n_each)
    X2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])

    X = np.vstack([X1, X2]) + np.random.randn(n_samples, 2) * noise
    y = np.array([0] * n_each + [1] * n_each).reshape(-1, 1)

    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


def make_xor(n_samples: int = 1000, noise: float = 0.05, seed: int = 42) -> tuple:
    """
    XOR pattern: 4 quadrants, alternating classes (checkerboard).
    Cannot be solved by any linear model — requires hidden layers.

    Class 0: quadrants (+,+) and (-,-) → same-sign inputs
    Class 1: quadrants (+,-) and (-,+) → different-sign inputs

    Returns
    -------
    X : shape [n_samples, 2]
    y : shape [n_samples, 1]
    """
    np.random.seed(seed)

    # Generate random points in [-1, 1] x [-1, 1], avoiding the ambiguous axis band
    X = np.random.uniform(-1, 1, (n_samples, 2))
    # Remove points too close to the axes (they are inherently ambiguous)
    mask = (np.abs(X[:, 0]) > 0.08) & (np.abs(X[:, 1]) > 0.08)
    X = X[mask]
    # Pad back to n_samples if needed
    while len(X) < n_samples:
        extra = np.random.uniform(-1, 1, (n_samples, 2))
        extra = extra[(np.abs(extra[:, 0]) > 0.08) & (np.abs(extra[:, 1]) > 0.08)]
        X = np.vstack([X, extra])
    X = X[:n_samples]

    # XOR label: 1 if signs differ (x1 * x2 < 0), 0 otherwise
    y = ((X[:, 0] * X[:, 1]) < 0).astype(float).reshape(-1, 1)

    # Add small noise
    X += np.random.randn(n_samples, 2) * noise
    return X.astype(np.float32), y.astype(np.float32)


def make_multiclass(
    n_samples: int = 1500,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Multiple Gaussian clusters — one per class.
    Each class is a blob of points centered at a random location.

    Returns
    -------
    X : shape [n_samples, 2]
    y : shape [n_samples, n_classes]  ← ONE-HOT encoded labels
    """
    np.random.seed(seed)
    n_each = n_samples // n_classes

    # Random cluster centers spread around the origin
    centers = np.random.uniform(-3, 3, (n_classes, 2))

    X_parts, y_parts = [], []
    for c, center in enumerate(centers):
        # Generate points around this cluster's center
        pts = center + np.random.randn(n_each, 2) * noise
        X_parts.append(pts)

        # One-hot label: e.g. class 1 of 3 → [0, 1, 0]
        one_hot = np.zeros((n_each, n_classes))
        one_hot[:, c] = 1.0
        y_parts.append(one_hot)

    X = np.vstack(X_parts)
    y = np.vstack(y_parts)

    idx = np.random.permutation(len(X))
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


def make_regression(
    n_samples: int = 1000,
    noise: float = 0.3,
    seed: int = 42,
) -> tuple:
    """
    Noisy sine wave — a classic regression benchmark.
    The network must learn the underlying y = sin(x) function.

    Returns
    -------
    X : shape [n_samples, 1]  ← single input feature
    y : shape [n_samples, 1]  ← continuous target value
    """
    np.random.seed(seed)

    # Evenly spaced x values in [0, 4π], randomly shuffled
    x = np.random.uniform(0, 4 * np.pi, n_samples)
    x_sorted = np.sort(x)

    # Target: sine wave + Gaussian noise
    y_clean = np.sin(x_sorted)
    y_noisy = y_clean + np.random.randn(n_samples) * noise

    return x_sorted.reshape(-1, 1).astype(np.float32), y_noisy.reshape(-1, 1).astype(np.float32)


def make_spirals(n_samples: int = 1000, noise: float = 0.1, seed: int = 42) -> tuple:
    """
    Two intertwined spirals. The ultimate test for 'Deep Think'.
    Standard shallow networks completely fail this task.
    """
    np.random.seed(seed)
    n = n_samples // 2
    
    def gen_spiral(delta_t, label):
        r = np.linspace(0.1, 5, n)
        t = np.linspace(0, 3 * np.pi, n) + delta_t
        x = r * np.sin(t) + np.random.randn(n) * noise
        y = r * np.cos(t) + np.random.randn(n) * noise
        return np.column_stack((x, y)), np.full((n, 1), label)

    X0, y0 = gen_spiral(0, 0)
    X1, y1 = gen_spiral(np.pi, 1)

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    
    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(np.float32)

def make_logic(n_samples: int = 1000, noise: float = 0.05, seed: int = 42) -> tuple:
    """
    High-dimensional boolean parity (XOR on steroids). 
    Perfect for 'Code' and 'Debug' modes which demand strict logic.
    Inputs are 8-dimensional boolean vectors (with slight noise).
    Output is 1 if the sum of true inputs is odd, 0 if even.
    """
    np.random.seed(seed)
    X_bool = np.random.randint(0, 2, (n_samples, 8))
    y = (X_bool.sum(axis=1) % 2).reshape(-1, 1)
    
    # Add slight noise to simulate analog transmission
    X = X_bool + np.random.randn(n_samples, 8) * noise
    return X.astype(np.float32), y.astype(np.float32)

def make_abstract(n_samples: int = 1000, noise: float = 1.0, seed: int = 42) -> tuple:
    """
    Highly overlapping, noisy abstract clusters.
    Perfect for 'Creative' mode which needs to handle ambiguity and soften outputs.
    """
    np.random.seed(seed)
    X0 = np.random.randn(n_samples // 2, 4) * noise - 0.5
    X1 = np.random.randn(n_samples // 2, 4) * noise + 0.5
    
    X = np.vstack([X0, X1])
    y = np.vstack([np.zeros((n_samples // 2, 1)), np.ones((n_samples // 2, 1))])
    
    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(np.float32)


# --- Dataset Registry ---------------------------------------------------------

DATASETS = {
    "moons":      {"fn": make_moons,      "task": "binary",     "input": 2, "output": 1},
    "circles":    {"fn": make_circles,    "task": "binary",     "input": 2, "output": 1},
    "xor":        {"fn": make_xor,        "task": "binary",     "input": 2, "output": 1},
    "multiclass": {"fn": make_multiclass, "task": "multiclass", "input": 2, "output": 3},
    "regression": {"fn": make_regression, "task": "regression", "input": 1, "output": 1},
    "spirals":    {"fn": make_spirals,    "task": "binary",     "input": 2, "output": 1},
    "logic":      {"fn": make_logic,      "task": "binary",     "input": 8, "output": 1},
    "abstract":   {"fn": make_abstract,   "task": "binary",     "input": 4, "output": 1},
}

# Human-readable display names shown in the UI
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "moons":      "🌙 Shape Sorter — separate two curved, interleaved shapes",
    "circles":    "🎯 Inside vs Outside — tell apart an inner ring from an outer ring",
    "xor":        "🧩 Logic Puzzle — learn a checkerboard pattern with no straight line solution",
    "multiclass": "🗂️ Group Classifier — sort data into 3 distinct colour-coded groups",
    "regression": "〰️ Curve Fitter — predict values along a noisy sine wave",
    "spirals":    "🌀 Spiral Tracer — untangle two interlocking spirals",
    "logic":      "💡 True/False Reasoning — learn Boolean logic rules from examples",
    "abstract":   "🎨 Pattern Finder — detect structure in overlapping, noisy clouds",
}


def get_dataset(name: str, **kwargs) -> tuple:
    """
    Return (X, y, task, input_size, output_size) for the named dataset.
    
    Usage: X, y, task, in_size, out_size = get_dataset("moons", n_samples=2000)
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
    info = DATASETS[name]
    X, y = info["fn"](**kwargs)
    return X, y, info["task"], info["input"], info["output"]
