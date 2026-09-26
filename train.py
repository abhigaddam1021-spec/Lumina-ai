#!/usr/bin/env python3
"""
train.py — Train the 150-Layer Neural Network from Scratch
============================================================

This is the main training script. Run it directly:

  python train.py                          # train on moons (default)
  python train.py --dataset circles        # concentric circles
  python train.py --dataset xor            # XOR problem
  python train.py --dataset multiclass     # 3-class blobs
  python train.py --dataset regression     # sine wave
  python train.py --mode deep_think        # use all 150 layers
  python train.py --epochs 200             # train longer
  python train.py --hidden-size 32         # smaller (faster) network

WHAT TO WATCH WHILE TRAINING
------------------------------
  loss     → training loss (should decrease over time)
  acc      → training accuracy (should increase)
  val_loss → validation loss (the important one — shows real performance)
  val_acc  → validation accuracy on unseen data

  If val_loss STOPS decreasing while train loss keeps dropping → OVERFITTING.
  The early stopping mechanism will catch this automatically.

After training, the model is saved to 'saved_model.npz'.
"""

import argparse
import sys

import numpy as np

# -- Check dependencies --------------------------------------------------------
try:
    import numpy as np
except ImportError:
    print("[ERR] NumPy not installed. Run:  pip install numpy")
    sys.exit(1)

from data.simulator import get_dataset, DATASETS
from nn.network     import NeuralNetwork, MODE_CONFIGS
from nn.trainer     import Trainer


# --- Normalization helpers ----------------------------------------------------

def normalize(X_train: np.ndarray, X_val: np.ndarray) -> tuple:
    """
    Standardize features to zero mean and unit variance.

    WHY NORMALIZE?
    ---------------
    Neural networks train much faster and more stably when all input
    features are on the same scale (~zero mean, ~unit variance).

    If one feature ranges from 0–1000 and another from 0–1, the network
    wastes iterations correcting for the scale difference.

    We compute mean/std from TRAINING data only, then apply to validation.
    NEVER compute statistics from validation/test data — that would be
    "data leakage" (peeking at future information).
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8  # + small epsilon to avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std   # use TRAINING statistics

    return X_train_norm, X_val_norm, mean, std


# --- Main ---------------------------------------------------------------------

def main():
    # Fix Windows terminal encoding so Unicode box-drawing characters print correctly
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Train the 150-layer neural network from scratch (no API key needed)."
    )
    parser.add_argument(
        "--dataset", default="moons",
        choices=list(DATASETS.keys()),
        help="Which simulated dataset to train on (default: moons)."
    )
    parser.add_argument(
        "--mode", default="quick",
        choices=list(MODE_CONFIGS.keys()),
        help="Which AI mode to use (default: quick)."
    )
    parser.add_argument(
        "--hidden-size", type=int, default=16,
        help="Neurons per hidden layer (default: 16). Larger = more capacity, slower."
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max training epochs (default: 100)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Initial learning rate (default: 0.001)."
    )
    parser.add_argument(
        "--n-samples", type=int, default=1200,
        help="Number of training samples to generate (default: 1200)."
    )
    parser.add_argument(
        "--no-residual", action="store_true",
        help="Disable residual (skip) connections."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate (default: 0.1)."
    )
    parser.add_argument(
        "--optimizer", default="adam",
        choices=["adam", "sgd", "sgd_momentum"],
        help="Optimizer to use (default: adam)."
    )
    parser.add_argument(
        "--save", default="saved_model",
        help="File path to save trained weights (default: saved_model)."
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  [BRAIN] DeepMind AI — Training from Scratch")
    print("=" * 60)
    print(f"  Dataset     : {args.dataset}")
    depth = MODE_CONFIGS[args.mode]["depth"]
    print(f"  Mode        : {args.mode}  ({depth} active layers)")
    print(f"  Hidden size : {args.hidden_size} neurons/layer")
    print(f"  Optimizer   : {args.optimizer}  (lr={args.lr})")

    # -- 1. Generate simulated dataset -------------------------------------
    print(f"\n  [DATA] Generating '{args.dataset}' dataset ({args.n_samples} samples)...")
    X, y, task, input_size, output_size = get_dataset(
        args.dataset, n_samples=args.n_samples
    )
    print(f"     X shape: {X.shape}  y shape: {y.shape}  task: {task}")

    # -- 2. Train/val split for normalization ------------------------------
    n_val   = int(len(X) * 0.15)
    n_train = len(X) - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    # Normalize features (important for stable training)
    X_train_n, X_val_n, feat_mean, feat_std = normalize(X_train, X_val)

    # Put back together (Trainer will re-split internally)
    X_norm = np.vstack([X_train_n, X_val_n])
    y_all  = np.vstack([y_train,   y_val])

    # -- 3. Build the network ----------------------------------------------
    print(f"\n  [BUILD] Building 150-layer network...")

    # Choose output activation based on task type
    out_act = {"binary": "sigmoid", "multiclass": "softmax", "regression": "linear"}[task]

    net = NeuralNetwork(
        input_size        = input_size,
        hidden_size       = args.hidden_size,
        output_size       = output_size,
        num_hidden_layers = 150,
        hidden_activation = "leaky_relu",   # handles deep nets well
        output_activation = out_act,
        task              = task,
        mode              = args.mode,
        use_residual      = not args.no_residual,
        dropout_rate      = args.dropout,
    )

    # -- 4. Create trainer and run -----------------------------------------
    trainer = Trainer(
        network              = net,
        optimizer_name       = args.optimizer,
        learning_rate        = args.lr,
        batch_size           = args.batch_size,
        epochs               = args.epochs,
        validation_split     = 0.15,
        early_stopping_patience = 15,
        lr_decay             = 0.997,
        verbose              = True,
    )

    print(f"\n  [RUN] Starting training...\n")
    history = trainer.fit(X_norm, y_all)

    # -- 5. Final evaluation -----------------------------------------------
    print("\n" + "-" * 60)
    print("  [STATS] Final Results:")
    if history["loss"]:
        print(f"     Best train loss : {min(history['loss']):.6f}")
        print(f"     Best val loss   : {min(history['val_loss']):.6f}")
        if task != "regression":
            print(f"     Best val acc    : {max(history['val_acc']):.4f}  "
                  f"({max(history['val_acc'])*100:.1f}%)")

    # -- 6. Save model -----------------------------------------------------
    net.save(args.save)
    print(f"\n  [OK] Training complete! Weights saved to '{args.save}.npz'")
    print("  Run 'python evaluate.py' to test the trained model.\n")


if __name__ == "__main__":
    main()
