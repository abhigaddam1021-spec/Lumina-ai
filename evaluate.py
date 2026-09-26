#!/usr/bin/env python3
"""
evaluate.py — Test a Trained Model
=====================================

Loads a saved model and runs evaluation:
  - Accuracy / loss on a fresh test set
  - Prints example predictions vs ground truth
  - Lets you type custom inputs (for regression/binary tasks)

Usage:
  python evaluate.py                        # evaluate on moons (default)
  python evaluate.py --dataset circles
  python evaluate.py --model saved_model    # load specific weights file
"""

import argparse
import sys
import numpy as np

from data.simulator import get_dataset, DATASETS
from nn.network     import NeuralNetwork, MODE_CONFIGS
from train          import normalize


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DeepMind AI model.")
    parser.add_argument("--dataset", default="moons", choices=list(DATASETS.keys()))
    parser.add_argument("--model",   default="saved_model",
                        help="Path to saved .npz weights file.")
    parser.add_argument("--mode",    default="quick", choices=list(MODE_CONFIGS.keys()))
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-samples",   type=int, default=500,
                        help="Number of fresh test samples to generate.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  [TEST] DeepMind AI — Model Evaluation")
    print("=" * 60)

    # -- 1. Generate fresh test data ---------------------------------------
    # IMPORTANT: use a different seed than training so this is truly unseen data
    X, y, task, input_size, output_size = get_dataset(
        args.dataset, n_samples=args.n_samples, seed=999
    )

    # Normalize using same rough statistics (in practice, you'd save the train stats)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # -- 2. Rebuild network architecture ----------------------------------
    out_act = {"binary": "sigmoid", "multiclass": "softmax", "regression": "linear"}[task]

    net = NeuralNetwork(
        input_size        = input_size,
        hidden_size       = args.hidden_size,
        output_size       = output_size,
        num_hidden_layers = 150,
        hidden_activation = "leaky_relu",
        output_activation = out_act,
        task              = task,
        mode              = args.mode,
    )

    # -- 3. Load saved weights ---------------------------------------------
    try:
        net.load(args.model)
    except FileNotFoundError:
        print(f"\n  [ERR] No saved model found at '{args.model}.npz'.")
        print("     Run 'python train.py' first to train the model.")
        sys.exit(1)

    # -- 4. Evaluate -------------------------------------------------------
    predictions = net.predict(X_norm)
    loss, _     = net.compute_loss(y, predictions)

    print(f"\n  Dataset  : {args.dataset}  ({args.n_samples} test samples)")
    print(f"  Mode     : {args.mode}  ({MODE_DEPTHS[args.mode]} layers active)")
    print(f"  Test loss: {loss:.6f}")

    if task != "regression":
        acc = net.accuracy(X_norm, y)
        print(f"  Test acc : {acc:.4f}  ({acc*100:.1f}%)")

    # -- 5. Show example predictions ---------------------------------------
    print("\n  [LIST] Sample predictions (first 10):")
    print(f"  {'True':>10}  {'Predicted':>12}  {'Correct?':>8}")
    print("  " + "-" * 38)

    classes = net.predict_classes(X_norm[:10])
    for i in range(min(10, len(X_norm))):
        if task == "binary":
            true_label = int(y[i, 0])
            pred_label = int(classes[i])
            correct    = "✓" if true_label == pred_label else "✗"
            print(f"  {true_label:>10}  {pred_label:>12}  {correct:>8}")
        elif task == "multiclass":
            true_label = int(np.argmax(y[i]))
            pred_label = int(classes[i])
            correct    = "✓" if true_label == pred_label else "✗"
            print(f"  {true_label:>10}  {pred_label:>12}  {correct:>8}")
        else:
            true_val = float(y[i, 0])
            pred_val = float(predictions[i, 0])
            print(f"  {true_val:>10.3f}  {pred_val:>12.3f}")

    print()


if __name__ == "__main__":
    main()
