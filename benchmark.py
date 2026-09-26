"""
benchmark.py — AI Intelligence Evaluation System
==================================================

Precisely measures four key metrics for the neural network:

  1. ACCURACY          (target > 95%)
     → Average validation accuracy across all benchmark datasets

  2. HALLUCINATION RATE (target < 1%)
     → % of inference predictions that CHANGE when tiny noise (std=0.01)
       is added to inputs. Measures prediction stability/robustness.

  3. COPY-PASTE RATE   (target < 0.5%)
     → max(0, train_acc - val_acc)
       Measures overfitting — if the AI "memorized" training data
       instead of truly generalizing, this gap is large.

  4. IQ SCORE          (target = 120)
     → Composite score derived from all three metrics.
       IQ 100 = baseline AI. IQ 120 = target. IQ 125 = perfect.

FORMULA for IQ:
    base = 100
    acc_pts   = (val_acc - 0.50) / 0.50 * 14     [0-14 pts]
    hall_pts  = (0.02 - hall_rate) / 0.02 * 3    [0-3 pts]
    overfit_pts = (0.01 - overfit) / 0.01 * 3    [0-3 pts]
    milestone = +5 if ALL three goals simultaneously met
    IQ = base + acc_pts + hall_pts + overfit_pts + milestone
    
    At targets (95%, 1% hall, 0.5% overfit) → IQ ≈ 120-121 ✓
    At perfect (100%, 0%, 0%)               → IQ = 125
"""

import sys
import numpy as np
from typing import Optional

# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from nn.network  import NeuralNetwork, MODE_CONFIGS
from nn.trainer  import Trainer
from data.simulator import get_dataset, DATASETS

# ---------------------------------------------------------------------------
# IQ Formula
# ---------------------------------------------------------------------------

def compute_iq(val_acc: float, overfit_rate: float, hall_rate: float) -> int:
    """
    Compute AI IQ score.

    Parameters
    ----------
    val_acc      : 0.0-1.0  validation accuracy
    overfit_rate : 0.0-1.0  train_acc - val_acc (positive = overfitting)
    hall_rate    : 0.0-1.0  fraction of predictions that change under tiny noise

    Returns
    -------
    int  IQ score (100 = baseline, 120 = target, 125 = perfect)
    """
    acc_pts  = max(0.0, (val_acc - 0.50) / 0.50 * 14.0)
    hall_pts = max(0.0, (0.02 - hall_rate) / 0.02 * 3.0)
    over_pts = max(0.0, (0.01 - overfit_rate) / 0.01 * 3.0)

    all_goals_met = (
        val_acc      >= 0.95  and
        overfit_rate <= 0.005 and
        hall_rate    <= 0.01
    )
    milestone = 5.0 if all_goals_met else 0.0

    return round(100.0 + acc_pts + hall_pts + over_pts + milestone)


# ---------------------------------------------------------------------------
# Hallucination Rate
# ---------------------------------------------------------------------------

def _to_class_labels(preds: np.ndarray) -> np.ndarray:
    """Convert raw network output to discrete class-label array."""
    if preds.ndim == 1 or preds.shape[1] == 1:
        # Binary: threshold at 0.5
        return (preds.ravel() >= 0.5).astype(np.int8)
    else:
        # Multiclass: argmax
        return np.argmax(preds, axis=1).astype(np.int8)


def measure_hallucination(net: NeuralNetwork, X: np.ndarray,
                           noise_std: float = 0.01) -> float:
    """
    Measure the fraction of predictions that change their CLASS DECISION
    when tiny input noise is added. Compares discrete labels (not floats)
    so a probability shift of 0.001 does NOT count as hallucination unless
    it actually crosses the decision boundary.

    Parameters
    ----------
    net       : trained NeuralNetwork
    X         : validation inputs, shape [n, features]
    noise_std : std-dev of the Gaussian noise probe (default 0.01)

    Returns
    -------
    float  hallucination rate: 0.0 (perfect) to 1.0 (fully unstable)
    """
    net.set_training(False)

    # Clean class-label decisions
    preds_clean_raw = net.predict(X)
    labels_clean    = _to_class_labels(preds_clean_raw)

    # Noisy predictions — 3 trials, count if ANY trial flips the label
    flipped_any = np.zeros(len(X), dtype=bool)
    for _ in range(3):
        noise = np.random.randn(*X.shape) * noise_std
        preds_noisy_raw = net.predict(X + noise)
        labels_noisy    = _to_class_labels(preds_noisy_raw)
        flipped_any    |= (labels_noisy != labels_clean)

    return float(np.mean(flipped_any))


# ---------------------------------------------------------------------------
# Full benchmark for ONE dataset
# ---------------------------------------------------------------------------

def benchmark_dataset(
    dataset_name: str,
    mode: str,
    hidden_size: int,
    lr: float,
    epochs: int,
    n_samples: int = 1200,
    verbose: bool = True,
) -> dict:
    """
    Train a fresh network on one dataset and measure all four metrics.

    Returns
    -------
    dict with keys: dataset, mode, val_acc, train_acc, overfit, hall_rate,
                    iq, goals_met  (dict of booleans)
    """
    # 1. Generate data
    X, y, task, in_size, out_size = get_dataset(dataset_name, n_samples=n_samples)

    # Normalize inputs to [-1, 1]
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_range = np.where(X_max - X_min == 0, 1, X_max - X_min)
    X = (X - X_min) / X_range * 2 - 1

    # Split: 70% train, 15% val, 15% held-out test (never seen during training)
    n  = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]
    X_train, y_train = X[:n_train],        y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],  y[n_train+n_val:]

    # 2. Build network
    net = NeuralNetwork(
        input_size=in_size,
        hidden_size=hidden_size,
        output_size=out_size,
        task=task,
        mode=mode,
    )

    # 3. Train with the Trainer
    trainer = Trainer(
        net,
        epochs=epochs,
        batch_size=32,
        learning_rate=lr,
        lr_decay=0.9995,
        early_stopping_patience=epochs,
        l2_lambda=1e-4,
        verbose=False,
    )
    trainer.fit(X_train, y_train)

    # 4. Measure accuracy
    net.set_training(False)
    train_acc = net.accuracy(X_train, y_train)
    val_acc   = net.accuracy(X_val,   y_val)
    test_acc  = net.accuracy(X_test,  y_test)
    overfit   = max(0.0, train_acc - val_acc)

    # 5. Measure hallucination on the held-out test set
    hall_rate = measure_hallucination(net, X_test, noise_std=0.01)

    # 6. Compute IQ
    iq = compute_iq(val_acc, overfit, hall_rate)

    goals = {
        "accuracy_95":     val_acc   >= 0.95,
        "hallucination_1": hall_rate <= 0.01,
        "copy_paste_05":   overfit   <= 0.005,
        "iq_120":          iq        >= 120,
    }
    all_met = all(goals.values())

    result = {
        "dataset":    dataset_name,
        "mode":       mode,
        "hidden_sz":  hidden_size,
        "lr":         lr,
        "epochs":     epochs,
        "train_acc":  train_acc,
        "val_acc":    val_acc,
        "test_acc":   test_acc,
        "overfit":    overfit,
        "hall_rate":  hall_rate,
        "iq":         iq,
        "goals":      goals,
        "all_met":    all_met,
    }

    if verbose:
        _print_result(result)

    return result


def _print_result(r: dict) -> None:
    """Pretty-print a benchmark result."""
    G = lambda ok: "PASS" if ok else "FAIL"
    print(f"\n  Dataset    : {r['dataset']} | Mode: {r['mode']} | "
          f"Hidden: {r['hidden_sz']} | LR: {r['lr']} | Epochs: {r['epochs']}")
    print(f"  Accuracy   : train={r['train_acc']*100:.2f}%  val={r['val_acc']*100:.2f}%  test={r['test_acc']*100:.2f}%")
    print(f"  Overfitting: {r['overfit']*100:.3f}%    [{G(r['goals']['copy_paste_05'])}  target <0.5%]")
    print(f"  Hallucin.  : {r['hall_rate']*100:.3f}%   [{G(r['goals']['hallucination_1'])}  target <1.0%]")
    print(f"  IQ Score   : {r['iq']}            [{G(r['goals']['iq_120'])}  target ≥120]")
    status = "ALL GOALS MET" if r['all_met'] else "GOALS NOT YET MET"
    print(f"  Status     : {status}")


# ---------------------------------------------------------------------------
# Full sweep: all datasets
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS = ["moons", "circles", "xor", "multiclass"]

def run_full_benchmark(mode: str, hidden_size: int, lr: float, epochs: int,
                       verbose: bool = True) -> dict:
    """
    Run benchmark on all core datasets and compute aggregate metrics.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  FULL BENCHMARK  mode={mode}  hidden={hidden_size}  lr={lr}  ep={epochs}")
        print(f"{'='*60}")

    results = []
    for ds in BENCHMARK_DATASETS:
        r = benchmark_dataset(ds, mode=mode, hidden_size=hidden_size,
                              lr=lr, epochs=epochs, verbose=verbose)
        results.append(r)

    avg_val_acc   = float(np.mean([r["val_acc"]   for r in results]))
    avg_overfit   = float(np.mean([r["overfit"]   for r in results]))
    avg_hall      = float(np.mean([r["hall_rate"] for r in results]))
    avg_iq        = compute_iq(avg_val_acc, avg_overfit, avg_hall)
    n_passed      = sum(1 for r in results if r["all_met"])

    aggregate = {
        "mode":        mode,
        "hidden_size": hidden_size,
        "lr":          lr,
        "epochs":      epochs,
        "avg_val_acc": avg_val_acc,
        "avg_overfit": avg_overfit,
        "avg_hall":    avg_hall,
        "avg_iq":      avg_iq,
        "n_passed":    n_passed,
        "n_total":     len(results),
        "all_passed":  n_passed == len(results),
        "results":     results,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  AGGREGATE RESULTS")
        print(f"  Avg Accuracy    : {avg_val_acc*100:.2f}%  [{'PASS' if avg_val_acc>=0.95 else 'FAIL'}]")
        print(f"  Avg Overfitting : {avg_overfit*100:.3f}% [{'PASS' if avg_overfit<=0.005 else 'FAIL'}]")
        print(f"  Avg Hallucin.   : {avg_hall*100:.3f}%   [{'PASS' if avg_hall<=0.01 else 'FAIL'}]")
        print(f"  Composite IQ    : {avg_iq}         [{'PASS' if avg_iq>=120 else 'FAIL'}]")
        print(f"  Datasets Passed : {n_passed}/{len(results)}")
        print(f"{'─'*60}")

    return aggregate


if __name__ == "__main__":
    # Quick self-test
    np.random.seed(0)
    r = run_full_benchmark(mode="quick", hidden_size=32, lr=0.005, epochs=100)
    print(f"\nFinal IQ: {r['avg_iq']}")
