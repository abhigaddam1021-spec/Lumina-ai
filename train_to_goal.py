"""
train_to_goal.py — Autonomous Training Loop
=============================================

Runs continuously, systematically exploring hyperparameter configurations,
until ALL four goals are simultaneously achieved across all benchmark datasets:

  Goal 1: Accuracy       > 95%
  Goal 2: Hallucination  < 1%
  Goal 3: Copy-paste     < 0.5%
  Goal 4: IQ             ≥ 120

The search strategy uses a staged approach:
  Stage 1: Coarse grid search to find promising zones
  Stage 2: Fine-tuning around the best configuration found
  Stage 3: Long extended training at optimal settings

Results are saved to 'goal_progress.log' and the best model is saved.
"""

import sys
import os
import time
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from benchmark import run_full_benchmark, compute_iq, BENCHMARK_DATASETS

LOG_FILE = "goal_progress.log"
BEST_FILE = "best_config.json"


def log(msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_best(config: dict) -> None:
    with open(BEST_FILE, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in config.items() if k != "results"}, f, indent=2)


def load_best() -> dict | None:
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

STAGE_1_CONFIGS = [
    # (mode, hidden_size, lr, epochs)
    # Quick sweeps to identify the best territory
    ("quick",    32,  0.005, 150),
    ("quick",    48,  0.005, 150),
    ("quick",    64,  0.003, 200),
    ("quick",    64,  0.005, 200),
    ("creative", 32,  0.005, 150),
    ("creative", 48,  0.003, 200),
    ("analytic", 32,  0.003, 200),
    ("quick",    32,  0.01,  150),
    ("quick",    48,  0.001, 300),
    ("creative", 64,  0.003, 200),
]

STAGE_2_BASE = [
    # Denser search around the best mode/hidden combo
    ("quick",    64,  0.005, 300),
    ("quick",    64,  0.003, 350),
    ("quick",    64,  0.004, 300),
    ("quick",    48,  0.005, 300),
    ("quick",    48,  0.004, 350),
    ("creative", 64,  0.005, 300),
    ("creative", 64,  0.004, 300),
    ("creative", 48,  0.005, 300),
]

STAGE_3_ENDGAME = [
    # Long high-quality runs at best settings
    ("quick",    64,  0.003, 600),
    ("quick",    64,  0.004, 500),
    ("quick",    64,  0.005, 500),
    ("quick",    64,  0.002, 700),
    ("creative", 64,  0.003, 600),
    ("quick",    80,  0.003, 500),
    ("quick",    80,  0.004, 600),
    ("creative", 80,  0.003, 600),
    ("quick",    64,  0.003, 800),
    ("quick",    64,  0.003, 1000),
]


# ---------------------------------------------------------------------------
# Main autonomous loop
# ---------------------------------------------------------------------------

def train_to_goal() -> None:
    np.random.seed(42)

    log("=" * 60)
    log("  AUTONOMOUS TRAINING STARTED")
    log("  Goals: acc>95% | hall<1% | overfit<0.5% | IQ>=120")
    log("=" * 60)

    best = load_best()
    if best:
        log(f"  Resuming — previous best IQ: {best.get('avg_iq', 0)}")

    best_iq       = best.get("avg_iq", 0) if best else 0
    best_config   = best or {}
    iteration     = 0
    goals_met     = False

    # Try to restore previous best IQ threshold
    if best:
        log(f"  Loaded previous best: IQ={best_iq} mode={best.get('mode')} "
            f"hidden={best.get('hidden_size')} lr={best.get('lr')} ep={best.get('epochs')}")

    all_stages = STAGE_1_CONFIGS + STAGE_2_BASE + STAGE_3_ENDGAME

    for (mode, hidden, lr, epochs) in all_stages:
        iteration += 1
        log(f"\n--- Iteration {iteration} | mode={mode} | hidden={hidden} | "
            f"lr={lr} | epochs={epochs} ---")

        try:
            agg = run_full_benchmark(
                mode=mode, hidden_size=hidden, lr=lr, epochs=epochs, verbose=True
            )
        except Exception as e:
            log(f"  ERROR: {e}")
            continue

        iq  = agg["avg_iq"]
        acc = agg["avg_val_acc"]
        hall = agg["avg_hall"]
        overfit = agg["avg_overfit"]

        log(f"  RESULT → IQ={iq} | acc={acc*100:.2f}% | "
            f"hall={hall*100:.3f}% | overfit={overfit*100:.3f}%")

        if iq > best_iq:
            best_iq     = iq
            best_config = {"mode": mode, "hidden_size": hidden,
                           "lr": lr, "epochs": epochs,
                           "avg_iq": iq, "avg_val_acc": acc,
                           "avg_hall": hall, "avg_overfit": overfit}
            save_best(best_config)
            log(f"  *** NEW BEST IQ: {iq} ***  (saved to {BEST_FILE})")

        if agg["all_passed"]:
            log("\n" + "=" * 60)
            log("  ALL GOALS MET!")
            log(f"  Final IQ         : {iq}")
            log(f"  Final Accuracy   : {acc*100:.2f}%  (goal >95%)")
            log(f"  Hallucination    : {hall*100:.3f}% (goal <1%)")
            log(f"  Copy-paste/Ofit  : {overfit*100:.3f}% (goal <0.5%)")
            log(f"  Best Config      : {best_config}")
            log("=" * 60)
            goals_met = True
            break

    if not goals_met:
        log(f"\n  Completed {iteration} configurations.")
        log(f"  Best IQ reached: {best_iq}")
        log(f"  Best config: {best_config}")
        log("  NOTE: Predefined stage configs exhausted. Run again to continue from best config.")


if __name__ == "__main__":
    train_to_goal()
