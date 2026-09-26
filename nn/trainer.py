"""
trainer.py — Training Loop
============================

THE TRAINING LOOP — THE HEART OF MACHINE LEARNING
----------------------------------------------------
Training a neural network is a loop that repeats thousands of times:

  For each EPOCH (full pass through the data):
    For each MINI-BATCH (small subset of data):
      1. FORWARD:   predict = network.forward(x_batch)
      2. LOSS:      loss, grad = compute_loss(y_batch, predict)
      3. BACKWARD:  network.backward(grad)      ← compute all gradients
      4. UPDATE:    optimizer.update(layers)    ← adjust all weights

After each epoch we evaluate on VALIDATION DATA to monitor overfitting.

KEY CONCEPTS
-------------
  EPOCH: One complete pass through the entire training dataset.
         More epochs = more learning, but eventually the model memorizes
         training data (overfitting) and gets worse on new data.

  MINI-BATCH: Instead of updating after each single example (slow) or
              after seeing ALL data (memory intensive), we update after
              seeing a small batch (e.g., 32 or 64 samples).
              This is called "mini-batch gradient descent".

  SHUFFLE: We shuffle the data before each epoch so the network doesn't
           memorize the ORDER of training examples.

  OVERFITTING: The model performs well on training data but poorly on
               unseen data. Like memorizing answers instead of understanding.
               Signs: training loss goes down, validation loss goes UP.

  EARLY STOPPING: Stop training when validation loss stops improving.
                  Prevents overfitting automatically.

  LEARNING RATE DECAY: Reduce the learning rate over time. Start with
                        big steps to find the rough minimum, then take
                        smaller steps to land precisely.
"""

import time
import numpy as np

from .network import NeuralNetwork
from .optimizer import Adam, get_optimizer


class Trainer:
    """
    Manages the full training loop for a NeuralNetwork.

    Parameters
    ----------
    network : NeuralNetwork
        The model to train.
    optimizer_name : str
        Which optimizer to use ('adam', 'sgd', 'sgd_momentum').
    learning_rate : float
        Starting learning rate.
    batch_size : int
        Number of samples per mini-batch.
    epochs : int
        Maximum number of training epochs.
    validation_split : float
        Fraction of training data to use for validation (0.0–0.5).
    early_stopping_patience : int
        Stop training if val_loss doesn't improve for this many epochs.
        Set to 0 to disable early stopping.
    lr_decay : float
        Multiply learning rate by this factor every epoch (e.g., 0.99).
        Set to 1.0 to disable decay.
    verbose : bool
        If True, print progress every epoch.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        optimizer_name: str = "adam",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.15,
        early_stopping_patience: int = 10,
        lr_decay: float = 0.995,
        clip_norm: float = 1.0,
        l2_lambda: float = 1e-4,
        verbose: bool = True,
    ):
        self.net          = network
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.val_split    = validation_split
        self.patience     = early_stopping_patience
        self.lr_decay     = lr_decay
        self.clip_norm    = clip_norm
        self.l2_lambda    = l2_lambda
        self.verbose      = verbose
        self.initial_lr   = learning_rate

        # Create optimizer instance
        self.optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)

    # --- Main Training Method ------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the network on the provided data.

        Parameters
        ----------
        X : np.ndarray, shape [n_samples, input_size]
            Training features.
        y : np.ndarray, shape [n_samples, output_size]
            Training labels.

        Returns
        -------
        dict
            Training history: {'loss', 'acc', 'val_loss', 'val_acc'}
        """
        # -- Split into train and validation -------------------------------
        n_val  = int(len(X) * self.val_split)
        n_train = len(X) - n_val

        # Shuffle before splitting
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train:], y[n_train:]

        if self.verbose:
            print(f"\n  Training samples  : {n_train:,}")
            print(f"  Validation samples: {n_val:,}")
            print(f"  Batch size        : {self.batch_size}")
            print(f"  Max epochs        : {self.epochs}")
            print(f"  Early stopping    : patience={self.patience}")
            print(f"  Learning rate     : {self.initial_lr}")
            self.net.summary()

        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

        # -- Early stopping state ------------------------------------------
        best_val_loss  = float("inf")
        best_weights   = None          # save best weights
        patience_count = 0             # how many epochs since last improvement

        start_time = time.time()

        # -- EPOCH LOOP ----------------------------------------------------
        for epoch in range(1, self.epochs + 1):

            # Shuffle training data at the start of each epoch
            perm = np.random.permutation(n_train)
            X_train, y_train = X_train[perm], y_train[perm]

            epoch_losses = []

            # -- MINI-BATCH LOOP -------------------------------------------
            self.net.set_training(True)
            for start in range(0, n_train, self.batch_size):
                end = start + self.batch_size

                x_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # -- 1. Forward pass ---------------------------------------
                predictions = self.net.forward(x_batch)

                # -- 2. Compute loss + initial gradient --------------------
                loss_val, grad = self.net.compute_loss(y_batch, predictions)
                epoch_losses.append(loss_val)

                # -- 3. Backward pass (compute all gradients) --------------
                self.net.backward(grad)

                # -- 4. Gradient Clipping & Update weights -----------------
                all_layers = self.net.hidden_layers[:self.net._active_layers]
                # all_layers += self.net.batch_norms[:self.net._active_layers]
                all_layers.append(self.net.output_layer)

                # Clip gradients to prevent exploding gradients in deep networks
                if self.clip_norm > 0:
                    total_norm = np.sqrt(sum(
                        np.sum(layer.dW ** 2) + np.sum(layer.db ** 2)
                        for layer in all_layers
                    ))
                    if total_norm > self.clip_norm:
                        scale = self.clip_norm / (total_norm + 1e-8)
                        for layer in all_layers:
                            layer.dW *= scale
                            layer.db *= scale

                # L2 weight decay: penalise large weights to prevent overfitting
                if self.l2_lambda > 0:
                    for layer in all_layers:
                        layer.dW += self.l2_lambda * layer.W

                self.optimizer.update(all_layers)

            # -- Epoch-level metrics ---------------------------------------
            self.net.set_training(False)  # MUST disable dropout for evaluation
            train_loss = float(np.mean(epoch_losses))
            train_acc  = self.net.accuracy(X_train, y_train)

            # Validation metrics (no gradient updates)
            val_preds     = self.net.forward(X_val)
            val_loss, _   = self.net.compute_loss(y_val, val_preds)
            val_acc       = self.net.accuracy(X_val, y_val)

            # Store in history
            history["loss"].append(train_loss)
            history["acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Also store in network for later access
            self.net.loss_history.append(train_loss)
            self.net.acc_history.append(train_acc)
            self.net.val_loss_history.append(val_loss)
            self.net.val_acc_history.append(val_acc)

            # -- Learning rate decay ---------------------------------------
            # Gradually reduce step size so we converge precisely
            if self.lr_decay < 1.0:
                self.optimizer.lr *= self.lr_decay

            # -- Early stopping check --------------------------------------
            if val_loss < best_val_loss - 1e-6:
                # Improvement! Save current weights.
                best_val_loss  = val_loss
                patience_count = 0
                best_weights   = self._snapshot_weights()
            else:
                patience_count += 1

            # -- Print progress --------------------------------------------
            if self.verbose:
                elapsed = time.time() - start_time
                self._print_progress(epoch, train_loss, train_acc, val_loss, val_acc, elapsed)

            # -- Early stop? -----------------------------------------------
            if self.patience > 0 and patience_count >= self.patience:
                print(f"\n  [STOP] Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {self.patience} epochs)")
                break

        # Restore best weights found during training
        if best_weights is not None:
            self._restore_weights(best_weights)
            print(f"\n  [OK] Restored best weights (val_loss={best_val_loss:.6f})")

        total_time = time.time() - start_time
        print(f"\n  [TIME] Training complete in {total_time:.1f}s")
        return history

    # --- Helpers -------------------------------------------------------------

    def _print_progress(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        elapsed: float,
    ) -> None:
        """Print one line of training progress."""
        bar_len = 20
        # Progress bar based on training loss (rough visual)
        mode = self.net.mode
        task = self.net.task

        if task == "regression":
            # For regression, don't show accuracy
            print(
                f"  Epoch {epoch:4d}/{self.epochs} | "
                f"loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"lr={self.optimizer.lr:.6f} | "
                f"[TIME]{elapsed:.1f}s"
            )
        else:
            print(
                f"  Epoch {epoch:4d}/{self.epochs} | "
                f"loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"lr={self.optimizer.lr:.6f}"
            )

    def _snapshot_weights(self) -> list[tuple]:
        """Copy current weights from all layers."""
        snapshot = []
        for layer in self.net.hidden_layers:
            snapshot.append((layer.W.copy(), layer.b.copy()))
        snapshot.append((self.net.output_layer.W.copy(), self.net.output_layer.b.copy()))
        return snapshot

    def _restore_weights(self, snapshot: list[tuple]) -> None:
        """Restore weights from a snapshot."""
        for i, layer in enumerate(self.net.hidden_layers):
            layer.W, layer.b = snapshot[i]
        self.net.output_layer.W, self.net.output_layer.b = snapshot[-1]
