"""
Training script for the LSTM–GNN–Attention yard prediction model.
论文第五章 §5.1 堆场作业需求预测模型 — 训练与验证

Implements two-window walk-forward validation (论文 §5.1.2):
  Window 1 (W1): 182 training days → 62 test days
  Window 2 (W2): 274 training days → 92 test days

Deep ensemble: K=5 models trained with different random seeds.
Early stopping patience = 10 epochs.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yard_prediction.lstm_gnn_attention import (
    LSTMGNNAttentionPredictor,
    total_quantile_loss,
    picp_metric,
    pinrw_metric,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# Constants (from paper §5.1.2)
# ──────────────────────────────────────────────────────
INPUT_WINDOW = 14            # days of history
FORECAST_STEPS = 7           # days to predict
N_BLOCKS = 8                 # yard blocks
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
GCN_HIDDEN = 64
GCN_OUT = 32
N_ATTN_HEADS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
MAX_EPOCHS = 200
ENSEMBLE_SIZE = 5            # K=5 models (deep ensemble paper §5.1.2)

# Walk-forward windows (paper §5.1.2)
WALK_FORWARD_WINDOWS = {
    "W1": {"train_days": 182, "test_days": 62},
    "W2": {"train_days": 274, "test_days": 92},
}

# Paths (relative to repo root)
RESULTS_DIR = _REPO_ROOT / "03_results" / "canonical"
MODELS_DIR = RESULTS_DIR / "yard_prediction_models"


# ──────────────────────────────────────────────────────
# Synthetic Data Generator (placeholder for real MCT data)
# ──────────────────────────────────────────────────────

def generate_synthetic_yard_data(
    n_days: int,
    n_blocks: int = N_BLOCKS,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic yard daily occupancy rate time series for testing.

    Mimics MCT 2024 yard daily occupancy rate with daily and weekly periodicity.

    Returns
    -------
    np.ndarray of shape (n_days, n_blocks, 3) — occupancy, vessel_arrival_factor,
    external_factor (e.g. weather / season).

    In production, replace this function with real data loading from
    01_data/yard_occupancy_mct_2024.csv.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_days)

    # Base occupancy with trend + weekly + daily patterns
    base = 0.55 + 0.05 * np.sin(2 * np.pi * t / 365)  # yearly seasonality
    weekly = 0.08 * np.sin(2 * np.pi * t / 7)           # weekly pattern
    noise = 0.03 * rng.randn(n_days)

    data = np.zeros((n_days, n_blocks, 3))
    for b in range(n_blocks):
        block_offset = 0.02 * b
        block_noise = 0.02 * rng.randn(n_days)
        occupancy = base + weekly + noise + block_offset + block_noise
        occupancy = np.clip(occupancy, 0.0, 1.0)

        # Vessel arrival factor (correlated with occupancy, lagged 1-2 days)
        vessel_factor = 0.3 + 0.4 * occupancy + 0.05 * rng.randn(n_days)

        # External factor (e.g., weather disruptions)
        ext_factor = 0.2 + 0.1 * np.sin(2 * np.pi * t / 30) + 0.03 * rng.randn(n_days)

        data[:, b, 0] = occupancy
        data[:, b, 1] = vessel_factor
        data[:, b, 2] = ext_factor

    return data


def prepare_sequences(
    data: np.ndarray,
    input_window: int = INPUT_WINDOW,
    forecast_steps: int = FORECAST_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw time series (n_days, n_blocks, n_features) into
    (X_time, X_space, y) supervised learning sequences.

    Returns
    -------
    X_time  : (n_samples, input_window, n_blocks * n_features)
    X_space : (n_samples, n_blocks, n_features)
    y       : (n_samples, forecast_steps) — target occupancy rates
              averaged across blocks.
    """
    n_days, n_blocks, n_features = data.shape
    X_time_list, X_space_list, y_list = [], [], []

    for i in range(input_window, n_days - forecast_steps + 1):
        # Time input: whole window, all blocks flattened
        seq = data[i - input_window:i]                     # (input_window, n_blocks, n_features)
        X_time_list.append(seq.reshape(input_window, -1))  # (input_window, n_blocks * n_features)

        # Space input: last timestep's per-block features
        X_space_list.append(data[i - 1])                   # (n_blocks, n_features)

        # Target: average occupancy across blocks for the forecast period
        target = data[i:i + forecast_steps, :, 0].mean(axis=1)  # (forecast_steps,)
        y_list.append(target)

    return (
        np.array(X_time_list, dtype=np.float32),
        np.array(X_space_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


# ──────────────────────────────────────────────────────
# One Model Training
# ──────────────────────────────────────────────────────

def train_single_model(
    seed: int,
    X_train_time: np.ndarray,
    X_train_space: np.ndarray,
    y_train: np.ndarray,
    X_val_time: np.ndarray,
    X_val_space: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> LSTMGNNAttentionPredictor:
    """
    Train a single LSTM–GNN–Attention predictor with given seed.
    Implements early stopping with patience = 10 (paper §5.1.2).

    Returns
    -------
    The trained model (on CPU) with best validation loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMGNNAttentionPredictor(
        input_window=INPUT_WINDOW,
        forecast_steps=FORECAST_STEPS,
        n_blocks=N_BLOCKS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        gcn_hidden=GCN_HIDDEN,
        gcn_out=GCN_OUT,
        n_attn_heads=N_ATTN_HEADS,
        dropout=DROPOUT,
    ).to(device)

    logger.info(
        "Model (seed=%d) parameters: %d",
        seed,
        model.count_parameters(),
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_time),
        torch.from_numpy(X_train_space),
        torch.from_numpy(y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_time),
        torch.from_numpy(X_val_space),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_Xt, batch_Xs, batch_y in train_loader:
            batch_Xt = batch_Xt.to(device)
            batch_Xs = batch_Xs.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            mean_pred, q10_pred, q90_pred = model(batch_Xt, batch_Xs)
            loss = total_quantile_loss(mean_pred, q10_pred, q90_pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_Xt.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_Xt, batch_Xs, batch_y in val_loader:
                batch_Xt = batch_Xt.to(device)
                batch_Xs = batch_Xs.to(device)
                batch_y = batch_y.to(device)
                mean_pred, q10_pred, q90_pred = model(batch_Xt, batch_Xs)
                loss = total_quantile_loss(mean_pred, q10_pred, q90_pred, batch_y)
                val_loss += loss.item() * batch_Xt.size(0)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0 and verbose:
            logger.info(
                "Epoch %3d | train_loss=%.6f | val_loss=%.6f",
                epoch, train_loss, val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.6f)",
                    epoch, best_val_loss,
                )
                break

    # Restore best state
    model.load_state_dict(best_state)
    model.to("cpu")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "seed": seed,
                "val_loss": best_val_loss,
                "config": {
                    "input_window": INPUT_WINDOW,
                    "forecast_steps": FORECAST_STEPS,
                    "n_blocks": N_BLOCKS,
                    "lstm_hidden": LSTM_HIDDEN,
                    "lstm_layers": LSTM_LAYERS,
                    "gcn_hidden": GCN_HIDDEN,
                    "gcn_out": GCN_OUT,
                    "n_attn_heads": N_ATTN_HEADS,
                    "dropout": DROPOUT,
                },
            },
            save_path,
        )
        logger.info("Model saved to %s", save_path)

    return model


# ──────────────────────────────────────────────────────
# Deep Ensemble Training   (论文 §5.1.2 — K=5)
# ──────────────────────────────────────────────────────

def train_ensemble(
    X_train_time: np.ndarray,
    X_train_space: np.ndarray,
    y_train: np.ndarray,
    X_val_time: np.ndarray,
    X_val_space: np.ndarray,
    y_val: np.ndarray,
    window_label: str = "W1",
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> List[LSTMGNNAttentionPredictor]:
    """
    Train an ensemble of K models with different random seeds.

    Models are saved to 03_results/canonical/yard_prediction_models/.

    Parameters
    ----------
    window_label : str — "W1" or "W2" for file naming.

    Returns
    -------
    List of trained models (on CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_seeds = [42 + i * 17 for i in range(ENSEMBLE_SIZE)]
    models: List[LSTMGNNAttentionPredictor] = []

    save_dir = MODELS_DIR / window_label
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, seed in enumerate(ensemble_seeds):
        logger.info("=" * 60)
        logger.info("Training ensemble model %d/%d (seed=%d)", i + 1, ENSEMBLE_SIZE, seed)

        save_path = save_dir / f"ensemble_seed{seed}.pt"
        model = train_single_model(
            seed=seed,
            X_train_time=X_train_time,
            X_train_space=X_train_space,
            y_train=y_train,
            X_val_time=X_val_time,
            X_val_space=X_val_space,
            y_val=y_val,
            device=device,
            save_path=save_path,
            verbose=verbose,
        )
        models.append(model)

    return models


# ──────────────────────────────────────────────────────
# Walk-Forward Validation   (论文 §5.1.2)
# ──────────────────────────────────────────────────────

def run_walk_forward(window_key: str = "W1",
                     data_seed: int = 42,
                     device: Optional[torch.device] = None,
                     verbose: bool = True) -> Dict[str, float]:
    """
    Run a single walk-forward window: generate synthetic data, train ensemble,
    evaluate on test set.

    Parameters
    ----------
    window_key : "W1" or "W2"

    Returns
    -------
    dict of metrics (mean_val_loss, test_loss, PICP, PINRW).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = WALK_FORWARD_WINDOWS[window_key]
    total_days = cfg["train_days"] + cfg["test_days"] + INPUT_WINDOW + FORECAST_STEPS

    logger.info("Generating synthetic data (%d days) for %s", total_days, window_key)
    raw_data = generate_synthetic_yard_data(n_days=total_days, seed=data_seed)

    X_time, X_space, y = prepare_sequences(raw_data)
    n_total = len(X_time)
    n_train = n_total - cfg["test_days"]

    X_train_t, X_val_t, X_test_t = (
        X_time[:n_train - FORECAST_STEPS],
        X_time[n_train - FORECAST_STEPS:n_train],
        X_time[n_train:],
    )
    X_train_s, X_val_s, X_test_s = (
        X_space[:n_train - FORECAST_STEPS],
        X_space[n_train - FORECAST_STEPS:n_train],
        X_space[n_train:],
    )
    y_train, y_val, y_test = (
        y[:n_train - FORECAST_STEPS],
        y[n_train - FORECAST_STEPS:n_train],
        y[n_train:],
    )

    logger.info(
        "%s — train: %d, val: %d, test: %d samples",
        window_key, len(y_train), len(y_val), len(y_test),
    )

    # Train ensemble
    models = train_ensemble(
        X_train_t, X_train_s, y_train,
        X_val_t, X_val_s, y_val,
        window_label=window_key,
        device=device,
        verbose=verbose,
    )

    # Evaluate ensemble on test set
    X_test_t_t = torch.from_numpy(X_test_t)
    X_test_s_t = torch.from_numpy(X_test_s)
    y_test_t = torch.from_numpy(y_test)

    all_means, all_q10s, all_q90s = [], [], []
    for m in models:
        m.to(device)
        m.eval()

    with torch.no_grad():
        for i in range(0, len(X_test_t_t), 64):
            batch_xt = X_test_t_t[i:i+64].to(device)
            batch_xs = X_test_s_t[i:i+64].to(device)

            means, q10s, q90s = [], [], []
            for m in models:
                m_pred, m_q10, m_q90 = m(batch_xt, batch_xs)
                means.append(m_pred.cpu())
                q10s.append(m_q10.cpu())
                q90s.append(m_q90.cpu())

            all_means.append(torch.stack(means).mean(dim=0))
            all_q10s.append(torch.stack(q10s).mean(dim=0))
            all_q90s.append(torch.stack(q90s).mean(dim=0))

    mean_pred = torch.cat(all_means, dim=0)
    q10_pred = torch.cat(all_q10s, dim=0)
    q90_pred = torch.cat(all_q90s, dim=0)

    test_loss = total_quantile_loss(mean_pred, q10_pred, q90_pred, y_test_t).item()
    test_picp = picp_metric(q10_pred, q90_pred, y_test_t)
    test_pinrw = pinrw_metric(q10_pred, q90_pred, y_test_t)

    logger.info("=" * 60)
    logger.info("%s Test Results:", window_key)
    logger.info("  Quantile Loss: %.6f", test_loss)
    logger.info("  PICP (80%% interval): %.4f", test_picp)
    logger.info("  PINRW: %.4f", test_pinrw)

    results = {
        "window": window_key,
        "test_loss": round(test_loss, 6),
        "picp": round(test_picp, 4),
        "pinrw": round(test_pinrw, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "ensemble_size": ENSEMBLE_SIZE,
    }

    # Save results
    results_path = RESULTS_DIR / f"yard_prediction_{window_key}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


# ──────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────

def main():
    """Run both walk-forward windows (W1, W2)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress minor warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    all_results = {}
    for wkey in ["W1", "W2"]:
        logger.info("\n\n%s %s %s", "=" * 25, wkey, "=" * 25)
        try:
            res = run_walk_forward(window_key=wkey, device=device, verbose=True)
            all_results[wkey] = res
        except Exception as exc:
            logger.error("Walk-forward %s failed: %s", wkey, exc)
            all_results[wkey] = {"error": str(exc)}

    # Save combined summary
    summary_path = RESULTS_DIR / "yard_prediction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Print table
    print("\n")
    print(f"{'Window':<8} {'Test Loss':<12} {'PICP':<8} {'PINRW':<8}")
    print("-" * 40)
    for wkey, res in all_results.items():
        if "error" in res:
            print(f"{wkey:<8} ERROR: {res['error']}")
        else:
            print(
                f"{wkey:<8} {res['test_loss']:<12.6f} "
                f"{res['picp']:<8.4f} {res['pinrw']:<8.4f}"
            )


if __name__ == "__main__":
    main()
