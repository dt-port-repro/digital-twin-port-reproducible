"""
Inference / prediction script for yard occupancy demand forecasting.
论文第五章 §5.1 堆场作业需求预测模型 — 推理接口

Loads a trained deep ensemble (K=5 models) and produces:
  - Mean forecast (7-day ahead)
  - 80% prediction interval [q10, q90]
  - Prediction Interval Coverage Probability (PICP)
  - Normalised interval width (PINRW)

Supports two loading modes:
  1. From saved checkpoint files (ensemble .pt files)
  2. Training-on-the-fly (fallback if no checkpoints exist)

Usage:
  python -m yard_prediction.inference --window W1
  python -m yard_prediction.inference --load-checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

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
from yard_prediction.train_predictor import (
    INPUT_WINDOW,
    FORECAST_STEPS,
    N_BLOCKS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    GCN_HIDDEN,
    GCN_OUT,
    N_ATTN_HEADS,
    DROPOUT,
    ENSEMBLE_SIZE,
    WALK_FORWARD_WINDOWS,
    RESULTS_DIR,
    MODELS_DIR,
    generate_synthetic_yard_data,
    prepare_sequences,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Ensemble Loading
# ──────────────────────────────────────────────────────

def load_ensemble_from_checkpoints(
    window_label: str = "W1",
    device: Optional[torch.device] = None,
) -> List[LSTMGNNAttentionPredictor]:
    """
    Load a deep ensemble from saved checkpoint files.

    Expects files at:
      03_results/canonical/yard_prediction_models/{window_label}/ensemble_seed*.pt

    Parameters
    ----------
    window_label : str — "W1" or "W2"

    Returns
    -------
    List of LSTMGNNAttentionPredictor instances (on CPU).

    Raises
    ------
    FileNotFoundError if no checkpoints are found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = MODELS_DIR / window_label
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"No ensemble checkpoints found at {checkpoint_dir}. "
            f"Run `train_predictor.py` first."
        )

    checkpoint_paths = sorted(checkpoint_dir.glob("ensemble_seed*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No ensemble_seed*.pt files at {checkpoint_dir}"
        )

    models: List[LSTMGNNAttentionPredictor] = []
    for ckpt_path in checkpoint_paths:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            model = LSTMGNNAttentionPredictor(
                input_window=config.get("input_window", INPUT_WINDOW),
                forecast_steps=config.get("forecast_steps", FORECAST_STEPS),
                n_blocks=config.get("n_blocks", N_BLOCKS),
                lstm_hidden=config.get("lstm_hidden", LSTM_HIDDEN),
                lstm_layers=config.get("lstm_layers", LSTM_LAYERS),
                gcn_hidden=config.get("gcn_hidden", GCN_HIDDEN),
                gcn_out=config.get("gcn_out", GCN_OUT),
                n_attn_heads=config.get("n_attn_heads", N_ATTN_HEADS),
                dropout=config.get("dropout", DROPOUT),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models.append(model)
            logger.info("Loaded model from %s (seed=%d)", ckpt_path, ckpt.get("seed", -1))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", ckpt_path, exc)

    if not models:
        raise RuntimeError("No models could be loaded from checkpoints.")

    logger.info("Loaded %d/%d ensemble models", len(models), ENSEMBLE_SIZE)
    return models


# ──────────────────────────────────────────────────────
# Ensemble Prediction
# ──────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_predict(
    models: List[LSTMGNNAttentionPredictor],
    X_time: np.ndarray,
    X_space: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce ensemble-averaged predictions.

    Parameters
    ----------
    models : list of trained LSTMGNNAttentionPredictor instances
    X_time : (n_samples, input_window, n_blocks * n_features)
    X_space : (n_samples, n_blocks, n_features)

    Returns
    -------
    mean_pred : (n_samples, forecast_steps) — ensemble mean
    q10_pred  : (n_samples, forecast_steps) — 10th percentile
    q90_pred  : (n_samples, forecast_steps) — 90th percentile
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    for m in models:
        m.to(device)
        m.eval()

    X_t = torch.from_numpy(X_time).to(device)
    X_s = torch.from_numpy(X_space).to(device)

    all_means, all_q10s, all_q90s = [], [], []
    for m in models:
        mean, q10, q90 = m(X_t, X_s)
        all_means.append(mean.cpu().numpy())
        all_q10s.append(q10.cpu().numpy())
        all_q90s.append(q90.cpu().numpy())

    mean_pred = np.mean(all_means, axis=0)
    q10_pred = np.mean(all_q10s, axis=0)
    q90_pred = np.mean(all_q90s, axis=0)

    return mean_pred, q10_pred, q90_pred


# ──────────────────────────────────────────────────────
# Single-Step Prediction Interface
# ──────────────────────────────────────────────────────

class YardDemandForecaster:
    """
    High-level wrapper for yard demand forecasting.

    Usage::

        forecaster = YardDemandForecaster.load(window="W1")
        mean, q10, q90 = forecaster.predict(new_data_14days)

    Where *new_data_14days* has shape (14, n_blocks, n_features).
    """

    def __init__(self, models: List[LSTMGNNAttentionPredictor]):
        self.models = models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def load(cls, window: str = "W1",
             train_if_missing: bool = False) -> "YardDemandForecaster":
        """
        Factory: load ensemble from checkpoints. Optionally train if missing.
        """
        try:
            models = load_ensemble_from_checkpoints(window_label=window)
        except (FileNotFoundError, RuntimeError):
            if train_if_missing:
                logger.info("No checkpoints found; training ensemble for %s ...", window)
                from yard_prediction.train_predictor import run_walk_forward
                run_walk_forward(window_key=window, verbose=False)
                models = load_ensemble_from_checkpoints(window_label=window)
            else:
                raise

        return cls(models)

    def predict(self,
                history: np.ndarray,
                return_interval: bool = True,
                ) -> Dict[str, np.ndarray]:
        """
        Predict next FORECAST_STEPS days of yard occupancy.

        Parameters
        ----------
        history : np.ndarray of shape (input_window, n_blocks, n_features)
            The last *input_window* days of data.

        Returns
        -------
        dict with keys:
            "mean" : (forecast_steps,)
            "q10"  : (forecast_steps,)
            "q90"  : (forecast_steps,)
        """
        assert history.shape == (INPUT_WINDOW, N_BLOCKS, 3), (
            f"Expected shape ({INPUT_WINDOW}, {N_BLOCKS}, 3), got {history.shape}"
        )

        # Prepare input (single sample)
        X_time = history.reshape(1, INPUT_WINDOW, -1).astype(np.float32)
        X_space = history[-1].reshape(1, N_BLOCKS, 3).astype(np.float32)  # last timestep

        mean_pred, q10_pred, q90_pred = ensemble_predict(
            self.models, X_time, X_space, device=self.device,
        )

        result = {
            "mean": mean_pred[0],
            "q10": q10_pred[0],
            "q90": q90_pred[0],
        }
        return result

    def evaluate(self,
                 X_time: np.ndarray,
                 X_space: np.ndarray,
                 y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble on a full test set.

        Returns
        -------
        dict with keys: loss, picp, pinrw
        """
        mean_pred, q10_pred, q90_pred = ensemble_predict(
            self.models, X_time, X_space, device=self.device,
        )

        t_mean = torch.from_numpy(mean_pred)
        t_q10 = torch.from_numpy(q10_pred)
        t_q90 = torch.from_numpy(q90_pred)
        t_target = torch.from_numpy(y_true)

        loss = total_quantile_loss(t_mean, t_q10, t_q90, t_target).item()
        picp = picp_metric(t_q10, t_q90, t_target)
        pinrw = pinrw_metric(t_q10, t_q90, t_target)

        return {"loss": loss, "picp": picp, "pinrw": pinrw}


# ──────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Yard demand prediction — inference (论文 §5.1)",
    )
    parser.add_argument(
        "--window", choices=["W1", "W2"], default="W1",
        help="Walk-forward window (default: W1)",
    )
    parser.add_argument(
        "--load-checkpoints", action="store_true",
        help="Load ensemble from saved checkpoints instead of training",
    )
    parser.add_argument(
        "--train-if-missing", action="store_true",
        help="Automatically train if no checkpoints exist",
    )
    parser.add_argument(
        "--data-seed", type=int, default=42,
        help="Random seed for synthetic data generation (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save prediction results JSON (default: 03_results/canonical/)",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Generate test data for the requested window
    cfg = WALK_FORWARD_WINDOWS[args.window]
    total_days = cfg["train_days"] + cfg["test_days"] + INPUT_WINDOW + FORECAST_STEPS
    logger.info("Generating synthetic data (%d days) for %s", total_days, args.window)
    raw_data = generate_synthetic_yard_data(n_days=total_days, seed=args.data_seed)

    X_time, X_space, y = prepare_sequences(raw_data)
    n_total = len(X_time)
    n_train = n_total - cfg["test_days"]

    X_test_t = X_time[n_train:]
    X_test_s = X_space[n_train:]
    y_test = y[n_train:]

    # Load or train forecaster
    if args.load_checkpoints:
        forecaster = YardDemandForecaster.load(
            window=args.window,
            train_if_missing=args.train_if_missing,
        )
    else:
        logger.info("Training ensemble from scratch for %s ...", args.window)
        from yard_prediction.train_predictor import run_walk_forward
        run_walk_forward(window_key=args.window, data_seed=args.data_seed, device=device, verbose=True)
        forecaster = YardDemandForecaster.load(window=args.window)

    # Evaluate on test set
    metrics = forecaster.evaluate(X_test_t, X_test_s, y_test)
    logger.info("=" * 50)
    logger.info("Inference Results — %s", args.window)
    logger.info("  Quantile Loss: %6f", metrics["loss"])
    logger.info("  PICP (80%% interval): %.4f", metrics["picp"])
    logger.info("  PINRW: %.4f", metrics["pinrw"])

    # Generate sample forecast
    sample_idx = 0
    sample_history = X_time[n_train + sample_idx].reshape(INPUT_WINDOW, N_BLOCKS, 3)
    forecast = forecaster.predict(sample_history)

    print("\nSample Forecast (next {} days):".format(FORECAST_STEPS))
    print(f"{'Day':<6} {'Mean':<10} {'q10':<10} {'q90':<10}")
    print("-" * 40)
    for d in range(FORECAST_STEPS):
        print(f"{d+1:<6} {forecast['mean'][d]:<10.4f} "
              f"{forecast['q10'][d]:<10.4f} {forecast['q90'][d]:<10.4f}")

    # Save results
    output_path = args.output
    if output_path is None:
        output_path = str(RESULTS_DIR / f"yard_prediction_{args.window}_inference.json")

    output_data = {
        "window": args.window,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "sample_forecast": {
            f"day_{d+1}": {
                "mean": round(float(forecast["mean"][d]), 6),
                "q10": round(float(forecast["q10"][d]), 6),
                "q90": round(float(forecast["q90"][d]), 6),
            }
            for d in range(FORECAST_STEPS)
        },
        "n_test_samples": len(y_test),
        "ensemble_size": ENSEMBLE_SIZE,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Inference results saved to %s", output_path)


if __name__ == "__main__":
    main()
