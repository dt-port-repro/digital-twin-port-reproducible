"""
实验运行器与对比基线（论文 §5.4 实验设计）

Implements experiment runners for §5.4 evaluation:
  - Three-stage allocation vs baseline (FCFS + random)
  - Multi-scenario execution (常规作业, 高峰压力, 异常情况)
  - PPO coordinated vs fixed-weight allocation
  - Results are saved to 03_results/canonical/

Usage:
    python -m yard_optimization.run_experiments            # run all
    python -m yard_optimization.run_experiments --scenario W1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# repo root for sibling imports
# ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yard_optimization.three_stage_allocation import (
    ThreeStageAllocator,
    AllocationConfig,
    ContainerInfo,
    YardLayout,
    FeasiblePosition,
    create_default_yard_layout,
    container_info_from_dict,
)

logger = logging.getLogger(__name__)

# Output directory
RESULTS_DIR = _REPO_ROOT / "03_results" / "canonical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Scenario definitions (论文 §6.2.2)
# ──────────────────────────────────────────────────────────────────

SCENARIOS = {
    "常规作业": {
        "n_containers": 2000,
        "yard_util_init": 0.65,
        "arrival_rate": 1.0,       # relative
        "weight_variation": 0.2,
    },
    "高峰压力": {
        "n_containers": 4000,
        "yard_util_init": 0.80,
        "arrival_rate": 1.8,
        "weight_variation": 0.3,
    },
    "异常情况": {
        "n_containers": 3000,
        "yard_util_init": 0.75,
        "arrival_rate": 1.3,
        "weight_variation": 0.5,
    },
}


# ──────────────────────────────────────────────────────────────────
# Baseline: FCFS + random allocation (论文 §5.4 baseline)
# ──────────────────────────────────────────────────────────────────

def baseline_fcfs_random(
    yard: YardLayout,
    containers: List[ContainerInfo],
    config: AllocationConfig,
    rng: np.random.RandomState,
) -> Tuple[List[FeasiblePosition], Dict[str, float]]:
    """
    Baseline allocation: FCFS + random feasible position.

    Simulates a naive yard allocation strategy where each container
    is placed at the first feasible position found (first-come,
    first-served), choosing randomly among the feasible set.

    Args:
        yard: Yard layout.
        containers: List of containers in arrival order.
        config: Allocation config.
        rng: Random state for reproducibility.

    Returns:
        (positions, metrics)
    """
    allocator = ThreeStageAllocator(yard, config)
    positions: List[FeasiblePosition] = []
    total_penalties = []

    for container in containers:
        feasible = allocator.stage1_screening(container, {})
        if not feasible:
            # No feasible position — place anywhere
            positions.append(
                FeasiblePosition(block_id=1, bay=1, row=0, tier=0)
            )
            continue

        # Random choice from feasible set
        pos = rng.choice(feasible)
        positions.append(pos)
        total_penalties.append(pos.total_penalty)

    metrics = {
        "mean_penalty": float(np.mean(total_penalties)) if total_penalties else 0.0,
        "total_penalty": float(np.sum(total_penalties)) if total_penalties else 0.0,
        "n_success": len(positions),
        "n_total": len(containers),
    }
    return positions, metrics


# ──────────────────────────────────────────────────────────────────
# Three-stage allocation (论文 §5.2.4)
# ──────────────────────────────────────────────────────────────────

def run_three_stage(
    yard: YardLayout,
    containers: List[ContainerInfo],
    config: AllocationConfig,
    ppo_weights: Optional[np.ndarray] = None,
) -> Tuple[List[FeasiblePosition], Dict[str, float]]:
    """
    Run the full three-stage allocation on a batch of containers.

    Args:
        yard: Yard layout.
        containers: List of containers in arrival order.
        config: Allocation config.
        ppo_weights: Optional 6-dim PPO weight scaling factors.

    Returns:
        (positions, metrics)
    """
    allocator = ThreeStageAllocator(yard, config)
    positions: List[FeasiblePosition] = []
    times = []
    total_penalties = []

    for container in containers:
        t0 = time.perf_counter()
        try:
            pos = allocator.allocate(container, {}, ppo_weights)
            elapsed = time.perf_counter() - t0
            positions.append(pos)
            total_penalties.append(pos.total_penalty)
            times.append(elapsed)
        except RuntimeError as e:
            logger.warning("Allocation failed: %s", e)
            positions.append(
                FeasiblePosition(block_id=1, bay=1, row=0, tier=0)
            )

    metrics = {
        "mean_penalty": float(np.mean(total_penalties)) if total_penalties else 0.0,
        "total_penalty": float(np.sum(total_penalties)) if total_penalties else 0.0,
        "mean_time_ms": float(np.mean(times)) * 1000 if times else 0.0,
        "p99_time_ms": float(np.percentile(times, 99)) * 1000 if times else 0.0,
        "n_success": len(positions),
        "n_total": len(containers),
        "n_feasible_all": sum(1 for p in positions if p.hard_constraints_ok),
    }
    return positions, metrics


# ──────────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────────

def generate_containers(
    scenario: str,
    yard: YardLayout,
    n_containers: Optional[int] = None,
    seed: int = 42,
) -> List[ContainerInfo]:
    """
    Generate a batch of containers for the given scenario.

    Args:
        scenario: Scenario name (常规作业, 高峰压力, 异常情况).
        yard: Yard layout (for reference).
        n_containers: Override number of containers.
        seed: Random seed.

    Returns:
        List of ContainerInfo objects.
    """
    params = SCENARIOS[scenario]
    rng = np.random.RandomState(seed)
    n = n_containers if n_containers is not None else params["n_containers"]

    types = ["general", "reefer", "dangerous"]
    type_probs = [0.85, 0.10, 0.05]  # typical distribution
    ports = ["SHA", "NING", "XMN", "YTN", "SIN"]

    containers = []
    for i in range(n):
        ctype = rng.choice(types, p=type_probs)
        length_20ft = rng.random() < 0.7  # 70% 20ft
        weight = 5.0 + rng.exponential(10.0)  # ~15t mean
        port = rng.choice(ports)
        berth = rng.randint(1, 4)

        containers.append(
            ContainerInfo(
                container_id=f"CONT_{scenario[:2]}_{i:06d}",
                length_20ft=length_20ft,
                weight_t=weight,
                discharge_port=port,
                container_type=ctype,
                destination_berth=berth,
                vessel_code=f"VSL_{scenario[:2]}_{i // 500}",
            )
        )

    logger.info("Generated %d containers for scenario '%s'", n, scenario)
    return containers


def run_single_experiment(
    scenario: str,
    method: str,
    seed: int = 42,
    ppo_weights: Optional[np.ndarray] = None,
    n_containers: Optional[int] = None,
) -> Dict:
    """
    Run a single experiment configuration.

    Args:
        scenario: Scenario name.
        method: 'three_stage' or 'baseline'.
        seed: Random seed.
        ppo_weights: Optional PPO weight scalings.
        n_containers: Override container count.

    Returns:
        Dict with experiment results.
    """
    config = AllocationConfig()
    yard = create_default_yard_layout()
    containers = generate_containers(scenario, yard, n_containers, seed)
    rng = np.random.RandomState(seed)

    t0 = time.perf_counter()

    if method == "baseline":
        positions, metrics = baseline_fcfs_random(yard, containers, config, rng)
    else:
        positions, metrics = run_three_stage(yard, containers, config, ppo_weights)

    elapsed = time.perf_counter() - t0

    result = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "n_containers": len(containers),
        "elapsed_s": round(elapsed, 3),
        **metrics,
    }

    logger.info(
        "Experiment done: scenario=%s method=%s penalty=%.4f time=%.2fs",
        scenario, method, metrics.get("mean_penalty", 0), elapsed,
    )
    return result


# ──────────────────────────────────────────────────────────────────
# PPO-coordinated experiment (论文 §5.3.2)
# ──────────────────────────────────────────────────────────────────

def run_ppo_coordinated_experiment(
    scenario: str,
    ppo_weights: np.ndarray,
    seed: int = 42,
    n_containers: Optional[int] = None,
) -> Dict:
    """
    Run experiment with PPO-coordinated weight scaling.

    The PPO weights are applied as multiplicative factors on the
    Stage 2 penalty components (§5.3.2).

    Args:
        scenario: Scenario name.
        ppo_weights: 6-dim weight scaling factors.
        seed: Random seed.
        n_containers: Override container count.

    Returns:
        Dict with experiment results.
    """
    return run_single_experiment(
        scenario=scenario,
        method="three_stage",
        seed=seed,
        ppo_weights=ppo_weights,
        n_containers=n_containers,
    )


# ──────────────────────────────────────────────────────────────────
# Multi-scenario experiment suite (论文 §5.4)
# ──────────────────────────────────────────────────────────────────

def run_experiments(
    scenarios: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_containers: Optional[int] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the full experiment suite for all scenarios and methods.

    Results are returned as a DataFrame and optionally saved to
    03_results/canonical/yard_optimization_experiments.parquet.

    Args:
        scenarios: List of scenario names (default: all).
        seeds: List of seeds (default: [42]).
        n_containers: Override container count.
        save: Whether to save results to parquet.

    Returns:
        DataFrame of results.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())
    if seeds is None:
        seeds = [42]

    methods = ["baseline", "three_stage"]

    # PPO weight vectors from paper results (§5.3.2 Table)
    # W1 learned "prioritize yard" strategy → weights biased toward
    # rehandle reduction and space utilization
    ppo_weights_w1 = np.array([0.8, 0.9, 1.2, 1.1, 1.3, 0.7])
    # W2: 95.3% yard + 4.7% balanced
    ppo_weights_w2 = np.array([0.9, 1.0, 1.1, 1.0, 1.2, 0.8])

    all_results = []

    for scenario in scenarios:
        for seed in seeds:
            for method in methods:
                result = run_single_experiment(
                    scenario=scenario,
                    method=method,
                    seed=seed,
                    n_containers=n_containers,
                )
                all_results.append(result)

        # PPO-coordinated experiments
        for window_name, ppo_w in [("W1", ppo_weights_w1), ("W2", ppo_weights_w2)]:
            result = run_ppo_coordinated_experiment(
                scenario=scenario,
                ppo_weights=ppo_w,
                seed=seeds[0] if seeds else 42,
                n_containers=n_containers,
            )
            result["method"] = f"ppo_{window_name}"
            all_results.append(result)

    df = pd.DataFrame(all_results)

    if save:
        out_path = RESULTS_DIR / "yard_optimization_experiments.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Results saved to %s", out_path)

    return df


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Yard optimization experiments (§5.4)"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
        help="Random seeds (default: 42 43 44)",
    )
    parser.add_argument(
        "--n-containers", type=int, default=None,
        help="Override number of containers",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save results to parquet",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    scenarios = (
        list(SCENARIOS.keys())
        if args.scenario in (None, "all")
        else [args.scenario]
    )

    logger.info("Starting experiments: scenarios=%s seeds=%s", scenarios, args.seeds)

    df = run_experiments(
        scenarios=scenarios,
        seeds=args.seeds,
        n_containers=args.n_containers,
        save=not args.no_save,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT SUMMARY")
    print("=" * 72)
    summary = (
        df.groupby(["scenario", "method"])["mean_penalty"]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
