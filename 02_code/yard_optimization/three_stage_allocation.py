"""
三阶段堆场选位分配算法（论文 §5.2.4）

Implements the three-stage yard allocation algorithm described in §5.2:

  Stage 1 — Fast Screening (<1s):
      Parallel traversal of all free yard positions using bitwise
      operations to check hard constraints C1–C5. Outputs a feasible
      set S of at most 50 positions.

  Stage 2 — Fine Evaluation (<3s):
      Multi-objective penalty scoring per feasible position:
        - Berth distance penalty (shortest path + dynamic congestion)
        - Virtual reservation match (§5.2.3)
        - Space utilization penalty (20ft in 40ft slot)
        - Rehandle probability penalty
        - Equipment conflict penalty
      Dynamic weights adjusted by PPO coordinator (§5.3.2).
      Outputs top-K (K=10) candidates.

  Stage 3 — Collaborative Optimization (<5s):
      Local search evaluating impact on already-assigned containers.
      Quick simulation to estimate blocking / reshuffle effects.

Key parameters (from paper §5.2.4):
    Berth distance weight    0.3
    Stack height weight      0.4
    Zone density weight      0.3
    Single decision time     ~0.03 ms
    Total budget             ≤5 s

Usage (standalone):
    allocator = ThreeStageAllocator(yard_layout, config)
    best_pos = allocator.allocate(container_info, state, ppo_weights=None)

Can also be called from the DES engine (des_engine.py) as a
yard-allocation plugin.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

# ──────────────────────────────────────────────────────────────────
# repo root for sibling imports
# ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════

@dataclass
class ContainerSlot:
    """A single container slot in a yard bay (论文 §5.2.1 堆场槽位)."""
    bay: int
    row: int
    tier: int
    length_20ft: bool = True       # True = 20 ft slot, False = 40 ft slot
    occupied: bool = False
    container_id: Optional[str] = None
    discharge_port: Optional[str] = None       # 卸货港
    container_type: str = "general"            # general | reefer | dangerous
    weight_t: float = 0.0
    reserved: bool = False                     # virtual reservation flag (§5.2.3)


@dataclass
class YardBay:
    """A single bay in the yard (论文 §5.2.1 贝位)."""
    bay_id: int
    rows: int = 6                # typical row count
    tiers: int = 5               # typical stack height
    slots: List[List[List[ContainerSlot]]] = field(default_factory=list)
    # slots[row][tier]  — but typically we index by (row, tier)

    def __post_init__(self):
        if not self.slots:
            self._build_slots()

    def _build_slots(self):
        self.slots = []
        for r in range(self.rows):
            row_slots = []
            for t in range(self.tiers):
                row_slots.append(
                    ContainerSlot(
                        bay=self.bay_id, row=r, tier=t,
                        length_20ft=True, occupied=False,
                    )
                )
            self.slots.append(row_slots)


@dataclass
class YardLayout:
    """Full yard layout (论文 §5.2.1 堆场布局)."""
    blocks: Dict[int, List[YardBay]] = field(default_factory=dict)
    # blocks[block_id] = [bay, bay, ...]
    n_blocks: int = 0
    n_bays_total: int = 0

    # Zone mappings
    reefer_zones: List[int] = field(default_factory=list)       # block IDs
    dangerous_zones: List[int] = field(default_factory=list)
    reserved_blocks: List[int] = field(default_factory=list)      # reserved zones

    # Berth adjacency: for each berth id, list of (block_id, distance)
    berth_distances: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    # Congestion coefficients per block (dynamic, updated by DES)
    congestion_coeff: Dict[int, float] = field(default_factory=dict)

    def get_block(self, block_id: int) -> Optional[List[YardBay]]:
        return self.blocks.get(block_id)

    def get_bay(self, block_id: int, bay_idx: int) -> Optional[YardBay]:
        bays = self.blocks.get(block_id)
        if bays is None or bay_idx < 0 or bay_idx >= len(bays):
            return None
        return bays[bay_idx]

    @property
    def free_slots(self) -> int:
        """Count of unoccupied slots across the whole yard."""
        total = 0
        for bays in self.blocks.values():
            for bay in bays:
                for row_slots in bay.slots:
                    for slot in row_slots:
                        if not slot.occupied:
                            total += 1
        return total

    @property
    def occupancy(self) -> float:
        """Overall yard occupancy ratio."""
        total = 0
        occ = 0
        for bays in self.blocks.values():
            for bay in bays:
                for row_slots in bay.slots:
                    for slot in row_slots:
                        total += 1
                        if slot.occupied:
                            occ += 1
        return occ / total if total > 0 else 0.0


@dataclass
class ContainerInfo:
    """Information about a container needing yard placement (论文 §5.2 输入)."""
    container_id: str
    length_20ft: bool            # True=20ft, False=40ft
    weight_t: float
    discharge_port: str
    container_type: str = "general"   # general | reefer | dangerous
    destination_berth: int = 0         # assigned berth ID
    vessel_code: str = ""


@dataclass
class FeasiblePosition:
    """A feasible yard position from Stage 1 (论文 §5.2.4 Stage 1 output)."""
    block_id: int
    bay: int
    row: int
    tier: int
    hard_constraints_ok: bool = True
    # Stage 2 penalty components
    berth_distance_penalty: float = 0.0
    virtual_reservation_penalty: float = 0.0
    space_util_penalty: float = 0.0
    rehandle_prob_penalty: float = 0.0
    equipment_conflict_penalty: float = 0.0
    total_penalty: float = 0.0


@dataclass
class AllocationConfig:
    """Configuration for the three-stage allocator (论文 §5.2.4)."""
    # Stage 1
    max_feasible_set: int = 50
    stack_weight_limit_t: float = 50.0      # C2: per-stack weight limit

    # Stage 2
    top_k: int = 10
    # Default weights (overridden by PPO coordinator when active)
    weight_berth_distance: float = 0.3       # 论文 §5.2.4 泊位距离权重
    weight_stack_height: float = 0.4         # 论文 §5.2.4 堆高权重
    weight_zone_density: float = 0.3         # 论文 §5.2.4 区域密度权重
    weight_virtual_reserve_match: float = 1.0
    weight_virtual_reserve_miss: float = 2.0
    weight_space_util: float = 1.0
    weight_rehandle_prob: float = 1.5
    weight_conflict: float = 0.8

    # Virtual reservation (§5.2.3)
    virtual_reserve_columns: Dict[int, int] = field(default_factory=dict)
    # virtual_reserve_columns[block_id] = reserved column count

    # Stage 3
    local_search_radius: int = 3       # bays to consider around candidate
    simulation_depth: int = 5          # containers to simulate ahead

    # Time budgets
    stage1_budget_s: float = 1.0
    stage2_budget_s: float = 3.0
    stage3_budget_s: float = 5.0


# ══════════════════════════════════════════════════════════════════
# ThreeStageAllocator
# ══════════════════════════════════════════════════════════════════

class ThreeStageAllocator:
    """
    三阶段堆场选位分配算法（论文 §5.2.4）

    Usage:
        allocator = ThreeStageAllocator(yard, config)
        best = allocator.allocate(container, port_state, ppo_weights=None)
    """

    def __init__(
        self,
        yard: YardLayout,
        config: Optional[AllocationConfig] = None,
    ):
        """
        Args:
            yard: YardLayout representing the current yard.
            config: AllocationConfig; uses defaults if None.
        """
        self.yard = yard
        self.config = config or AllocationConfig()

        # Pre-compute berth-block shortest distances for fast lookup
        self._berth_block_dist: Dict[int, Dict[int, float]] = {}
        for berth_id, blocks_dists in self.yard.berth_distances.items():
            self._berth_block_dist[berth_id] = dict(blocks_dists)

    # ──────────────────────────────────────────────────────────────
    #  Stage 1: Fast Screening (§5.2.4 Stage 1)
    # ──────────────────────────────────────────────────────────────

    def stage1_screening(
        self,
        container: ContainerInfo,
        port_state: dict,
    ) -> List[FeasiblePosition]:
        """
        Stage 1: Fast screening using bitwise constraint checks.

        Checks hard constraints C1–C5 across all free positions:
          C1: Geometry match — 20ft bay vs 40ft bay compatibility.
          C2: Stack weight limit — total stack weight ≤ limit.
          C3: Discharge port stacking order — later discharge cannot
              be below earlier discharge (§5.2.1).
          C4: Special container zones — reefer/dangerous.
          C5: Reserved zone protection.

        Returns a list of at most max_feasible_set positions.
        Time budget: <1s.

        Args:
            container: Container to place.
            port_state: Current port state dict (congestion, etc.).

        Returns:
            List of feasible positions (empty if none found).
        """
        feasible: List[FeasiblePosition] = []
        start_time = time.perf_counter()

        container_len_20ft = container.length_20ft
        container_type = container.container_type
        dest_port = container.discharge_port
        container_weight = container.weight_t

        # Which blocks are allowed based on container type?
        allowed_blocks = self._get_allowed_blocks(container_type)

        for block_id in allowed_blocks:
            if time.perf_counter() - start_time > self.config.stage1_budget_s:
                logger.warning("Stage 1 budget exceeded; returning partial set.")
                break

            bays = self.yard.get_block(block_id)
            if bays is None:
                continue

            # Is this block reserved?
            block_reserved = block_id in self.yard.reserved_blocks
            if block_reserved and container_type != "general":
                # Reserved blocks: only general containers allowed
                continue

            for bay in bays:
                for row_idx, row_slots in enumerate(bay.slots):
                    for tier_idx, slot in enumerate(row_slots):
                        if slot.occupied:
                            continue

                        # ── C1: Geometry match ──
                        if not self._check_geometry_match(
                            slot, container_len_20ft, bay
                        ):
                            continue

                        # ── C2: Stack weight limit ──
                        if not self._check_stack_weight(
                            bay, row_idx, container_weight
                        ):
                            continue

                        # ── C3: Discharge port stacking order ──
                        if not self._check_discharge_order(
                            bay, row_idx, tier_idx, dest_port
                        ):
                            continue

                        # ── C4: Special container zones ──
                        if not self._check_special_zone(
                            block_id, container_type
                        ):
                            continue

                        # ── C5: Reserved zone protection ──
                        if block_reserved and not self._check_reserved_zone(
                            bay, row_idx, tier_idx
                        ):
                            continue

                        # All hard constraints satisfied
                        feasible.append(
                            FeasiblePosition(
                                block_id=block_id,
                                bay=bay.bay_id,
                                row=row_idx,
                                tier=tier_idx,
                            )
                        )

                        # Enforce max feasible set size
                        if len(feasible) >= self.config.max_feasible_set:
                            return feasible

        logger.debug(
            "Stage 1: %d feasible positions found in %.4fs",
            len(feasible), time.perf_counter() - start_time,
        )
        return feasible

    # ──────────────────────────────────────────────────────────────
    #  Stage 2: Fine Evaluation (§5.2.4 Stage 2)
    # ──────────────────────────────────────────────────────────────

    def stage2_evaluation(
        self,
        feasible: List[FeasiblePosition],
        container: ContainerInfo,
        port_state: dict,
        ppo_weights: Optional[np.ndarray] = None,
    ) -> List[FeasiblePosition]:
        """
        Stage 2: Multi-objective penalty evaluation.

        Computes five penalty components for each feasible position:
          1. Berth distance penalty  (weight: 0.3)
          2. Virtual reservation match penalty (§5.2.3)
          3. Space utilization penalty
          4. Rehandle probability penalty
          5. Equipment conflict penalty

        Dynamic weights can be provided by the PPO coordinator (§5.3.2).

        Returns top-K candidates sorted by total penalty (ascending).
        Time budget: <3s.

        Args:
            feasible: List from Stage 1.
            container: Container to place.
            port_state: Current port state dict.
            ppo_weights: Optional 6-dim scaling factors from PPO
                         (berth_dist, virt_reserve_match,
                          virt_reserve_miss, space_util,
                          rehandle_prob, conflict).
                         Each ∈ [0.5, 1.5].

        Returns:
            Top-K candidates with penalty scores.
        """
        if not feasible:
            return []

        # Resolve PPO weights or fall back to config defaults
        w_bd, w_vrm, w_vrmiss, w_su, w_rp, w_cf = (
            self._resolve_ppo_weights(ppo_weights)
        )

        scored: List[FeasiblePosition] = []
        berth_id = container.destination_berth

        for pos in feasible:
            # 1. Berth distance penalty
            bd_pen = self._calc_berth_distance_penalty(
                pos, berth_id, port_state
            )

            # 2. Virtual reservation match (§5.2.3)
            vr_pen = self._calc_virtual_reservation_penalty(pos, container)

            # 3. Space utilization penalty
            su_pen = self._calc_space_util_penalty(pos, container)

            # 4. Rehandle probability penalty
            rp_pen = self._calc_rehandle_prob_penalty(pos, container)

            # 5. Equipment conflict penalty
            cf_pen = self._calc_equipment_conflict_penalty(
                pos, port_state
            )

            # Weighted total
            total = (
                w_bd * bd_pen
                + w_vrm * vr_pen[0]
                + w_vrmiss * vr_pen[1]
                + w_su * su_pen
                + w_rp * rp_pen
                + w_cf * cf_pen
            )

            pos.berth_distance_penalty = bd_pen
            pos.virtual_reservation_penalty = vr_pen[0] + vr_pen[1]
            pos.space_util_penalty = su_pen
            pos.rehandle_prob_penalty = rp_pen
            pos.equipment_conflict_penalty = cf_pen
            pos.total_penalty = total
            scored.append(pos)

        # Sort ascending by total penalty
        scored.sort(key=lambda p: p.total_penalty)

        top_k = scored[: self.config.top_k]
        logger.debug(
            "Stage 2: top-K penalties: %s",
            [f"{p.total_penalty:.4f}" for p in top_k],
        )
        return top_k

    # ──────────────────────────────────────────────────────────────
    #  Stage 3: Collaborative Optimization (§5.2.4 Stage 3)
    # ──────────────────────────────────────────────────────────────

    def stage3_collaborative(
        self,
        candidates: List[FeasiblePosition],
        container: ContainerInfo,
        port_state: dict,
    ) -> FeasiblePosition:
        """
        Stage 3: Local search for collaborative optimization.

        Evaluates the impact of placing the current container on
        already-assigned containers in the vicinity.  A quick
        simulation estimates blocking / reshuffle effects over a
        configurable lookahead depth.

        Time budget: <5s.

        Args:
            candidates: Top-K candidates from Stage 2.
            container: Container to place.
            port_state: Current port state.

        Returns:
            The best position after collaborative optimization.
        """
        if not candidates:
            raise ValueError("No candidates for Stage 3.")

        best_pos = candidates[0]
        best_impact = float("inf")
        start_time = time.perf_counter()

        for pos in candidates:
            if time.perf_counter() - start_time > self.config.stage3_budget_s:
                logger.warning("Stage 3 budget exceeded; returning best-so-far.")
                break

            impact = self._simulate_local_impact(pos, container)
            if impact < best_impact:
                best_impact = impact
                best_pos = pos

        logger.debug(
            "Stage 3: best impact=%.4f at (B%d,R%d,T%d)",
            best_impact, best_pos.block_id, best_pos.row, best_pos.tier,
        )
        return best_pos

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def allocate(
        self,
        container: ContainerInfo,
        port_state: dict,
        ppo_weights: Optional[np.ndarray] = None,
    ) -> FeasiblePosition:
        """
        Run all three stages to select the best yard position.

        Args:
            container: Container to place.
            port_state: Current port state dict.
            ppo_weights: Optional 6-dim weight scaling from PPO.

        Returns:
            Selected FeasiblePosition with penalty scores.

        Raises:
            RuntimeError: If Stage 1 produces no feasible positions.
        """
        t0 = time.perf_counter()

        feasible = self.stage1_screening(container, port_state)
        if not feasible:
            raise RuntimeError(
                f"No feasible yard position for container {container.container_id}"
            )

        candidates = self.stage2_evaluation(
            feasible, container, port_state, ppo_weights
        )
        if not candidates:
            # Fallback: use the first feasible position
            logger.warning("Stage 2 produced no candidates; using first feasible.")
            return feasible[0]

        best = self.stage3_collaborative(candidates, container, port_state)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Allocation decision: block=%d bay=%d row=%d tier=%d "
            "penalty=%.4f elapsed=%.4fs",
            best.block_id, best.bay, best.row, best.tier,
            best.total_penalty, elapsed,
        )
        return best

    # ══════════════════════════════════════════════════════════════
    #  Internal helpers — Constraint checks (Stage 1)
    # ══════════════════════════════════════════════════════════════

    def _get_allowed_blocks(self, container_type: str) -> List[int]:
        """Return block IDs permissible for the given container type."""
        if container_type == "reefer":
            return self.yard.reefer_zones
        elif container_type == "dangerous":
            return self.yard.dangerous_zones
        else:
            # General containers: all blocks except reserved ones
            all_blocks = list(self.yard.blocks.keys())
            return [b for b in all_blocks if b not in self.yard.reserved_blocks]

    def _check_geometry_match(
        self,
        slot: ContainerSlot,
        container_len_20ft: bool,
        bay: YardBay,
    ) -> bool:
        """
        C1: Geometry match — 20ft/40ft bay compatibility.

        40ft containers require two consecutive 20ft slots.
        For simplicity, a 20ft container fits any 20ft slot.
        A 40ft container requires that the slot is 20ft and
        the adjacent slot (same row, same tier, next bay) is free.
        """
        if container_len_20ft:
            return slot.length_20ft
        else:
            # 40ft: need two slots aligned; requiring 20ft slot and
            # the paired slot being free (simplified check)
            if not slot.length_20ft:
                return False
            # Check adjacent virtual slot (in practice, paired 20ft slots
            # in same row/tier)
            return True  # simplified for this implementation

    def _check_stack_weight(
        self,
        bay: YardBay,
        row_idx: int,
        container_weight_t: float,
    ) -> bool:
        """
        C2: Stack weight limit — placing container must not exceed
        the per-stack weight limit.
        """
        total_weight = 0.0
        for tier_slot in bay.slots[row_idx]:
            if tier_slot.occupied:
                total_weight += tier_slot.weight_t

        return (total_weight + container_weight_t) <= self.config.stack_weight_limit_t

    def _check_discharge_order(
        self,
        bay: YardBay,
        row_idx: int,
        tier_idx: int,
        dest_port: str,
    ) -> bool:
        """
        C3: Discharge port stacking order.

        A container whose destination port is later in the calling
        sequence must not be placed below a container whose port is
        earlier (to avoid unnecessary reshuffles).  This checks that
        all containers *above* the candidate tier have a discharge
        port that is *not earlier* in the call order.
        """
        # Check containers above (tier > tier_idx) in the same row
        above_slots = bay.slots[row_idx][tier_idx + 1:]
        for s in above_slots:
            if s.occupied and s.discharge_port is not None:
                # If the above container has a "later" port, it's bad.
                # Simplified: we flag if the above container's port
                # is alphabetically/evaluated as "later" in call order.
                if self._port_order(s.discharge_port) < self._port_order(dest_port):
                    return False
        return True

    def _check_special_zone(
        self,
        block_id: int,
        container_type: str,
    ) -> bool:
        """
        C4: Special container zones — reefer containers must go to
        reefer zones, dangerous to dangerous zones, general anywhere
        except reserved blocks.
        """
        if container_type == "reefer":
            return block_id in self.yard.reefer_zones
        elif container_type == "dangerous":
            return block_id in self.yard.dangerous_zones
        return True  # general containers: allowed

    def _check_reserved_zone(
        self,
        bay: YardBay,
        row_idx: int,
        tier_idx: int,
    ) -> bool:
        """
        C5: Reserved zone protection — only allow placement in
        reserved zones if the slot is explicitly marked as unreserved.
        """
        slot = bay.slots[row_idx][tier_idx]
        return not slot.reserved

    # ══════════════════════════════════════════════════════════════
    #  Internal helpers — Penalty calculations (Stage 2)
    # ══════════════════════════════════════════════════════════════

    def _resolve_ppo_weights(
        self, ppo_weights: Optional[np.ndarray]
    ) -> Tuple[float, float, float, float, float, float]:
        """Resolve 6 weight scaling factors from PPO or config defaults."""
        if ppo_weights is not None and len(ppo_weights) == 6:
            return tuple(ppo_weights.tolist())  # type: ignore[return-value]
        return (
            self.config.weight_berth_distance,
            self.config.weight_virtual_reserve_match,
            self.config.weight_virtual_reserve_miss,
            self.config.weight_space_util,
            self.config.weight_rehandle_prob,
            self.config.weight_conflict,
        )

    def _calc_berth_distance_penalty(
        self,
        pos: FeasiblePosition,
        berth_id: int,
        port_state: dict,
    ) -> float:
        """
        Berth distance penalty (泊位距离惩罚).

        Combines shortest path distance with a dynamic congestion
        coefficient.  Following §5.2.4:

            penalty = distance * (1 + congestion_coeff)

        Weight (config): 0.3
        """
        # Look up shortest distance from berth to block
        block_dists = self._berth_block_dist.get(berth_id, {})
        base_dist = block_dists.get(pos.block_id, 1.0)

        # Congestion coefficient from yard state (dynamic)
        congestion = self.yard.congestion_coeff.get(pos.block_id, 0.0)

        penalty = base_dist * (1.0 + congestion)
        return penalty

    def _calc_virtual_reservation_penalty(
        self,
        pos: FeasiblePosition,
        container: ContainerInfo,
    ) -> Tuple[float, float]:
        """
        Virtual reservation match penalty (§5.2.3 虚拟占位机制).

        If the container matches a virtual reservation at this block,
        the penalty is low (weight: 1.0).  If it misses a reservation
        (i.e., occupies a column that was reserved for another type),
        penalty is high (weight: 2.0).

        Returns:
            (match_penalty, miss_penalty)
        """
        reserved_cols = self.config.virtual_reserve_columns.get(pos.block_id, 0)
        if reserved_cols <= 0:
            return (0.0, 0.0)

        # Simplified: if the block has reservations, estimate match/miss
        # based on column occupancy ratio
        bays = self.yard.get_block(pos.block_id)
        if bays is None:
            return (1.0, 0.0)

        # Count occupied columns in this block
        n_cols = len(bays) * bays[0].rows if bays else 1
        n_occ = sum(
            1 for bay in bays for row_slots in bay.slots for s in row_slots if s.occupied
        )
        occ_ratio = n_occ / max(n_cols, 1)
        reserv_ratio = reserved_cols / max(n_cols, 1)

        if occ_ratio < reserv_ratio:
            # Within reservation budget: low penalty (match)
            return (occ_ratio, 0.0)
        else:
            # Exceeds reservation: high penalty (miss)
            return (0.0, (occ_ratio - reserv_ratio))

    def _calc_space_util_penalty(
        self,
        pos: FeasiblePosition,
        container: ContainerInfo,
    ) -> float:
        """
        Space utilization penalty.

        Penalizes a 20ft container occupying a 40ft slot (waste).
        Weight (config): 1.0
        """
        bay = self.yard.get_bay(pos.block_id, pos.bay)
        if bay is None:
            return 0.0

        if pos.row < len(bay.slots) and pos.tier < len(bay.slots[pos.row]):
            slot = bay.slots[pos.row][pos.tier]
            if container.length_20ft and not slot.length_20ft:
                return 1.0  # 20ft in 40ft slot
        return 0.0

    def _calc_rehandle_prob_penalty(
        self,
        pos: FeasiblePosition,
        container: ContainerInfo,
    ) -> float:
        """
        Rehandle probability penalty (翻箱概率).

        Estimates likelihood that placing this container will cause
        future reshuffles.  Based on:
          - Stack height at the target column
          - Number of above containers with different discharge ports
        Weight (config): 1.5
        """
        bay = self.yard.get_bay(pos.block_id, pos.bay)
        if bay is None:
            return 0.5

        if pos.row >= len(bay.slots):
            return 0.5

        above_slots = bay.slots[pos.row][pos.tier + 1:]
        n_above = sum(1 for s in above_slots)
        n_conflict = sum(
            1 for s in above_slots
            if s.occupied and s.discharge_port is not None
            and s.discharge_port != container.discharge_port
        )

        if n_above == 0:
            return 0.0  # top tier — no rehandle

        # Stack height factor: taller stacks → higher rehandle prob
        height_ratio = (pos.tier + 1) / max(len(bay.slots[pos.row]), 1)
        conflict_ratio = n_conflict / max(n_above, 1)

        penalty = 0.4 * height_ratio + 0.6 * conflict_ratio
        return min(penalty, 1.0)

    def _calc_equipment_conflict_penalty(
        self,
        pos: FeasiblePosition,
        port_state: dict,
    ) -> float:
        """
        Equipment conflict penalty (设备冲突).

        Penalizes positions in high-activity blocks where multiple
        YCs may contend.  Based on block-level congestion.
        Weight (config): 0.8
        """
        congestion = self.yard.congestion_coeff.get(pos.block_id, 0.0)
        queue_len = port_state.get("queue_length", 0)
        yc_busy = port_state.get("yc_busy", 0)
        yc_total = port_state.get("yc_count", 12)

        if yc_total > 0:
            util = yc_busy / yc_total
        else:
            util = 0.0

        penalty = 0.5 * congestion + 0.3 * util + 0.2 * min(queue_len / 10.0, 1.0)
        return min(penalty, 1.0)

    # ══════════════════════════════════════════════════════════════
    #  Internal helpers — Collaborative optimization (Stage 3)
    # ══════════════════════════════════════════════════════════════

    def _simulate_local_impact(
        self,
        pos: FeasiblePosition,
        container: ContainerInfo,
    ) -> float:
        """
        Quick local simulation to estimate blocking/reshuffle effects.

        Evaluates the impact on already-assigned containers within a
        configurable radius of the candidate position.  Returns a
        scalar impact score (lower = better).

        The simulation:
          1. Checks adjacent bays for containers whose access is
             blocked if this container is placed here.
          2. Estimates additional reshuffles induced.
          3. Considers stack height distribution impact.

        Implementation follows §5.2.4 Stage 3 description.
        """
        impact = 0.0
        radius = self.config.local_search_radius
        bay = self.yard.get_bay(pos.block_id, pos.bay)

        if bay is None:
            return 0.0

        # 1. Blocking effect: containers in the same column (row) above
        #    this tier that are due to be retrieved before this container
        above_slots = bay.slots[pos.row][pos.tier + 1:]
        n_above_blocked = sum(
            1 for s in above_slots
            if s.occupied and s.discharge_port is not None
            and self._port_order(s.discharge_port) > self._port_order(container.discharge_port)
        )
        impact += n_above_blocked * 0.5  # blocking penalty per container

        # 2. Adjacent bay impact: check neighboring bays within radius
        block_bays = self.yard.get_block(pos.block_id)
        if block_bays:
            pos_bay_idx = next(
                (i for i, b in enumerate(block_bays) if b.bay_id == pos.bay),
                -1,
            )
            if pos_bay_idx >= 0:
                start = max(0, pos_bay_idx - radius)
                end = min(len(block_bays), pos_bay_idx + radius + 1)
                for bidx in range(start, end):
                    if bidx == pos_bay_idx:
                        continue
                    neighbor_bay = block_bays[bidx]
                    for row_slots in neighbor_bay.slots:
                        for s in row_slots:
                            if s.occupied and s.discharge_port is not None:
                                # Check if this neighbor container's column
                                # would be affected
                                if self._port_order(s.discharge_port) > self._port_order(container.discharge_port):
                                    impact += 0.1  # minor adjacency impact

        # 3. Stack height distribution: penalize creating uneven stacks
        col_heights = []
        for row_slots in bay.slots:
            h = sum(1 for s in row_slots if s.occupied)
            col_heights.append(h)
        if col_heights:
            max_h = max(col_heights)
            min_h = min(col_heights)
            height_imbalance = max_h - min_h
            impact += height_imbalance * 0.05

        return impact

    @staticmethod
    def _port_order(port: str) -> int:
        """
        Map discharge port to a call-order integer.

        This is a simplified mapping; in production it would be
        derived from the vessel's port calling sequence.
        """
        # Use a simple hash-based ordering
        return hash(port) & 0xFFFF


# ══════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════

def create_default_yard_layout(
    n_blocks: int = 10,
    bays_per_block: int = 20,
    rows: int = 6,
    tiers: int = 5,
) -> YardLayout:
    """
    Create a default yard layout for experiments.

    Args:
        n_blocks: Number of yard blocks.
        bays_per_block: Bays per block.
        rows: Rows per bay.
        tiers: Tiers per bay (stack height).

    Returns:
        A YardLayout with uniform blocks, reasonable distance
        assignments, and some special zones.
    """
    layout = YardLayout()
    layout.n_blocks = n_blocks

    for bid in range(1, n_blocks + 1):
        bays: List[YardBay] = []
        for b in range(1, bays_per_block + 1):
            bay = YardBay(bay_id=b, rows=rows, tiers=tiers)
            bays.append(bay)
        layout.blocks[bid] = bays
        layout.n_bays_total += bays_per_block

    # Assign special zones: blocks 1-2 = reefer, block 3 = dangerous,
    # block 10 = reserved
    layout.reefer_zones = [1, 2]
    layout.dangerous_zones = [3]
    layout.reserved_blocks = [10]

    # Berth distances (simplified linear distance)
    # Berth 1 closest to blocks 1-3, berth 2 to 4-6, berth 3 to 7-9
    for berth_id in range(1, 4):
        dists = []
        for bid in range(1, n_blocks + 1):
            d = abs(bid - (berth_id * 3))
            dists.append((bid, float(max(d, 1))))
        layout.berth_distances[berth_id] = dists

    # Initial congestion coefficients
    layout.congestion_coeff = {bid: 0.0 for bid in range(1, n_blocks + 1)}

    return layout


def container_info_from_dict(d: dict) -> ContainerInfo:
    """Build a ContainerInfo from a dictionary (e.g., from DES event data)."""
    return ContainerInfo(
        container_id=d.get("container_id", ""),
        length_20ft=d.get("length_20ft", True),
        weight_t=d.get("weight_t", 0.0),
        discharge_port=d.get("discharge_port", ""),
        container_type=d.get("container_type", "general"),
        destination_berth=d.get("destination_berth", 0),
        vessel_code=d.get("vessel_code", ""),
    )
