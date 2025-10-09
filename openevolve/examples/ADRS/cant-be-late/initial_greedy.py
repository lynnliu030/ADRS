# EVOLVE-BLOCK START
"""
Minimal greedy seed (non‑UP):
- Prefer SPOT when available
- Otherwise wait (NONE)
- Enforce hard safety lines with tick‑aligned checks; equality is unsafe
  • If need1d ≥ left_ticks → use ON_DEMAND (point of no return)
  • If in 2d zone and no productive SPOT → use ON_DEMAND

No comparisons to any baseline/average progress; uses only tick variables.
"""

import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class GreedySafetyStrategy(Strategy):
    NAME = "greedy_safety_seed"

    def __init__(self, args):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        gap = env.gap_seconds

        # Remaining work (conservative)
        work_left = self.task_duration - sum(self.task_done_time)
        if work_left <= 1e-9:
            return ClusterType.NONE

        # Tick‑aligned bookkeeping (equality is unsafe)
        left_ticks = max(0, math.floor((self.deadline - env.elapsed_seconds) / gap))
        need1d = math.ceil((work_left + self.restart_overhead) / gap)
        need2d = math.ceil((work_left + 2 * self.restart_overhead) / gap)

        # 1) Point of no return → must guarantee finish
        if need1d >= left_ticks:
            return ClusterType.ON_DEMAND

        # 2) In 2d zone: only ride a currently productive SPOT; else OD
        if need2d >= left_ticks:
            if env.cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # 3) Safe zone: greedy for SPOT; otherwise wait to save cost
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

# EVOLVE-BLOCK END