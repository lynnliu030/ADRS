# EVOLVE-BLOCK START
"""
Uniform Progress seed (legacy behavior, baseline initial program).
This is a simple UP variant: chase uniform progress line; use OD to catch up.
"""

import argparse
import logging
import math
from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

logger = logging.getLogger(__name__)


class UniformProgressStrategy(strategy.Strategy):
    NAME = "uniform_progress_seed"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)

    def _uniform_progress_condition(self) -> float:
        c0 = self.task_duration
        remaining = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        deadline = self.deadline
        return t * c0 / deadline - (c0 - remaining)

    def _hysteresis_condition(self) -> float:
        d = self.restart_overhead
        t = self.env.elapsed_seconds
        remaining = self.task_duration - sum(self.task_done_time)
        c0 = self.task_duration
        deadline = self.deadline
        return (t + 2 * d) * c0 / deadline - (c0 - remaining)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        remaining = self.task_duration - sum(self.task_done_time)
        if remaining <= 1e-3:
            return ClusterType.NONE

        decision = ClusterType.SPOT if has_spot else ClusterType.NONE
        if self._uniform_progress_condition() > 0:
            decision = ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        if env.cluster_type == ClusterType.ON_DEMAND:
            if not has_spot or self._hysteresis_condition() >= 0:
                decision = ClusterType.ON_DEMAND

        remaining_time = self.deadline - env.elapsed_seconds
        gap = env.gap_seconds
        need_ticks = math.ceil((remaining + self.restart_overhead) / gap)
        left_ticks = math.floor(remaining_time / gap)

        if need_ticks >= left_ticks:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        return decision

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

# EVOLVE-BLOCK END