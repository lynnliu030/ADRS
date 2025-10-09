# EVOLVE-BLOCK START
"""
Strategy Keywords: [bang-bang, phase-lock, low-switch, deadline-lock]
Core Idea: Two-phase policy. Phase 1 (pre-lock): aggressively use SPOT and
short NONE waits; avoid OD unless slack is critically low. Phase 2 (post-lock):
switch to ON_DEMAND and stop switching to de-risk the tail, unless SPOT is already
productively running and safety nets allow riding it. Lock time is set by a
simple function of deadline and restart overhead, without uniform progress.
"""
import argparse
import math
from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType


class BangBangLockStrategy(strategy.Strategy):
    NAME = 'bang_bang_lock'

    def _time_left(self) -> float:
        return max(0.0, self.deadline - self.env.elapsed_seconds)

    def _work_left(self) -> float:
        return self.task_duration - sum(self.task_done_time)

    def _ticks(self, seconds: float) -> int:
        gap = self.env.gap_seconds
        return int(math.ceil(seconds / gap))

    def _left_ticks(self) -> int:
        gap = self.env.gap_seconds
        return int(math.floor(self._time_left() / gap))

    def _need_ticks(self, extra_overheads: float) -> int:
        gap = self.env.gap_seconds
        return int(math.ceil(max(0.0, self._work_left() + extra_overheads) / gap))

    def _lock_time(self) -> float:
        # Lock with a conservative cushion: finish time + 3*d
        return max(0.0, self.deadline - 3 * self.restart_overhead)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Done
        if self._work_left() <= 1e-3:
            return ClusterType.NONE

        d = self.restart_overhead
        left_ticks = self._left_ticks()

        # Safety nets
        if self._need_ticks(d) >= left_ticks:
            if self.env.cluster_type == ClusterType.SPOT and has_spot and self.remaining_restart_overhead < 1e-3:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        if self._need_ticks(2 * d) >= left_ticks:
            if self.env.cluster_type == ClusterType.SPOT and has_spot and self.remaining_restart_overhead < 1e-3:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Determine phase
        pre_lock = self.env.elapsed_seconds < self._lock_time()

        if pre_lock:
            # Phase 1: prefer SPOT; NONE allowed; OD only if slack small
            if has_spot:
                return ClusterType.SPOT
            # Slack in ticks
            slack_ticks = left_ticks - self._need_ticks(0.0)
            od_ticks = self._ticks(d)
            if slack_ticks <= 2 * od_ticks:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        else:
            # Phase 2: lock OD to de-risk tail (avoid switches)
            if self.env.cluster_type == ClusterType.SPOT and has_spot and self.remaining_restart_overhead < 1e-3:
                # Ride productive SPOT if already on it and safety nets ok
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: 'argparse.ArgumentParser') -> 'BangBangLockStrategy':
        args, _ = parser.parse_known_args()
        return cls(args)
# EVOLVE-BLOCK END

