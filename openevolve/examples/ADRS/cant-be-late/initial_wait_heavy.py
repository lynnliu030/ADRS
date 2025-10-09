# EVOLVE-BLOCK START
"""
Strategy Keywords: [wait-heavy, slack-only, tick-aligned, low-switch]
Core Idea: Pure slack-window policy that prefers waiting (NONE) when safe.
It avoids uniform-progress pacing and does not use lag. Decisions use
tick-aligned time comparisons and only switch to ON_DEMAND when slack
falls below a chase threshold. Exits ON_DEMAND when slack is comfortably
large again, with simple hysteresis to avoid flapping.
"""
import argparse
import math
from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType


class WaitHeavySlackStrategy(strategy.Strategy):
    NAME = 'wait_heavy_slack'

    def _time_left_ticks(self):
        gap = self.env.gap_seconds
        rem = max(0.0, self.deadline - self.env.elapsed_seconds)
        return int(math.floor(rem / gap))

    def _need_ticks(self, extra_overheads: float) -> int:
        gap = self.env.gap_seconds
        work_left = self.task_duration - sum(self.task_done_time)
        need = max(0.0, work_left + extra_overheads)
        return int(math.ceil(need / gap))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Task done
        if self.task_duration - sum(self.task_done_time) <= 1e-3:
            return ClusterType.NONE

        gap = self.env.gap_seconds
        d = self.restart_overhead
        left_ticks = self._time_left_ticks()

        # Hard safety nets (tick-aligned)
        if self._need_ticks(d) >= left_ticks:
            # Point of no return
            if self.env.cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3 and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        if self._need_ticks(2 * d) >= left_ticks:
            # Danger zone: only ride productive SPOT, otherwise OD
            if self.env.cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3 and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Safe zone: prefer SPOT
        if has_spot:
            return ClusterType.SPOT

        # Slack-only policy (no lag):
        # Define tick thresholds from overhead
        od_ticks = int(math.ceil(d / gap))
        slack_ticks = left_ticks - self._need_ticks(0.0)

        # Chase threshold: if slack too low, use OD; otherwise wait NONE
        # Use distinct thresholds for ON_DEMAND exit/entry to avoid flapping
        chase_threshold = 2 * od_ticks    # below this -> use OD
        exit_threshold = 3 * od_ticks     # above this -> safe to leave OD

        if self.env.cluster_type == ClusterType.ON_DEMAND:
            # Stay on OD until slack comfortably recovers and SPOT available
            if has_spot and slack_ticks > exit_threshold:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Not on OD and no SPOT: wait when slack is healthy; chase when tight
        if slack_ticks > chase_threshold:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: 'argparse.ArgumentParser') -> 'WaitHeavySlackStrategy':
        args, _ = parser.parse_known_args()
        return cls(args)
# EVOLVE-BLOCK END

