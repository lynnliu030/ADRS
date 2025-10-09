# EVOLVE-BLOCK START
"""
Strategy Keywords: [dynamic-lock, bang-bang, wait-heavy, low-switch]
Core Idea: Dynamic Backload Lock (DBL)

Two behaviours:
  - Pre-lock: Prefer SPOT; NONE when safe; use ON_DEMAND only if slack is tight.
  - Post-lock: Lock to ON_DEMAND to de-risk the tail (no further switching),
    except if currently on a productive SPOT and safety nets allow riding it.

Lock condition is dynamic: when remaining wall-clock ticks are close to the
ticks required to finish on ON_DEMAND (including one restart overhead), we
flip to post-lock behaviour. This avoids late surprises without chasing
uniform progress.
"""
import argparse
import math
from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType


class DynamicBackloadLockStrategy(strategy.Strategy):
    NAME = 'dynamic_backload_lock'

    def _work_left(self) -> float:
        return self.task_duration - sum(self.task_done_time)

    def _left_ticks(self) -> int:
        gap = self.env.gap_seconds
        rem = max(0.0, self.deadline - self.env.elapsed_seconds)
        return int(math.floor(rem / gap))

    def _need_ticks(self, extra_overheads: float) -> int:
        gap = self.env.gap_seconds
        need = max(0.0, self._work_left() + extra_overheads)
        return int(math.ceil(need / gap))

    def _od_ticks(self) -> int:
        gap = self.env.gap_seconds
        return int(math.ceil(self.restart_overhead / gap))

    def _post_lock(self) -> bool:
        # Dynamic lock: if the time left is close to the OD-only finish ticks
        # (work + one restart), add a small cushion in ticks.
        left_ticks = self._left_ticks()
        need_od_ticks = self._need_ticks(self.restart_overhead)
        cushion = 2 * self._od_ticks()
        return left_ticks <= need_od_ticks + cushion

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Task complete
        if self._work_left() <= 1e-3:
            return ClusterType.NONE

        d = self.restart_overhead
        left_ticks = self._left_ticks()

        # Safety nets (tick-aligned)
        if self._need_ticks(d) >= left_ticks:
            # Point of no return
            if (
                self.env.cluster_type == ClusterType.SPOT
                and has_spot
                and self.remaining_restart_overhead < 1e-3
            ):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if self._need_ticks(2 * d) >= left_ticks:
            # Danger zone: only ride productive SPOT, else OD
            if (
                self.env.cluster_type == ClusterType.SPOT
                and has_spot
                and self.remaining_restart_overhead < 1e-3
            ):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Phase selection
        post_lock = self._post_lock()

        if not post_lock:
            # Pre-lock: prefer SPOT; NONE allowed; OD only if slack tight
            if has_spot:
                return ClusterType.SPOT
            slack_ticks = left_ticks - self._need_ticks(0.0)
            if slack_ticks <= 2 * self._od_ticks():
                return ClusterType.ON_DEMAND
            return ClusterType.NONE

        # Post-lock: lock to ON_DEMAND; only ride current productive SPOT
        if (
            self.env.cluster_type == ClusterType.SPOT
            and has_spot
            and self.remaining_restart_overhead < 1e-3
        ):
            return ClusterType.SPOT
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: 'argparse.ArgumentParser') -> 'DynamicBackloadLockStrategy':
        args, _ = parser.parse_known_args()
        return cls(args)

# EVOLVE-BLOCK END

