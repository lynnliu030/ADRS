import argparse
import json
import logging
import math
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

import typing

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class MultiRegionRCCRThresholdStrategy(MultiRegionStrategy):
    """
    Hybrid experimental version.
    Keeps the sticky naming from script B but uses script A's global RC/CR decision logic.
    Used to verify whether _condition2() causes the performance drop.
    """

    NAME = "multi_region_rc_cr_threshold"

    def __init__(self, args):
        super().__init__(args)

    # --- RC/CR formula copied from script A ---
    def _condition(self):
        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        return c_0 - c_t - t * c_0 / r_0

    def _condition2(self):
        d = self.restart_overhead
        t = self.env.elapsed_seconds
        c_t = self.task_duration - sum(self.task_done_time)
        c_0 = self.task_duration
        r_0 = self.deadline
        return (t + 2 * d) * c_0 / r_0 - (c_0 - c_t)

    def get_global_spot_status(self) -> tuple[bool, Optional[int]]:
        if not hasattr(self.env, "get_num_regions") or self.env.get_num_regions() <= 1:
            has_spot = self.env.spot_available()
            return has_spot, (0 if has_spot else None)
        availabilities = self.env.get_all_regions_spot_available()
        global_has_spot = any(availabilities)
        best_region = availabilities.index(True) if global_has_spot else None
        return global_has_spot, best_region

    # --- Core change: replace script B's _step with script A's version ---
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decision logic copied verbatim from script A.
        """
        env = self.env
        global_has_spot, best_spot_region = self.get_global_spot_status()

        remaining_time = (
            math.floor((self.deadline - env.elapsed_seconds) / env.gap_seconds)
            * env.gap_seconds
        )
        remaining_task_time = self.task_duration - sum(self.task_done_time)

        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        # Initial decision based on global spot availability
        request_type = ClusterType.SPOT if global_has_spot else ClusterType.NONE

        # Adjust decision using the RC/CR formula
        if self._condition() < 0:
            request_type = (
                ClusterType.SPOT if global_has_spot else ClusterType.ON_DEMAND
            )

        # **Sticky logic**
        if last_cluster_type == ClusterType.ON_DEMAND:
            if not global_has_spot or self._condition2() >= 0:
                logger.debug(f"{env.tick}: Keep on-demand VM due to _condition2")
                request_type = ClusterType.ON_DEMAND

        # Deadline pressure check
        total_task_remaining = (
            math.ceil((remaining_task_time + self.restart_overhead) / env.gap_seconds)
            * env.gap_seconds
        )

        if total_task_remaining >= remaining_time:
            if (
                last_cluster_type == ClusterType.SPOT
                and self.remaining_restart_overhead < 1e-3
            ):
                request_type = ClusterType.SPOT
            else:
                request_type = ClusterType.ON_DEMAND

            if self.restart_overhead == 0 and global_has_spot:
                request_type = ClusterType.SPOT

        # Switching logic
        if request_type == ClusterType.SPOT and best_spot_region is not None:
            if not has_spot:  # has_spot reflects local spot availability
                if env.get_current_region() != best_spot_region:
                    env.switch_region(best_spot_region)
                    logger.debug(
                        f"Tick {env.tick}: Proactively switched to {best_spot_region} for spot."
                    )

        logger.debug(f"Tick {env.tick}: Final Decision: {request_type.name}")
        return request_type
