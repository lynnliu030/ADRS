import argparse
import json
import logging
import math
import random
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
    这是一个“混合实验”版本。
    它使用了脚本B的“粘滞”名称，但内部嫁接了脚本A的、基于全局RC/CR的决策逻辑。
    用于验证 _condition2() 是否是导致性能下降的根源。
    """

    NAME = "multi_region_rc_cr_threshold_optimized"

    def __init__(self, args):
        super().__init__(args)
        self.last_request_type = None

    # --- 从脚本A嫁接过来的RC/CR公式 ---
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

    # --- 核心改动：用脚本A的_step方法，替换掉脚本B原来的_step方法 ---
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        这是从脚本A完整复制过来的决策逻辑。
        """
        env = self.env
        has_preempted = not has_spot and self.last_request_type == ClusterType.SPOT
        if has_preempted:
            # random switch to a spot region
            best_spot_region = random.choice(range(self.env.get_num_regions()))
            self.env.switch_region(best_spot_region)
            logger.debug(
                f"Tick {env.tick}: Proactively switched to {best_spot_region} for spot."
            )

        remaining_time = (
            math.floor((self.deadline - env.elapsed_seconds) / env.gap_seconds)
            * env.gap_seconds
        )
        remaining_task_time = self.task_duration - sum(self.task_done_time)

        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        # 初始决策基于全局Spot
        request_type = ClusterType.SPOT if has_spot else ClusterType.NONE

        # 使用RC/CR公式进行修正
        if self._condition() < 0:
            request_type = (
                ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND
            )

        # **关键的粘滞逻辑**
        if last_cluster_type == ClusterType.ON_DEMAND:
            if not has_spot or self._condition2() >= 0:
                logger.debug(f"{env.tick}: Keep on-demand VM due to _condition2")
                request_type = ClusterType.ON_DEMAND

        # Deadline压力判断
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

            if self.restart_overhead == 0 and has_spot:
                request_type = ClusterType.SPOT


        # Final safety check: if we decide on SPOT, ensure one is actually available globally.
        # This is the key fix to prevent crashes when all regions are preempted.
        if request_type == ClusterType.SPOT and not has_spot:
            logger.warning(
                f"Tick {env.tick}: Overriding SPOT decision to NONE because no region is available."
            )
            request_type = ClusterType.NONE
        logger.debug(f"Tick {env.tick}: Final Decision: {request_type.name}")

        self.last_request_type = request_type
        return request_type
