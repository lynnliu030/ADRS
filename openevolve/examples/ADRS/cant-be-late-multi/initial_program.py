# EVOLVE-BLOCK-START

import configargparse
import json
import logging
import math
import typing

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env, task

logger = logging.getLogger(__name__)

class EvolutionaryStrategy(MultiRegionStrategy):
    """
    A robust, stateful, and well-structured strategy for multi-region environments.
    This initial program serves as a strong and safe starting point for evolution.
    It correctly handles the object lifecycle, provides a basic caching mechanism,
    and implements a sound, urgency-based heuristic.
    """
    NAME = 'evolutionary_robust_starter'

    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        # --- Framework Lifecycle Note ---
        # `self.env` and `self.task` are NOT available in `__init__`.
        # They are initialized later by the framework via the `reset()` method.
        # Therefore, any attributes that depend on them must be initialized here
        # as None or empty, and populated in `reset()`.

        # --- State Variables ---
        self.initialized: bool = False
        self.region_cache: typing.Dict[int, typing.Dict[str, typing.Any]] = {}
        self.next_exploration_target_idx: int = 0

    def reset(self, env: 'env.Env', task: 'task.Task'):
        """Called by the framework to initialize environment-dependent state."""
        super().reset(env, task)
        # Initialize the cache for all known regions
        for i in range(self.env.get_num_regions()):
            self.region_cache[i] = {'has_spot': None, 'last_checked': -1}
        self.initialized = True
        logger.info(f"{self.NAME} strategy has been reset and initialized.")

    def _is_behind_schedule(self) -> bool:
        """
        Calculates if current progress is behind the required linear schedule.
        Returns True if behind, False if on-track or ahead.
        """
        if not self.initialized:
            # Should not happen in a normal run, but a safeguard.
            return False

        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline

        if r_0 <= t:
            return True
        if t == 0:
            return False

        # Required progress assuming a linear work completion schedule.
        required_progress = t * (c_0 / r_0)
        actual_progress = c_0 - c_t

        return actual_progress < required_progress

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Main decision logic for the strategy.
        """
        if not self.initialized or self.task_done:
            return ClusterType.NONE

        current_region_idx = self.env.get_current_region()
        num_regions = self.env.get_num_regions()


        # Update cache with the latest information for the current region
        self.region_cache[current_region_idx]['has_spot'] = has_spot
        self.region_cache[current_region_idx]['last_checked'] = self.env.elapsed_seconds

        # --- Decision Logic ---
        if self._is_behind_schedule():
            # URGENT: We are behind schedule. Prioritize getting work done.
            if has_spot:
                return ClusterType.SPOT
            else:
                # No spot here. We must use ON_DEMAND to avoid falling further behind.
                return ClusterType.ON_DEMAND
        else:
            # NOT URGENT: We are ahead of schedule. We can prioritize cost savings.
            if has_spot:
                # Spot is available here, use it.
                return ClusterType.SPOT
            else:
                # No spot here. Let's explore. Instead of just cycling, a smarter
                # evolution could use the cache to find a promising region.
                self.next_exploration_target_idx = (current_region_idx + 1) % num_regions
                self.env.switch_region(self.next_exploration_target_idx)
                return ClusterType.NONE

# EVOLVE-BLOCK-END
