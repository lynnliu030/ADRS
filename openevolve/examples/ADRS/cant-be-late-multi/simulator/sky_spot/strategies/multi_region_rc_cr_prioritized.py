import argparse
import json
import logging
import math
from typing import Optional, List, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

import typing
if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class MultiRegionRCCRThresholdStrategy(MultiRegionStrategy):
    """
    Smarter, cost-aware multi-region RC-CR threshold strategy.
    
    Key improvements:
    1. Switches to the cheapest available spot region at every step.
    2. Considers on-demand only when falling behind or facing deadline pressure.
    3. Simplified, more direct logic for region and cluster type selection.
    """
    NAME = 'multi_region_rc_cr_threshold'

    def __init__(self, args):
        # Pass 'args' to the parent classes to handle overheads.
        super().__init__(args)
        # Track region switch history to avoid thrashing
        self.region_switch_cooldown = 0
        self.min_cooldown_ticks = 3  # Minimum ticks between region switches
        self.last_region_costs = {}  # Track cost performance per region

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

    def _find_cheapest_spot_region(
        self,
        prices: List[Optional[float]],
        availabilities: List[bool]
    ) -> Tuple[Optional[int], Optional[float]]:
        """Finds the region with the minimum spot price among available ones."""
        cheapest_region_idx = None
        min_price = float('inf')

        for i, price in enumerate(prices):
            if availabilities[i] and price is not None and price < min_price:
                min_price = price
                cheapest_region_idx = i
        
        if cheapest_region_idx is None:
            return None, None
        return cheapest_region_idx, min_price

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Cost-aware decision-making logic.
        """
        env = self.env
        
        # Ensure env has the new methods before calling them
        if not hasattr(env, 'get_all_regions_spot_available') or not hasattr(env, 'get_all_regions_spot_prices'):
            # Fallback to old logic or handle error if env does not support multi-region price info
            return super()._step(last_cluster_type, has_spot)

        # Get current global state
        availabilities = env.get_all_regions_spot_available()
        prices = env.get_all_regions_spot_prices()
        
        cheapest_spot_idx, min_spot_price = self._find_cheapest_spot_region(prices, availabilities)
        
        current_region = env.get_current_region()
        
        remaining_time = math.floor((self.deadline - env.elapsed_seconds) /
                                    env.gap_seconds) * env.gap_seconds
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        
        # Determine if we are under pressure to use a reliable instance
        is_behind_schedule = self._condition() < 0
        is_deadline_imminent = (math.ceil(
            (remaining_task_time + self.restart_overhead) / env.gap_seconds
        ) * env.gap_seconds) >= remaining_time

        must_run = is_behind_schedule or is_deadline_imminent
        
        request_type = ClusterType.NONE

        if must_run:
            # Under pressure, default to on-demand unless a spot instance is a better choice
            request_type = ClusterType.ON_DEMAND
        
        if cheapest_spot_idx is not None:
            # If a spot instance is available, it's always preferred over NONE.
            # If we must run, we still prefer spot if available.
            request_type = ClusterType.SPOT

        # Final decision and region switching
        if request_type == ClusterType.SPOT:
            # We decided to use Spot. Make sure we are in the cheapest region.
            if current_region != cheapest_spot_idx:
                env.switch_region(cheapest_spot_idx)
                logger.info(f"Tick {env.tick}: Switching from region {current_region} to cheapest region {cheapest_spot_idx} (Price: {min_spot_price:.4f})")
        
        # Logging
        log_data = {
            'tick': self.env.tick,
            'decision': request_type.name,
            'current_region': self.env.get_current_region(),
            'cheapest_spot_region': cheapest_spot_idx,
            'min_spot_price': min_spot_price,
            'must_run': must_run,
            'is_behind': is_behind_schedule,
            'deadline_imminent': is_deadline_imminent,
            'accumulated_cost': self.env.accumulated_cost,
            'remaining_task_time': remaining_task_time,
        }
        logger.info(f"STRATEGY_DECISION: {json.dumps(log_data)}")

        return request_type

    @classmethod
    def _from_args(cls, parser: 'argparse.ArgumentParser') -> 'MultiRegionRCCRThresholdStrategy':
        group = parser.add_argument_group('MultiRegionRCCRThresholdStrategy')
        # This defines the 'migration_cost'. For now, its default is 0.
        group.add_argument('--region-switch-overhead-hours',
                           type=float,
                           default=0.0,
                           help='Overhead of region switching in hours (default: 0.0, i.e., free data migration)')
        args, _ = parser.parse_known_args()
        return cls(args)

    # Helper methods like get_global_spot_status, _condition, _condition2 remain the same
    def get_global_spot_status(self) -> tuple[bool, Optional[int]]:
        if not hasattr(self.env, 'get_num_regions') or self.env.get_num_regions() <= 1:
            has_spot = self.env.spot_available()
            return has_spot, (0 if has_spot else None)
        availabilities = self.env.get_all_regions_spot_available()
        global_has_spot = any(availabilities)
        best_region = availabilities.index(True) if global_has_spot else None
        return global_has_spot, best_region
