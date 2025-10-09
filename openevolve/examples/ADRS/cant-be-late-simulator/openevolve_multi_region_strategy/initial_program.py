#!/usr/bin/env python3
"""
Simple greedy multi-region strategy for evolution starting point.
Uses basic greedy decisions without RC-CR constraints to allow maximum exploration freedom.
"""

import argparse
import typing
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType
from sky_spot.multi_region_types import TryLaunch, Terminate, Action, LaunchResult

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


# EVOLVE-BLOCK START
class SimpleGreedyMultiRegionStrategy(MultiRegionStrategy):
    """
    Simple greedy multi-region strategy.
    Always tries SPOT first across all regions, falls back to ON_DEMAND if needed.
    No complex RC-CR conditions - maximum freedom for evolution to explore.
    """
    
    NAME = 'evolved_greedy_multi'
    
    def __init__(self, args):
        args.keep_on_demand = None
        super().__init__(args)
    
    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
    
    def _step_multi(self) -> typing.Generator[Action, typing.Optional[LaunchResult], None]:
        """Simple waiting strategy - wait for SPOT, use ON_DEMAND only when deadline critical."""
        
        # Check if task is done
        remaining_task_seconds = self.task_duration - sum(self.task_done_time)
        if remaining_task_seconds <= 1e-3:
            # Terminate all active instances
            active_instances = self.env.get_active_instances()
            for region in active_instances:
                yield Terminate(region=region)
            return
        
        # Get current state
        active_instances = self.env.get_active_instances()
        
        # Critical deadline check - only use ON_DEMAND when we absolutely must
        remaining_time = self.deadline - self.env.elapsed_seconds
        time_needed = remaining_task_seconds + self.restart_overhead
        
        if time_needed >= remaining_time:  # No margin left
            # Must use ON_DEMAND to meet deadline
            if ClusterType.ON_DEMAND not in active_instances.values():
                # Terminate any SPOT instances
                for region in list(active_instances.keys()):
                    if active_instances[region] == ClusterType.SPOT:
                        yield Terminate(region=region)
                
                # Launch ON_DEMAND
                result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                assert result is not None
                assert result.success, "ON_DEMAND should always succeed"
            return
        
        # If we already have an instance running, keep it
        if active_instances:
            return
        
        # No instance running - try SPOT in all regions
        for region in range(self.env.num_regions):
            result = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
            assert result is not None
            if result.success:
                return
        
        # No SPOT available anywhere - just wait (do nothing)
        # Don't fallback to ON_DEMAND unless deadline critical
        return
    
    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> 'SimpleGreedyMultiRegionStrategy':
        return cls(parser.parse_args())
# EVOLVE-BLOCK END