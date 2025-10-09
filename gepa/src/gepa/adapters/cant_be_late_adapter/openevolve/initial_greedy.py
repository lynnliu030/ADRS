"""
**Strategy Keywords:** simple-greedy
**Core Idea:** Simple greedy strategy - use SPOT when available, wait otherwise, switch to ON_DEMAND only when deadline approaches
"""

import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'
    
    def __init__(self, args):
        super().__init__(args)
    
    def reset(self, env, task):
        super().reset(env, task)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        
        # Task completion check
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        
        # Calculate remaining time until deadline
        remaining_time = self.deadline - env.elapsed_seconds
        
        # Simple deadline check: if we're running out of time, use ON_DEMAND
        # Add restart overhead to account for potential restart
        if remaining_task_time + self.restart_overhead >= remaining_time:
            # We need ON_DEMAND to guarantee completion
            return ClusterType.ON_DEMAND
        
        # Simple greedy logic: use SPOT if available, wait otherwise
        if has_spot:
            return ClusterType.SPOT
        else:
            # Just wait for SPOT to become available
            return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)