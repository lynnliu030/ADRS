"""
**Strategy Keywords:** [uniform-progress, hysteresis, cost-aware, deadline-guaranteed]
**Core Idea:** This strategy ensures the task maintains a steady, uniform rate of progress from start to deadline. If it falls behind, it aggressively uses on-demand instances to catch up. To prevent instability, it uses a hysteresis mechanism, only switching away from a stable on-demand instance after building up a significant progress buffer.
"""
import argparse
import logging
import math

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

import typing
if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class UniformProgressStrategy(strategy.Strategy):
    """
    The Uniform Progress strategy with hysteresis.
    """
    NAME = 'uniform_progress_evolving'

    def __init__(self, args):
        if not hasattr(args, 'keep_on_demand'):
            args.keep_on_demand = False
        super().__init__(args)

    def reset(self, env: 'env.Env', task: 'task.Task'):
        """Resets the strategy's state for a new evaluation run."""
        super().reset(env, task)

    def _uniform_progress_condition(self):
        """
        Checks if the current progress is behind the ideal uniform progress.

        Returns:
            A float. If > 0, the job is behind schedule. If < 0, it's ahead.
        """
        c_0 = self.task_duration
        # c_t is the remaining task time
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        
        # Formula: expected_progress - actual_progress
        # expected_progress = t * (c_0 / r_0)
        # actual_progress = c_0 - c_t
        return t * c_0 / r_0 - (c_0 - c_t)

    def _hysteresis_condition(self):
        """
        Checks if enough buffer is built up to switch from on-demand to spot.
        
        This prevents "flapping" by ensuring we are ahead of schedule by at
        least two restart overheads' worth of progress before leaving a
        stable on-demand instance.

        Returns:
            A float. If >= 0, stay on on-demand. If < 0, it's safe to switch.
        """
        d = self.restart_overhead
        t = self.env.elapsed_seconds
        c_t = self.task_duration - sum(self.task_done_time)
        c_0 = self.task_duration
        r_0 = self.deadline

        # Formula: expected_progress_with_buffer - actual_progress
        # expected_progress_with_buffer = (t + 2 * d) * c_0 / r_0
        # actual_progress = c_0 - c_t
        return (t + 2 * d) * c_0 / r_0 - (c_0 - c_t)

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        """The main decision-making function for the strategy."""
        env = self.env
        current_cluster_type = env.cluster_type

        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            # Task is done.
            return ClusterType.NONE

        # Default decision: use spot if available, otherwise wait.
        request_type = ClusterType.SPOT if has_spot else ClusterType.NONE

        # --- Core Uniform Progress Logic ---
        # If we are behind schedule, be more aggressive.
        if self._uniform_progress_condition() > 0:
            request_type = ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        # --- Hysteresis Logic ---
        # If currently on a stable on-demand instance, be conservative about switching off.
        if current_cluster_type == ClusterType.ON_DEMAND:
            # Stay on on-demand unless a spot is available AND we have a sufficient progress buffer.
            if not has_spot or self._hysteresis_condition() >= 0:
                logger.debug(f'{env.tick}: Hysteresis engaged. Staying on on-demand.')
                request_type = ClusterType.ON_DEMAND

        # --- Final Safety Nets (Non-negotiable) ---
        remaining_time = math.floor((self.deadline - env.elapsed_seconds) /
                                    env.gap_seconds) * env.gap_seconds

        # Safety Net 1: The R(t) < C(t) + 2d condition.
        # If idle/on-demand, and we are close to the point where one more preemption
        # would cause a deadline miss, we must switch to on-demand.
        required_time_with_2d = math.ceil(
            (remaining_task_time + 2 * self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds
        
        if required_time_with_2d >= remaining_time and current_cluster_type in [
                ClusterType.NONE, ClusterType.ON_DEMAND
        ]:
            logger.debug(f'{env.tick}: Safety net (2d) triggered. Forcing on-demand.')
            return ClusterType.ON_DEMAND

        # Safety Net 2: The R(t) < C(t) + d condition.
        # This is the absolute point of no return. We must use an instance to finish.
        required_time_with_1d = math.ceil(
            (remaining_task_time + self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds

        if required_time_with_1d >= remaining_time:
            # If on a spot instance, ride it out until preemption.
            if current_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                logger.debug(f'{env.tick}: Safety net (1d) reached. Staying on spot until preemption.')
                return ClusterType.SPOT
            else:
                # Otherwise, we must switch to on-demand immediately.
                logger.debug(f'{env.tick}: Safety net (1d) triggered. Forcing on-demand.')
                # Special case: if overhead is zero, we can still use spot.
                if self.restart_overhead == 0 and has_spot:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND

        return request_type

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser') -> 'UniformProgressStrategy':
        """Creates an instance of the strategy from command-line arguments."""
        # This strategy is parameter-free.
        args, _ = parser.parse_known_args()
        return cls(args)
