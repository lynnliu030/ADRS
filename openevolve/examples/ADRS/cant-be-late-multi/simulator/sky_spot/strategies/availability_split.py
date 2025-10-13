import logging
import math
import typing
import json # For reading traces
import os

from sky_spot.strategies import strategy
from sky_spot.strategies import multi_strategy  # Import the MultiRegionStrategy
from sky_spot.task import Task, ChainedTask
from sky_spot.utils import ClusterType
# Need MultiTraceEnv for sub_env type hint and SubtaskMultiEnvSwitcher for main env type
from sky_spot.env import SubtaskMultiEnvSwitcher, MultiTraceEnv 

if typing.TYPE_CHECKING:
    from sky_spot import env
    import argparse

logger = logging.getLogger(__name__)

def _calculate_spot_availability(trace_file: str) -> float:
    """Reads a trace file (new format: {"data": [0,1,...]}) and calculates the percentage of time spot is available."""
    # Removed try...except block as requested

    # Ensure the path exists before trying to open
    assert os.path.exists(trace_file), f"Trace file not found: {trace_file}"

    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    
    # Assert that the loaded data is a dictionary and contains the 'data' key
    assert isinstance(trace_data, dict), f"Trace file {trace_file} does not contain a JSON dictionary. Found type: {type(trace_data)}"
    assert 'data' in trace_data, f"Trace file {trace_file} dictionary is missing the 'data' key."
    
    # Get the list of availability values
    availability_list = trace_data['data']
    assert isinstance(availability_list, list), f"Value for 'data' key in {trace_file} is not a list. Found type: {type(availability_list)}"

    total_gaps = len(availability_list)
    if total_gaps == 0:
        logger.warning(f"Availability list in trace file is empty: {trace_file}")
        # Return 0 availability for empty files, as division by zero would occur otherwise
        return 0.0 
        
    # Check the structure of the first element - Now it should be 0 or 1
    if availability_list: # Ensure list is not empty before accessing element 0
        first_element = availability_list[0]
        assert isinstance(first_element, int) and first_element in [0, 1], (
            f"First element in availability list of trace file {trace_file} is not 0 or 1. "
            f"Found type: {type(first_element)}, Value: {first_element}"
        )

    # Calculate availability based on the sum of 1s in the list
    spot_available_gaps = sum(availability_list) # Summing 0s and 1s directly
    availability = spot_available_gaps / total_gaps
    logger.debug(f"Calculated availability for {trace_file}: {availability:.2%}")
    return availability

class AvailabilitySplitStrategy(multi_strategy.MultiRegionStrategy):
    """
    A strategy for ChainedTask with SubtaskMultiEnvSwitcher.

    It splits the total deadline proportionally based on sub-task durations,
    calculates average spot availability from traces for each sub-task's env,
    and uses a simple fallback-to-on-demand logic based on the calculated 
    intermediate deadlines for each sub-task phase. It prioritizes finishing
    the sub-task associated with the less available resource within its 
    proportional deadline slice.
    """
    NAME = 'availability_split'

    def __init__(self, args):
        super().__init__(args)
        # These will be populated in reset()
        self.sub_deadlines_seconds: list[float] = []
        self.cumulative_sub_deadlines_seconds: list[float] = []
        self.current_subtask_idx: int = 0
        self.less_available_idx: int = -1 # Index of the sub-task with less availability

    def reset(self, env: 'env.Env', task: Task):
        super().reset(env, task) # Sets self.env, self.task, self.deadline, self.restart_overheads_seconds etc.

        self.restart_overheads_seconds = [o * 3600.0 for o in self.restart_overheads]

        # --- Assertions ---
        if not isinstance(task, ChainedTask):
            raise TypeError(f"{self.NAME} requires a ChainedTask.")
        # Type hint for clarity, self.task is already set by super().reset
        self.task = task  # We'll rely on super's typing here

        if not isinstance(env, SubtaskMultiEnvSwitcher):
             raise TypeError(f"{self.NAME} requires a SubtaskMultiEnvSwitcher environment.")
        # Avoid overriding with more specific type hint, the parent class already handles this
        
        num_sub_tasks = len(self.task._sub_tasks)
        if num_sub_tasks != len(env.sub_environments):
             raise ValueError("Number of sub-tasks and sub-environments must match.")
        if num_sub_tasks == 0:
             raise ValueError("ChainedTask cannot have zero sub-tasks.")
        
        # --- Calculate Availability for each Sub-environment ---
        sub_env_availabilities = []
        for i, sub_env in enumerate(env.sub_environments):
            # Check if the sub-environment is a MultiTraceEnv to access traces
            if not isinstance(sub_env, MultiTraceEnv):
                 logger.warning(f"Sub-environment {i} is type {type(sub_env)}, not MultiTraceEnv. Cannot calculate availability from traces. Assigning default 100%.")
                 sub_env_availabilities.append(1.0) 
                 continue

            region_availabilities = []
            # Access trace files directly from the sub-environment's config
            trace_files = sub_env.config.get('trace_files', []) 
            if not trace_files:
                 logger.warning(f"Sub-environment {i} (MultiTraceEnv) has no 'trace_files' in its config. Assuming default 100% availability.")
                 region_availabilities.append(1.0) # Assign default if no traces found
            else:
                for trace_file in trace_files:
                    # Resolve relative paths if necessary (assuming paths are relative to workspace root)
                    # For now, assume paths are correct as provided.
                    availability = _calculate_spot_availability(trace_file)
                    region_availabilities.append(availability)
            
            # Calculate average availability across regions for this sub-task's env
            if not region_availabilities: # Avoid division by zero if loop didn't run
                 avg_availability = 1.0 # Default if no valid availabilities calculated
                 logger.warning(f"Could not calculate any region availability for sub-env {i}. Defaulting to 100%.")
            else:
                 avg_availability = sum(region_availabilities) / len(region_availabilities)
            
            sub_env_availabilities.append(avg_availability)
            logger.info(f"Sub-task {i} environment average spot availability: {avg_availability:.2%}")

        # --- Determine Less Available Task Index ---
        if len(sub_env_availabilities) > 0:
            # Find the minimum availability value
            min_availability = min(sub_env_availabilities)
            # Find the first index matching this minimum value
            self.less_available_idx = sub_env_availabilities.index(min_availability)
            logger.info(f"Sub-task {self.less_available_idx} identified as having lower availability ({min_availability:.2%})")
        else:
            # This case should ideally not be reached due to earlier checks
            self.less_available_idx = 0 
            logger.warning("Could not determine availabilities, defaulting less available task to index 0.")

        # --- Split Deadline Proportionally ---
        total_task_duration_seconds = self.task.get_total_duration_seconds()
        total_deadline_seconds = self.deadline # self.deadline is already in seconds

        if total_task_duration_seconds <= 0:
             logger.warning("Total task duration is zero or negative. Cannot calculate proportional deadlines.")
             # Assign equal deadlines or handle error? Let's assign zero.
             self.sub_deadlines_seconds = [0.0] * num_sub_tasks
        else:
            self.sub_deadlines_seconds = []
            for i, sub_task in enumerate(self.task._sub_tasks):
                sub_duration_seconds = sub_task.get_total_duration_seconds()
                # Calculate the duration slice for this sub-task based on the total deadline
                proportional_deadline_duration = (sub_duration_seconds / total_task_duration_seconds) * total_deadline_seconds
                self.sub_deadlines_seconds.append(proportional_deadline_duration)
                logger.info(f"Sub-task {i} proportional deadline duration slice: {proportional_deadline_duration / 3600:.2f} hours")

        # --- Calculate Cumulative Deadlines (absolute time by which each task *must* be finished) ---
        self.cumulative_sub_deadlines_seconds = []
        cumulative_time = 0
        for duration_slice in self.sub_deadlines_seconds:
             cumulative_time += duration_slice
             self.cumulative_sub_deadlines_seconds.append(cumulative_time)
             
        # Adjust the very last cumulative deadline to exactly match the total deadline 
        # to account for potential floating-point inaccuracies.
        if self.cumulative_sub_deadlines_seconds:
             self.cumulative_sub_deadlines_seconds[-1] = total_deadline_seconds 
        
        logger.info(f"Calculated cumulative absolute deadlines (seconds): {self.cumulative_sub_deadlines_seconds}")

        # --- Initialize State ---
        self.current_subtask_idx = 0 # Start with the first sub-task

    def _get_current_restart_overhead(self) -> float:
        """Gets the restart overhead for the current sub-task index."""
        # Assumes self.restart_overheads_seconds is populated by MultiRegionStrategy.reset()
        current_idx = self.current_subtask_idx

        overheads = self.restart_overheads_seconds
        num_overheads = len(overheads)
        
        if num_overheads == 1:
             # If only one overhead is given, use it for all sub-tasks
             return overheads[0]
        elif current_idx < num_overheads:
             # If enough overheads are provided, use the one matching the index
             return overheads[current_idx]
        else:
             # Fallback if not enough overheads were provided (use the last one)
             logger.warning(f"Subtask index {current_idx} >= number of restart overheads ({num_overheads}). Using the last available overhead.")
             return overheads[-1]

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Using correct type hints to avoid issues
        env = self.env  # Type should be handled by parent class now
        task = self.task # Type should be handled by parent class now
        
        assert isinstance(task, ChainedTask), "Task must be a ChainedTask"
        assert isinstance(env, SubtaskMultiEnvSwitcher), "Environment must be a SubtaskMultiEnvSwitcher"

        # --- Check Overall Task Completion First ---
        # Use the task's is_done property which checks total progress vs total duration
        if task.is_done:
             logger.debug("Overall ChainedTask is done.")
             return ClusterType.NONE
             
        # --- Determine Current Sub-Task and its Deadline ---
        idx = self.current_subtask_idx
        # Ensure index is within bounds of calculated deadlines
        if idx >= len(self.cumulative_sub_deadlines_seconds):
             logger.error(f"Current subtask index {idx} is out of bounds for cumulative deadlines list (len={len(self.cumulative_sub_deadlines_seconds)}). Task status: {task.get_info()}")
             # This indicates a potential logic error or unexpected state. Fallback?
             return ClusterType.NONE # Stop processing if state is inconsistent

        current_absolute_deadline_seconds = self.cumulative_sub_deadlines_seconds[idx]
        remaining_phase_deadline_seconds = max(0.0, current_absolute_deadline_seconds - env.elapsed_seconds)

        # --- Calculate Remaining Duration for Current Sub-Task ---
        # Use the task's get_info() which derives progress from the shared task_done_time list
        task_info = task.get_info()
        
        # Verify our internal index matches the task's derived index
        task_derived_idx = task_info.get('current_sub_task_index', -1)
        if idx != task_derived_idx:
             # This can happen if the task progresses enough to switch index *between*
             # the start of this _step call and when get_info() was calculated internally.
             # It's usually safer to trust the task's calculation based on actual progress.
             logger.debug(f"Strategy index ({idx}) differs from task's derived index ({task_derived_idx}). Updating strategy index.")
             # Update our internal index to match the task's view
             self.current_subtask_idx = task_derived_idx
             idx = task_derived_idx # Use the updated index for the rest of the logic
             # Re-fetch deadlines based on the potentially updated index
             if idx >= len(self.cumulative_sub_deadlines_seconds):
                 logger.error(f"Updated subtask index {idx} is now out of bounds for cumulative deadlines.")
                 return ClusterType.NONE
             current_absolute_deadline_seconds = self.cumulative_sub_deadlines_seconds[idx]
             remaining_phase_deadline_seconds = max(0.0, current_absolute_deadline_seconds - env.elapsed_seconds)

        # Calculate remaining duration based on task's info dict
        current_subtask_target = task_info.get('current_sub_task_Target(seconds)', 0)
        current_subtask_done = task_info.get('current_sub_task_Done(seconds)', 0)
        remaining_current_subtask_duration_seconds = max(0.0, current_subtask_target - current_subtask_done)
        
        # Check if the current sub-task is effectively done based on remaining duration
        current_sub_task_is_done = remaining_current_subtask_duration_seconds <= 1e-8 

        # --- Check if Current Sub-Task is Done and Advance Index (if not last task) ---
        if current_sub_task_is_done and idx < len(task._sub_tasks) - 1:
             logger.info(f"Sub-task {idx} considered done at {env.elapsed_seconds / 3600:.2f}h (Remaining: {remaining_current_subtask_duration_seconds:.2f}s). Moving to sub-task {idx + 1}.")
             # Advance the internal index for the *next* step
             self.current_subtask_idx = idx + 1
             # For *this* step, since the task is done, request NONE (or potentially start next immediately?)
             # Let's request NONE for this step, the next step will handle the new sub-task.
             # This avoids complex logic of recalculating everything for the next task within this step.
             return ClusterType.NONE 
        elif current_sub_task_is_done and idx == len(task._sub_tasks) - 1:
             # Last sub-task is done, the overall task should be done too.
             logger.debug(f"Last sub-task {idx} considered done.")
             # The initial check task.is_done should catch this, but double-check.
             return ClusterType.NONE

        # --- Simple Fallback Logic based on Sub-Deadline ---
        request_type = ClusterType.NONE # Default
        current_restart_overhead = self._get_current_restart_overhead()

        # Time needed to finish the *current* sub-task if we switch to OD now
        time_needed_1d = math.ceil(
             (remaining_current_subtask_duration_seconds + current_restart_overhead) / env.gap_seconds
        ) * env.gap_seconds

        # Floor remaining deadline time for *this phase* to nearest gap
        remaining_phase_deadline_gapped = math.floor(
             remaining_phase_deadline_seconds / env.gap_seconds) * env.gap_seconds

        # --- Decision Logic ---
        if time_needed_1d >= remaining_phase_deadline_gapped:
             # Not enough time left in this phase's deadline to finish with spot (or current state) + one overhead.
             # Must switch to/stay on ON_DEMAND unless already on spot with no pending restart.
             if last_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3 and has_spot:
                 # Currently on Spot, spot is available, and no restart pending -> keep Spot
                 logger.debug(f'{env.elapsed_seconds // env.gap_seconds}: Sub-deadline ({current_absolute_deadline_seconds/3600:.2f}h) near for task {idx}. Keeping Spot (Need {time_needed_1d/3600:.2f}h, Have {remaining_phase_deadline_gapped/3600:.2f}h)')
                 request_type = ClusterType.SPOT
             else:
                 # Not on spot, or restarting, or spot not available -> must go On-Demand
                 logger.debug(f'{env.elapsed_seconds // env.gap_seconds}: Sub-deadline ({current_absolute_deadline_seconds/3600:.2f}h) reached for task {idx}. Switching/staying On-demand (Need {time_needed_1d/3600:.2f}h, Have {remaining_phase_deadline_gapped/3600:.2f}h)')
                 request_type = ClusterType.ON_DEMAND
        else:
             # Enough time remaining in this phase's deadline. Prefer spot if available.
             request_type = ClusterType.SPOT if has_spot else ClusterType.NONE
             logger.debug(f'{env.elapsed_seconds // env.gap_seconds}: Sub-deadline ({current_absolute_deadline_seconds/3600:.2f}h) OK for task {idx}. Requesting {request_type} (Need {time_needed_1d/3600:.2f}h, Have {remaining_phase_deadline_gapped/3600:.2f}h)')

        # --- Multi-Region Recovery Attempt (within current sub-task's env) ---
        # If we decided NONE primarily because spot is not available (has_spot=False),
        # try switching regions within the active sub-environment.
        if request_type == ClusterType.NONE and last_cluster_type == ClusterType.SPOT and not has_spot:
             logger.debug(f"{env.elapsed_seconds // env.gap_seconds}: Preempted in region {env.get_current_region()}, trying to find spot elsewhere within sub-task {idx}'s allowed regions.")
             
             # Get the *currently active* sub-environment from the switcher
             # Use hasattr to check if the attribute exists first
             if hasattr(env, 'current_sub_env_index'):
                 current_sub_env_index = env.current_sub_env_index # Get index switcher is using
                 if current_sub_env_index != idx:
                      # This might indicate the switcher hasn't advanced yet, even if our logic has.
                      # Use the index the *environment* believes it's on for region switching.
                      logger.warning(f"Strategy index ({idx}) differs from Env's active sub-env index ({current_sub_env_index}). Using Env's index for region switching.")
                      active_idx_for_env = current_sub_env_index
                 else:
                      active_idx_for_env = idx
             else:
                 logger.warning("Environment does not have 'current_sub_env_index' attribute. Using strategy's current_subtask_idx.")
                 active_idx_for_env = idx

             # Check if the active sub-environment supports multi-region operations
             current_sub_env = env.sub_environments[active_idx_for_env] 
             if isinstance(current_sub_env, MultiTraceEnv):
                 num_sub_env_regions = current_sub_env.get_num_regions()
                 # We need the region index *within the sub-environment*
                 current_sub_env_region_idx = current_sub_env.current_region 
                 
                 switched = False
                 # Iterate through regions *of the sub-environment*
                 for i in range(num_sub_env_regions):
                     # Don't try to switch to the same region
                     if i == current_sub_env_region_idx:
                          continue
                          
                     # Check if spot is available in the *target region* of the *active sub-env*
                     if current_sub_env.spot_available_in_region(i):
                         logger.info(f"Found available spot in region {i} for sub-task {active_idx_for_env}. Attempting switch.")
                         try:
                             # Check if the switcher environment has the switching method
                             if hasattr(env, 'switch_active_sub_env_region'):
                                 # Tell the *switcher* environment to switch the region of its *active* sub-env
                                 env.switch_active_sub_env_region(i) 
                                 # Add the overhead for the switch. The main strategy step() handles decrementing this.
                                 # Use the overhead specific to the task we are *currently* working on (idx)
                                 self.remaining_restart_overhead += self._get_current_restart_overhead() 
                                 request_type = ClusterType.SPOT # Update request type
                                 switched = True
                                 logger.info(f"Successfully switched active sub-env {active_idx_for_env} to region {i}.")
                                 break # Stop searching once a switch is made
                             else:
                                 # Fallback if the method doesn't exist - try another approach
                                 logger.warning("Environment does not have 'switch_active_sub_env_region' method. Trying alternative approach.")
                                 # Just use the sub-environment's switch_region directly
                                 current_sub_env.switch_region(i)
                                 self.remaining_restart_overhead += self._get_current_restart_overhead()
                                 request_type = ClusterType.SPOT # Update request type
                                 switched = True
                                 logger.info(f"Used direct switch_region on sub-env {active_idx_for_env} to region {i}.")
                                 break
                         except Exception as e:
                             logger.error(f"Error occurred during region switch attempt to region {i} for sub-env {active_idx_for_env}: {e}")
                             # Keep request_type as NONE if switch fails
                 
                 if not switched:
                     logger.debug(f"No other region found with spot for sub-task {active_idx_for_env} environment.")
             else:
                 logger.debug(f"Current active sub-environment ({active_idx_for_env}) is not a MultiTraceEnv, cannot switch regions.")


        return request_type

    # --- Boilerplate for Registration and Config ---
    @classmethod
    def add_parser_args(cls, parser: 'argparse.ArgumentParser'):
        """Add arguments specific to this strategy (if any)."""
        # Currently, no specific arguments needed for this strategy itself.
        pass

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser') -> 'AvailabilitySplitStrategy':
        """Create an instance from parsed arguments."""
        args, _ = parser.parse_known_args()
        # Pass the args namespace to the constructor
        return cls(args)

    @property
    def name(self) -> str:
        """Return the name of the strategy."""
        return self.NAME

    @property
    def config(self) -> dict:
        """Return the configuration dictionary for logging."""
        # Get the base config from the parent class
        cfg = super().config 
        # Add strategy-specific parameters calculated during reset
        cfg.update({
            'less_available_idx': self.less_available_idx,
            'sub_deadlines_seconds': self.sub_deadlines_seconds,
            'cumulative_sub_deadlines_seconds': self.cumulative_sub_deadlines_seconds,
            # Optionally include calculated availabilities if useful for logging
            # 'calculated_availabilities': self._calculated_availabilities # Would need to store this in reset
        })
        return cfg 