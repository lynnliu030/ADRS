"""
Democratic Trust Propagation (DTP) algorithm for repairing network counter values.
"""
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import copy
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.snapshot_cache import SnapshotCache
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_utils import NetworkTopology, compute_predicted_counters, extract_perturbed_snapshot, load_paths_for_timestamp
import common_utils
import pandas as pd
from tqdm import tqdm

@dataclass
class RepairConfig:
    disable_cache: bool = False
    interface_threshold: float = 0.03
    num_trials: int = 30
    similarity_threshold: float = 0.05
    seed: int = 42
    num_perturbed_nodes: int = 0 # For evaluation only, this perturbs node values. This is a bit of a kludge, but we want to be able to test what happens if our paths are buggy. This would happen for particular routers not reporting their hops correctly. This parameter corresponds to how many nodes in the topology misreport their paths.

def update_row_with_repairs(row, repair_results, method):
    """Update a row with repaired values and confidences."""
    output_row = copy.deepcopy(row)
    
    # Set repair type
    output_row['repair_type'] = method
    
    # Track confidences for averaging
    all_confidences = []
    
    # Update the row with repaired values
    for interface_name, (in_val, out_val) in repair_results['corrected_values'].items():
        if interface_name.endswith('_external'):
            # For external interfaces
            node_name = interface_name[:-9]  # Remove '_external'
            output_row[f"low_{node_name}_origination"]['corrected'] = in_val
            output_row[f"low_{node_name}_termination"]['corrected'] = out_val
        else:
            # For regular interfaces
            parts = interface_name.split('_egress_to_')
            src, dst = parts[0], parts[1]
            output_row[f"low_{interface_name}"]['corrected'] = out_val
            output_row[f"low_{src}_ingress_from_{dst}"]['corrected'] = in_val
    
    # Add confidences
    for interface_name, (in_conf, out_conf) in repair_results['confidences'].items():
        if interface_name.endswith('_external'):
            node_name = interface_name[:-9]
            output_row[f"low_{node_name}_origination"]['confidence'] = in_conf
            output_row[f"low_{node_name}_termination"]['confidence'] = out_conf
            all_confidences.extend([in_conf, out_conf])
        else:
            parts = interface_name.split('_egress_to_')
            src, dst = parts[0], parts[1]
            output_row[f"low_{interface_name}"]['confidence'] = out_conf
            output_row[f"low_{src}_ingress_from_{dst}"]['confidence'] = in_conf
            all_confidences.extend([in_conf, out_conf])
    
    # Set repair_confidence as average of all confidences
    output_row['repair_confidence'] = sum(all_confidences) / len(all_confidences) if all_confidences else None
    
    return output_row
    
class LinkValue: # Unidirectional
    def __init__(self, link_id: Tuple[int, int], in_value: float, out_value: float, demand_based_prediction: float = None, name: str = "Intf"):
        self.link_id = link_id
        self.measured_in_value = in_value
        self.measured_out_value = out_value
        self.demand_based_predicton = demand_based_prediction
        self.locked_value = None
        self.confidence = None
        self.name = name
    
    @property
    def is_locked(self):
        assert (self.confidence is None) == (self.locked_value is None), "Confidence and locked value must be set together"
        return self.confidence is not None and self.locked_value is not None
    
    @property
    def possible_values(self) -> List[float]:
        """Get possible (non-None) values for this link, link must not be locked."""
        assert self.locked_value is None, "Cannot get possible values for locked link."
        return [val for val in [self.measured_in_value, self.measured_out_value, self.demand_based_predicton] if val is not None]
    
    def pick_rand_val_or_locked(self):
        """Pick a random value from the available measured values, or the locked value if locked."""
        if self.is_locked:
            return self.locked_value
        assert len(self.possible_values) > 0, "No measured values available - need to handle re-deriving this."
        return random.choice(self.possible_values)
    
    def lock_val(self, value: float, confidence: float):
        self.locked_value = value
        self.confidence = confidence
    
    def __str__(self):
        return f"{self.name}({self.link_id})in/out/predicted: {self.measured_in_value}/{self.measured_out_value}/{self.demand_based_predicton}, locked/conf: {self.locked_value}/{self.confidence}"

class DemocraticTrustPropagationRepair:
    """
    Self-contained implementation of the Probabilistic DTP algorithm, rewritten for clarity 3/27.
    Works directly with topology JSON and CSV snapshots.
    """
    def __init__(self, topology: Dict, config: RepairConfig = RepairConfig()):
        """Initialize with network topology."""
        if config.seed is not None:
            random.seed(config.seed)
        self.config = config
        self.network_topology = NetworkTopology(topology)
        self.cache = SnapshotCache("cache/repair", disable_cache=self.config.disable_cache)

    def _compute_predicted_counters(self, snapshot: Dict, paths: Dict) -> Dict[Tuple[int, int], float]:
        """Compute predicted counter values using paths and demands.
        Args:
            snapshot: Dict containing all interface counter values
            paths: Dict containing paths between nodes
        
        Returns:
            Dictionary of predicted counter values for each link
        """
        if not paths:
            return {}
        
        return compute_predicted_counters(
            snapshot, 
            paths, 
            self.network_topology, 
            paths_perturbed_num_nodes=self.config.num_perturbed_nodes
        )

    def _get_link_value(self, snapshot: Dict, link_id: Tuple[int, int], demand_based_predictions: Dict[Tuple[int, int], float] = None) -> LinkValue:
        """Get a pair of interface counter values from snapshot, corresponding to a link (node_id -> neighbor_id).
        Args:
            snapshot: Dict containing all interface counter values
            link_id: ID of the link, (src_id, dst_id)
            demand_based_predictions: Optional dictionary of predicted counter values
        Returns:
            LinkValue object containing the interface values
            """
            
        src_id, dst_id = link_id
        
        # Get the snapshot keys and interface name from topology
        in_key, out_key = self.network_topology.get_snapshot_keys_for_link(link_id)
        interface_name = self.network_topology.get_interface_name(src_id, dst_id)
        
        # Look up values in snapshot (using None for missing keys)
        in_val = snapshot[in_key] if in_key is not None else None
        out_val = snapshot[out_key] if out_key is not None else None
        
        # Get demand-based prediction if available
        prediction = None
        if demand_based_predictions:
            prediction = demand_based_predictions.get(link_id, None)
            assert prediction is not None, f"Expected prediction for link {link_id} not found in demand-based predictions"   
        
        return LinkValue(link_id, 
                         in_val, out_val, demand_based_prediction=prediction, name=interface_name)

    def _try_router_assignments(self, node_id: int, link_states: Dict[Tuple[int, int], LinkValue]) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Try different assignments of interface values for a router.
        
        Args:
            node_id: ID of the router node
            link_states: Dictionary of link states
        Returns:
            Dictionary of router beliefs for each link
        """
        relevant_link_ids = self.network_topology.get_incident_link_ids(node_id)

        # For each link that is not locked, derive its value from sum in/out of all other values, then consider the most common value.
        # Ex: {(0,1): [256.2, 320.2]}
        derived_assignments = {link_id: [] for link_id in relevant_link_ids if not link_states[link_id].is_locked}
        
        for _ in range(self.config.num_trials):
            # For each link, randomly choose between self-measured, other-measured values, or demand-based prediction.
            assignment = {}
            for link_id in relevant_link_ids:
                link_state = link_states[link_id]
                assignment[link_id] = link_state.pick_rand_val_or_locked()
            
            total_entering = sum(val for (_, dst_id), val in assignment.items() if dst_id == node_id)
            total_leaving = sum(val for (src_id, _), val in assignment.items() if src_id == node_id)
            
            # Rederive unique value for each link that is not locked.
            for link_id in derived_assignments.keys():
                val = assignment[link_id]
                # Different logic if it's incoming or leaving traffic
                if link_id[0] == node_id: # Outgoing link
                    # Derived value is the remainder when subtracting this value from total_out
                    derived = abs(total_entering - (total_leaving - val))
                else: # Incoming link
                    derived = abs(total_leaving - (total_entering - val))
                derived_assignments[link_id].append(derived)
        
        # Compute the most common value for each derived assignment, as well as a confidence value (number of times that value appears).
        router_beliefs = {}

        for link_id, values in derived_assignments.items():
            most_frequent_val, relative_freq = common_utils.calculate_most_frequent_val_fuzzy(values, relative_tolerance=self.config.similarity_threshold)
            router_beliefs[link_id] = (most_frequent_val, relative_freq)

        # Note, this only contains router beliefs for the links that are not locked.
        return router_beliefs

    def _collect_votes(self, links: Dict) -> Dict:
        """Collect votes for all unlocked links from measurements and router beliefs."""
        votes = {}  # link_id -> [(value, weight), ...]
        
        # Add votes from measured values for unlocked links
        for link_id, link_state in links.items():
            if link_state.is_locked:
                continue
            votes[link_id] = [(v, 0.66) for v in link_state.possible_values]
        
        # Add votes from router beliefs (flow conservation)
        for node_id in self.network_topology.get_all_node_ids():
            beliefs = self._try_router_assignments(node_id, links)
            for link_id, (value, conf) in beliefs.items():
                assert not links[link_id].is_locked, "Expected to skip locked links"
                votes[link_id].append((value, conf))
        
        return votes
    
    def _select_best_candidate(self, votes: Dict) -> Optional[Tuple]:
        """Select the link with highest confidence to lock next.
        
        Returns:
            Tuple of (link_id, value, confidence) or None if no candidate found
        """
        max_conf = -1
        best_result = None
        
        # We will not lock in external interface values until all internal interfaces are locked.
        # This condition works by the pigeonhole principle: if there are more votes than 2x nodes, 
        # there must be external interfaces that we should handle last.
        should_skip_external = len(votes) > (self.network_topology.get_node_count() * 2)
        
        for link_id, vote_list in votes.items():
            # Skip external interfaces if requested
            if should_skip_external and -1 in link_id:
                continue
                
            vals, confs = zip(*vote_list)
            most_common_val, conf_of_most_common = common_utils.calculate_most_frequent_val_fuzzy(
                vals, weight_list=confs, relative_tolerance=self.config.similarity_threshold)
            
            if conf_of_most_common > max_conf:
                max_conf = conf_of_most_common
                best_result = (link_id, most_common_val, conf_of_most_common)
        
        return best_result
    
    def _run_consensus_algorithm(self, links: Dict) -> Dict:
        """Run the consensus algorithm to determine the most likely value for each link."""
        num_locked = 0
        
        while num_locked < len(links):
            # Collect all votes for this round
            votes = self._collect_votes(links)
            
            # Select the most confident candidate to lock
            best_candidate = self._select_best_candidate(votes)
            
            if best_candidate:
                link_id, value, confidence = best_candidate
                links[link_id].lock_val(value, confidence)
                num_locked += 1
            else:
                # No more candidates to lock
                break
        
        return links

    def _initialize_links(self, snapshot: Dict, paths: Dict = None) -> Dict:
        """Initialize LinkValue objects for all links in the topology."""
        links = {}
        
        # Compute demand-based predictions if paths are available
        demand_based_predictions = None
        if paths:
            demand_based_predictions = self._compute_predicted_counters(snapshot, paths)
        
        # Create LinkValue objects for all links
        for node_id in self.network_topology.get_all_node_ids():
            for link_id in self.network_topology.get_incident_link_ids(node_id):
                links[link_id] = self._get_link_value(snapshot, link_id, demand_based_predictions)
        
        return links
    
    def _build_results(self, links: Dict) -> Dict[str, Any]:
        """Build the final results dictionary from the processed links."""
        results = {
            'corrected_values': {},
            'confidences': {},
            'original_values': {}
        }
        
        for (node_id, neighbor_id), link_state in links.items():
            # Skip external interfaces as source (not reported in data)
            if node_id == -1:
                continue
            
            interface_name = self.network_topology.get_interface_name(node_id, neighbor_id)
            
            # Get link states for both directions
            out_link_state = links[(node_id, neighbor_id)]
            in_link_state = links[(neighbor_id, node_id)]
            
            # Original values
            orig_in = in_link_state.measured_out_value  # "in" for this interface is "out" for incoming link
            orig_out = out_link_state.measured_in_value  # "out" for this interface is "in" for outgoing link
            results['original_values'][interface_name] = (orig_in, orig_out)
            
            # Corrected values
            corrected_in = in_link_state.locked_value if in_link_state.is_locked else in_link_state.measured_out_value
            corrected_out = out_link_state.locked_value if out_link_state.is_locked else out_link_state.measured_in_value
            results['corrected_values'][interface_name] = (corrected_in, corrected_out)
            
            # Confidence values
            in_conf = in_link_state.confidence if in_link_state.is_locked else 0.0
            out_conf = out_link_state.confidence if out_link_state.is_locked else 0.0
            results['confidences'][interface_name] = (in_conf, out_conf)
        
        return results

    def repair_snapshot(self, snapshot: Dict, paths: Dict = None) -> Dict[str, Any]:
        """
        Process a single snapshot and return corrected values with confidences.
        
        Args:
            snapshot: Dict containing interface counter values
            paths: Optional dict containing paths between nodes for demand-based predictions
            
        Returns:
            Dict containing:
            - corrected_values: Dict mapping interface names to corrected (in, out) values
            - confidences: Dict mapping interface names to (in_conf, out_conf) values
            - original_values: Dict mapping interface names to original (in, out) values
        """
        links = self._initialize_links(snapshot, paths)
        
        # Run the consensus algorithm to determine the most likely value for each link
        repaired_links = self._run_consensus_algorithm(links)

        # Prepare results
        return self._build_results(repaired_links)
    
    def process_row(self, row: pd.Series, paths: Dict = None) -> pd.Series:
        """Process a single row for repair_df method.
        
        Args:
            row: DataFrame row containing snapshot data
            paths: Optional dict containing paths between nodes
            
        Returns:
            Repaired row with corrected values and confidences
        """
        # Extract snapshot from row
        snapshot = extract_perturbed_snapshot(row)
        
        # Repair the snapshot
        repair_results = self.repair_snapshot(snapshot, paths)
        
        # Update row with repairs
        repaired_row = update_row_with_repairs(row, repair_results, "DTP")
        
        return repaired_row
    
    def repair_df(self, df: pd.DataFrame, paths_path: str) -> pd.DataFrame:
        """Take a given DataFrame and repair it using the specified method.
        Args:
            df (pd.DataFrame): The DataFrame to repair.
            paths_path (str): Path to the paths file or directory.
        """
        
        result = self.cache.get_result(df, "repair_df", self.network_topology.get_uuid(), paths_path, self.config)
        if result is not None:
            return result
        
        # Run rows in parallel
        with ProcessPoolExecutor() as executor:
            if paths_path is not None:
                futures = []
                for i in range(len(df)):
                    row = df.iloc[i]
                    try:
                        paths = load_paths_for_timestamp(paths_path, row['timestamp'])
                        futures.append(executor.submit(self.process_row, row, paths))
                    except ValueError as e:
                        print(f"Warning: {e}, skipping row {i}")
                        futures.append(None)
            else:
                futures = [
                    executor.submit(self.process_row, df.iloc[i], None)
                    for i in range(len(df))
                ]
            
            # Collect results
            repaired_rows = []
            for i, future in enumerate(tqdm(futures, desc="Repairing rows")):
                if future is not None:
                    try:
                        result_row = future.result()
                        repaired_rows.append(result_row)
                    except Exception as e:
                        print(f"Warning: Error repairing row {i}: {e}")

        
        # Save all repaired rows
        df_repaired = pd.DataFrame(repaired_rows)
        df_repaired.attrs['Repair strategy'] = "DTP w/ paths" if paths_path is not None else "DTP w/o paths"
        df_repaired.attrs['config'] = self.config
        
        self.cache.store_result(df, df_repaired, "repair_df", self.network_topology.get_uuid(), paths_path, self.config)
        
        return df_repaired
