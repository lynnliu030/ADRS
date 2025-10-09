import hashlib
import json
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import random
import os
from datetime import datetime

class NetworkTopology:
    def __init__(self, topology_dict: Dict):
        """Initialize with network topology JSON structure."""
        self.topology = topology_dict
        self.nodes = {node['id']: node for node in topology_dict['nodes']}
        self.node_name_to_id = {node['name']: node['id'] for node in topology_dict['nodes']}
        
        # Build adjacency lists and interface mappings
        self.adjacency = {node_id: [] for node_id in self.nodes}
        self.interfaces = {}  # (node_id, neighbor_id) -> interface_name
        
        # Add regular interfaces from topology links
        for link in topology_dict['links']:
            src, dst = link['source'], link['target']
            self.adjacency[src].append(dst)
            self.adjacency[dst].append(src)
            
            # Create interface names based on node names
            src_name = self.nodes[src]['name']
            dst_name = self.nodes[dst]['name']
            
            # Interface from src to dst 
            self.interfaces[(src, dst)] = f"{src_name}_egress_to_{dst_name}"
            # Interface from dst to src
            self.interfaces[(dst, src)] = f"{dst_name}_egress_to_{src_name}"
            
        # Add external interface for each node
        # This represents both origination (in) and termination (out) traffic
        for node_id, node in self.nodes.items():
            # Add to adjacency list with special sentinel value -1 to indicate external interface
            self.adjacency[node_id].append(-1)
            
            # Add interface name
            self.interfaces[(node_id, -1)] = f"{node['name']}_external"
        
        # Get uuid from dumping to json and hashing
        json_str = json.dumps(self.topology)
        hash_str = hashlib.sha256(json_str.encode()).hexdigest()[:8]
        self.uuid = f'n={len(self.nodes)}/l={len(self.interfaces)}/h={hash_str}'
        
    def get_uuid(self) -> str:
        """Get a unique identifier for the topology that is stable across runs."""
        return self.uuid
            
    def get_incident_link_ids(self, node_id: int) -> List[Tuple[int, int]]:
        """Get list of IDs for all links incident to a node."""
        return [(node_id, neighbor) for neighbor in self.adjacency[node_id]] + [(neighbor, node_id) for neighbor in self.adjacency[node_id]]
    
    def get_egress_intf_id(self, node_id: int) -> Tuple[int, int]:
        """Get the egress interface ID for a node."""
        return (node_id, -1)
    
    def get_ingress_intf_id(self, node_id: int) -> Tuple[int, int]:
        """Get the ingress interface ID for a node."""
        return (-1, node_id)
    
    def get_snapshot_keys_for_link(self, link_id: Tuple[int, int]) -> Tuple[Optional[str], Optional[str]]:
        """Get the snapshot keys for looking up (in_value, out_value) for a link.
        
        Returns:
            Tuple of (in_key, out_key) where either might be None for external interfaces
        """
        src_id, dst_id = link_id
        
        # TODO(akrentsel): clean up this logic, can be shorter.
        
        # Handle special case for external links
        if dst_id == -1:  # External link (internal -> external)
            node_name = self.get_node_name(src_id)
            in_key = f"low_{node_name}_termination"  # Traffic "termination" at this node
            out_key = None
            return (in_key, out_key)
            
        elif src_id == -1:  # External link (external -> internal)
            node_name = self.get_node_name(dst_id)
            in_key = None
            out_key = f"low_{node_name}_origination"  # Traffic "originating" at this node
            return (in_key, out_key)
            
        else:  # Regular internal link
            src_name = self.get_node_name(src_id)
            dst_name = self.get_node_name(dst_id)
            in_key = f"low_{src_name}_egress_to_{dst_name}"
            out_key = f"low_{dst_name}_ingress_from_{src_name}"
            return (in_key, out_key)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get all neighbor IDs for a node (including external interface as -1)."""
        return self.adjacency[node_id]
    
    def get_interface_name(self, node_id: int, neighbor_id: int) -> str:
        """Get the interface name for a link between two nodes."""
        # Handle special case for external -> internal links 
        if node_id == -1:
            return f"non-existant_external_to_{neighbor_id}"
        return self.interfaces[(node_id, neighbor_id)]
    
    def get_all_node_ids(self) -> List[int]:
        """Get all node IDs in the topology."""
        return list(self.nodes.keys())
    
    def get_node_name(self, node_id: int) -> str:
        """Get node name by ID."""
        return self.nodes[node_id]['name']
    
    def get_node_count(self) -> int:
        """Get the total number of nodes in the topology."""
        return len(self.nodes)

def are_similar(alloc1, alloc2, threshold=0.05, zero_tolerance=1e-9):
    """
    Checks if two allocations (lists of numbers) are similar.

    Two allocations are similar if, for every corresponding element pair (a, b),
    the absolute difference abs(a - b) is within a given percentage threshold
    of 'a'. Special handling is included for zero values.

    Args:
        alloc1 (list): The first allocation list.
        alloc2 (list): The second allocation list.
        threshold (float): The maximum allowed relative difference (e.g., 0.05 for 5%).
        zero_tolerance (float): Values smaller than this are treated as zero for comparison.

    Returns:
        bool: True if allocations are similar, False otherwise.
    """
    assert len(alloc1) == len(alloc2), "Both allocations must have the same length."

    for i in range(len(alloc1)):
        val1 = alloc1[i]
        val2 = alloc2[i]
        diff = abs(val1 - val2)

        # Handle cases where val1 is zero or very close to it
        if abs(val1) < zero_tolerance:
            # If val1 is near zero, val2 must also be near zero to be similar
            if abs(val2) >= zero_tolerance:
                return False
        # Handle non-zero val1
        elif diff > threshold * abs(val1):
             # If the difference exceeds the threshold relative to val1, they are not similar
            return False
            
    # If all elements passed the check, the allocations are similar
    return True

def calculate_most_frequent_list_fuzzy(allocations_list, similarity_threshold=0.05):
    """
    Finds the most common "clump" of similar resource allocations.

    Args:
        allocations_list (list[list[float]]): A list of resource allocations,
            where each allocation is a list of numbers. All sublists must
            have the same length.
        similarity_threshold (float): The threshold used by are_similar()
            to determine if two allocations belong to the same clump (e.g., 0.05 for 5%).

    Returns:
        tuple: A tuple containing:
            - list[float] or None: A representative allocation from the largest clump.
                                   None if the input list is empty.
            - float: The frequency of the largest clump (proportion of total allocations),
                     between 0.0 and 1.0.
    """
    if not allocations_list:
        return None, 0.0

    total_allocations = len(allocations_list)

    # Optional: Validate that all sublists have the same length
    if total_allocations > 0:
        n = len(allocations_list[0])
        for alloc in allocations_list[1:]:
            if len(alloc) != n:
                raise ValueError("All sublists (allocations) must have the same length.")

    # Stores tuples: (representative_allocation, list_of_member_indices_in_clump)
    # Using indices can be slightly more memory efficient if allocations are large
    clumps = []
    # Keep track of which allocations have been assigned to a clump
    assigned = [False] * total_allocations

    for i in range(total_allocations):
        if assigned[i]:
            continue # Skip if already part of a clump

        # Start a new clump with allocation 'i' as the representative
        current_allocation = allocations_list[i]
        new_clump_members = [i] # Store index
        assigned[i] = True

        # Check subsequent allocations to see if they belong to this new clump
        for j in range(i + 1, total_allocations):
            if not assigned[j]: # Only check unassigned allocations
                other_allocation = allocations_list[j]
                if are_similar(current_allocation, other_allocation, similarity_threshold):
                    new_clump_members.append(j) # Add index to current clump
                    assigned[j] = True # Mark as assigned

        # Add the newly formed clump (representative, member indices)
        clumps.append((current_allocation, new_clump_members))

    # Find the largest clump
    if not clumps:
         # This case should technically not be reachable if allocations_list is not empty
         return None, 0.0

    largest_clump_rep = None
    max_clump_size = -1

    for representative, members in clumps:
        if len(members) > max_clump_size:
            max_clump_size = len(members)
            # Use the first item added (the representative) as the example to return
            largest_clump_rep = representative # Representative stored in the tuple

    # Calculate frequency
    frequency = max_clump_size / total_allocations if total_allocations > 0 else 0.0

    return largest_clump_rep, frequency

def calculate_most_frequent_val_fuzzy(data, weight_list=None, relative_tolerance=0.02):
    """
    Calculates the most frequent "fuzzy" value in a list of numbers.

    Numbers are grouped into "clumps" if they are within the specified
    relative tolerance of the clump's current average.

    Args:
        data (list): A list of numbers (integers or floats).
        weight_list (list): A list of weights for each number in data.
                            Defaults to None, which gives equal weight to all numbers.
                            Note the relative_frequency still has the number of elements in the clump as the denominator, not the sum of weights.
        relative_tolerance (float): The tolerance for grouping numbers.
                                     Numbers x and y are considered the same
                                     if abs(x - y) <= relative_tolerance * y_avg,
                                     where y_avg is the current average of a clump.
                                     Defaults to 0.02 (2%).

    Returns:
        tuple: (most_frequent_average, relative_frequency)
               Returns (None, 0) if the input list is empty.
        Raises:
            ValueError: If relative_tolerance is not between 0 and 1.
    """
    assert data, "Input data list must not be empty"
    assert 0 <= relative_tolerance < 1, "relative_tolerance must be between 0 (inclusive) and 1 (exclusive)"
    if weight_list is not None:
        assert len(data) == len(weight_list), "Data and weight list must have the same length"

    # List to store clump information: {'sum': float, 'count': int, 'average': float}
    clumps = []
    total_count = len(data)

    def _is_close(value, clump_avg, tolerance):
        if clump_avg <= 0.001: # Avoid division by zero
             return value == 0
        # Check if the value is within the relative tolerance of the clump average
        return abs(value - clump_avg) <= tolerance * abs(clump_avg)

    for index in range(len(data)):
        value = float(data[index])
        weight = 1.0 if weight_list is None else weight_list[index]
        found_clump = False
        # Try to add to an existing clump
        for clump in clumps:
            if _is_close(value, clump['average'], relative_tolerance):
                clump['sum'] += value
                clump['weight'] += weight
                clump['count'] += 1
                clump['average'] = clump['sum'] / clump['count'] # Update average
                found_clump = True
                break # Added to this clump, move to next value in data
        # If no suitable clump was found, create a new one
        if not found_clump:
            clumps.append({
                'sum': value,
                'weight': weight,
                'count': 1,
                'average': value
            })
    assert clumps, "No clumps found, this should not happen"

    # Find the clump dictionary with the maximum 'count' value
    most_frequent_clump = max(clumps, key=lambda clump: clump['count'])
    representative_value = most_frequent_clump['average']
    relative_frequency = most_frequent_clump['weight'] / total_count
    return representative_value, relative_frequency

def perc_equal(a: float, b: float, threshold: float) -> bool:
    """Check if two values are equal within threshold percentage."""
    if a == b:
        return True
    # Calculate percentage difference relative to larger value
    larger = max(abs(a), abs(b))
    if larger == 0:
        return True
    diff = abs(a - b) / larger
    return diff <= threshold


def main():
    # --- Example Usage ---
    allocations = [
        [1.05, 7.8, 3.0],
        [1.0, 7.7, 3.05],
        [5.4, 12.0, 1.0],
        [1.02, 7.9, 3.1] # Another one similar to the first clump
    ]

    most_common, freq = calculate_most_frequent_list_fuzzy(allocations)

    print(f"Input Allocations:\n{allocations}\n")
    if most_common:
        print(f"Most Common Allocation Clump Representative: {most_common}")
        print(f"Frequency of this Clump: {freq:.2f} ({freq*100:.0f}%)")
    else:
        print("The input list was empty.")

    print("-" * 20)

    # Example with less similarity
    allocations2 = [
        [10, 20, 30],
        [11, 21, 29], # Within 10% but not 5% of first element
        [50, 50, 50]
    ]

    most_common2, freq2 = calculate_most_frequent_list_fuzzy(allocations2, similarity_threshold=0.05)

    print(f"Input Allocations:\n{allocations2}\n")
    if most_common2:
        print(f"Most Common Allocation Clump Representative: {most_common2}")
        print(f"Frequency of this Clump: {freq2:.2f} ({freq2*100:.0f}%)") # Each will be its own clump
    else:
        print("The input list was empty.")

    print("-" * 20)

    # Example with zeros
    allocations3 = [
        [10, 0, 5],
        [10.1, 0.00001, 5.01], # Similar
        [10.5, 1.0, 5.2],     # Not similar (due to middle element)
        [9.8, 0, 4.9]          # Similar
    ]

    most_common3, freq3 = calculate_most_frequent_list_fuzzy(allocations3, similarity_threshold=0.05)

    print(f"Input Allocations:\n{allocations3}\n")
    if most_common3:
        print(f"Most Common Allocation Clump Representative: {most_common3}")
        print(f"Frequency of this Clump: {freq3:.2f} ({freq3*100:.0f}%)") # Should be 3/4 = 75%
    else:
        print("The input list was empty.")
        
    print("-" * 20) 
    print("Examples of using calculate_most_frequent_val_fuzzy:")
    print("--- Example 1: Basic Usage (Default Tolerance) ---")
    data1 = [10.0, 10.1, 10.2, 15.0, 15.1, 9.9]
    # Expected: Clump around 10 (10.0, 10.1, 10.2, 9.9) count=4, Clump around 15 (15.0, 15.1) count=2
    # Most frequent is the clump of 4. Average will be close to 10.05. Freq = 4/6 = 0.667
    avg1, freq1 = calculate_most_frequent_val_fuzzy(data1)
    print(f"Data: {data1}")
    print(f"Most Frequent Average: {avg1:.4f}")
    # Note: The frequency returned is sum_weights/total_count. Here weights are 1.
    print(f"Relative Frequency (sum_weights_in_clump / total_items): {freq1:.4f}")
    # You might also want the count-based frequency:
    mf_clump1 = max([c for c in calculate_most_frequent_val_fuzzy.__globals__['clumps']], key=lambda c: c['count']) # Access internal state for demo
    print(f"(For comparison) Count-based Frequency (count_in_clump / total_items): {mf_clump1['count']/len(data1):.4f}")
    print("-" * 20)


    print("--- Example 2: Higher Tolerance ---")
    data2 = [10, 11, 12, 18, 19, 20]
    # Default tolerance (2%): 11 vs 10 (10% diff) -> separate. 12 vs 10.5 (avg) -> ~14% diff -> separate. Likely 3 clumps.
    # Higher tolerance (15%):
    # 10 -> Clump 1 {avg: 10, count: 1}
    # 11 vs 10 (10% diff < 15%) -> Clump 1 {avg: 10.5, count: 2}
    # 12 vs 10.5 (abs(1.5) / 10.5 = 14.3% < 15%) -> Clump 1 {avg: 11, count: 3}
    # 18 vs 11 (large diff) -> Clump 2 {avg: 18, count: 1}
    # 19 vs 18 (5.5% diff < 15%) -> Clump 2 {avg: 18.5, count: 2}
    # 20 vs 18.5 (abs(1.5) / 18.5 = 8.1% < 15%) -> Clump 2 {avg: 19, count: 3}
    # Two clumps of count 3. `max` will likely pick the first one found (around 11).
    avg2, freq2 = calculate_most_frequent_val_fuzzy(data2, relative_tolerance=0.15)
    print(f"Data: {data2}, Tolerance: 0.15")
    print(f"Most Frequent Average: {avg2:.4f}") # Should be avg of [10, 11, 12] = 11.0
    print(f"Relative Frequency: {freq2:.4f}") # Should be 3/6 = 0.5
    print("-" * 20)


    print("--- Example 3: Using Weights ---")
    data3 = [10.0, 10.1, 25.0, 25.2, 25.1]
    weights3 = [10, 10,  1,    1,    1]
    # Clump 1: [10.0, 10.1] -> count=2, total_weight=20
    # Clump 2: [25.0, 25.2, 25.1] -> count=3, total_weight=3
    # Most frequent clump is Clump 2 (by count = 3).
    # Average should be around 25.1.
    # Returned frequency = sum_weights_in_clump_2 / total_items = 3 / 5 = 0.6
    avg3, freq3 = calculate_most_frequent_val_fuzzy(data3, weight_list=weights3)
    print(f"Data: {data3}")
    print(f"Weights: {weights3}")
    print(f"Most Frequent Average (based on count): {avg3:.4f}")
    print(f"Relative Frequency (sum_weights_in_clump / total_items): {freq3:.4f}")
    mf_clump3 = max([c for c in calculate_most_frequent_val_fuzzy.__globals__['clumps']], key=lambda c: c['count']) # Access internal state for demo
    print(f"(Clump Details) Count: {mf_clump3['count']}, Total Weight in Clump: {mf_clump3['weight']}")
    print("-" * 20)


    print("--- Example 4: All Distinct Values (Low Tolerance) ---")
    data4 = [10, 20, 30, 40]
    # With low tolerance, each likely forms its own clump.
    # `max` will likely return the first one found.
    # Expected: Avg = 10.0, Freq = 1/4 = 0.25
    avg4, freq4 = calculate_most_frequent_val_fuzzy(data4, relative_tolerance=0.01)
    print(f"Data: {data4}, Tolerance: 0.01")
    print(f"Most Frequent Average: {avg4:.4f}")
    print(f"Relative Frequency: {freq4:.4f}")
    print("-" * 20)


    print("--- Example 5: Single Element ---")
    data5 = [42.5]
    avg5, freq5 = calculate_most_frequent_val_fuzzy(data5)
    print(f"Data: {data5}")
    print(f"Most Frequent Average: {avg5:.4f}") # Should be 42.5
    print(f"Relative Frequency: {freq5:.4f}") # Should be 1/1 = 1.0
    print("-" * 20)


    print("--- Example 6: Zero and Near-Zero Values ---")
    data6 = [0.0, 0.01, -0.01, 5.0, 5.05, -0.005]
    # Tol = 0.02 (2%)
    # 0.0 -> Clump 1 {avg: 0, count: 1}
    # 0.01 vs 0 -> close? No (special case). -> Clump 2 {avg: 0.01, count: 1}
    # -0.01 vs 0 -> close? No. vs 0.01 -> abs(-0.02) <= 0.02 * 0.01 = 0.0002? No. -> Clump 3 {avg: -0.01, count: 1}
    # 5.0 vs 0, 0.01, -0.01 -> No. -> Clump 4 {avg: 5.0, count: 1}
    # 5.05 vs 5.0 -> abs(0.05) <= 0.02 * 5.0 = 0.1? Yes. -> Clump 4 {avg: 5.025, count: 2}
    # -0.005 vs 0 -> close? No. vs 0.01 -> abs(-0.015) <= 0.02 * 0.01 = 0.0002? No. vs -0.01 -> abs(0.005) <= 0.02 * abs(-0.01) = 0.0002? No. vs 5.025 -> No. -> Clump 5 {avg: -0.005, count: 1}
    # Most frequent is Clump 4 (count 2). Avg = 5.025. Freq = 2/6 = 0.333
    avg6, freq6 = calculate_most_frequent_val_fuzzy(data6, relative_tolerance=0.02)
    print(f"Data: {data6}, Tolerance: 0.02")
    print(f"Most Frequent Average: {avg6:.4f}")
    print(f"Relative Frequency: {freq6:.4f}")
    print("-" * 20)


def compute_predicted_counters(snapshot: Dict, paths: Dict, topology: NetworkTopology, paths_perturbed_num_nodes: int = 0) -> Dict[Tuple[int, int], float]:
    """
    Compute predicted counter values using paths and demands.
    
    Args:
        snapshot: Dict containing interface counter values
        paths: Dict containing paths between nodes
        topology: NetworkTopology instance containing network topology
        paths_perturbed_num_nodes: Number of nodes in the perturbed paths (default: 0). This is a bit of a kludge, but
            we want to be able to test what happens if our paths are buggy. This would happen for particular routers not
            reporting their hops correctly. This parameter corresponds to how many nodes in the topology misreport their paths.
        
    Returns:
        Dict mapping (src_id, dst_id) to predicted counter value
    """
    # Pick random nodes to 'perturb'
    node_ids_to_perturb = random.sample(topology.get_all_node_ids(), paths_perturbed_num_nodes)
    
    # Initialize counter predictions
    predictions = {}
    for node_id in topology.get_all_node_ids():
        for link_id in topology.get_incident_link_ids(node_id):
            predictions[link_id] = 0.0

    # For each source-dest pair
    for src_name in paths:
        for dst_name, wcmp_paths in paths[src_name].items():
            assert src_name != dst_name, "Source and destination names should not be the same"
                
            # Get demand value from snapshot
            demand_key = f"high_{src_name}_{dst_name}"
            
            # Get perturbed demand value
            assert demand_key in snapshot, f"Expected demand key {demand_key} not found in snapshot"
            # Check if demand is numberic
            
            assert isinstance(snapshot[demand_key], (int, float)), f"Expected demand value to be numeric, got {type(snapshot[demand_key])}"
            demand = snapshot[demand_key]
            
            # Process each WCMP path
            for path_info in wcmp_paths:
                path = path_info['path']
                
                # Place portion of demand on this path.
                flow_amount = demand * path_info['weight']
                
                # Add flow amount to ingress and egress interfaces
                flow_origin_id = topology.node_name_to_id[src_name]
                flow_destination_id = topology.node_name_to_id[dst_name]
                predictions[(-1, flow_origin_id)] += flow_amount  # Ingress from external interface at src
                predictions[(flow_destination_id, -1)] += flow_amount  # Egress to external interface at dst
                
                # Add flow amount to each link along the path
                for i in range(len(path)-1):
                    src_node = path[i]
                    dst_node = path[i+1]
                    src_id = topology.node_name_to_id[src_node]
                    dst_id = topology.node_name_to_id[dst_node]
                    
                    assert src_id != -1 and dst_id != -1, 'Path hops should not be external interfaces'
                    
                    if src_id in node_ids_to_perturb:
                        # Skip adding this hop to the prediction
                        continue
                    
                    # Check if link exists in topology
                    if (src_id, dst_id) not in predictions:
                        raise ValueError(f"Path uses link {src_node}->{dst_node} which does not exist in topology")
                    
                    predictions[(src_id, dst_id)] += flow_amount

    return predictions

def load_paths_for_timestamp(paths_path: str, timestamp: str) -> Dict:
    """Load paths for a given timestamp. If paths_path is a directory, load the file for the given timestamp."""
    if os.path.isfile(paths_path):
        path = paths_path
    else:
        # Convert timestamp from "YYYY/MM/DD HH:MM UTC" to "YYYY-MM-DD_HH_MM_00"
        dt = datetime.strptime(timestamp, "%Y/%m/%d %H:%M UTC")
        filename = dt.strftime("paths_%Y-%m-%d_%H_%M_00.json")
        path = os.path.join(paths_path, filename)
    
    try:
        with open(path) as f:
            return json.load(f)["paths"]
    except FileNotFoundError:
        raise ValueError(f"Paths file not found: {path}")

def extract_perturbed_snapshot(row: pd.Series) -> Dict:
    """Create a snapshot with perturbed values from a row, default to ground truth if perturbed is None."""
    snapshot = {}
    
    # Copy non-dictionary columns as is
    snapshot['timestamp'] = row['timestamp']
    snapshot['telemetry_perturbed_type'] = row['telemetry_perturbed_type']
    snapshot['input_perturbed_type'] = row['input_perturbed_type']
    snapshot['true_detect_inconsistent'] = row['true_detect_inconsistent']
    
    # Extract perturbed values from dictionary columns
    for col in row.index:
        if col not in snapshot:  # Skip the columns we already processed
            val = row[col]
            if not isinstance(val, dict):
                # For non-dictionary columns, use the value directly
                snapshot[col] = val
            else:
                if 'perturbed' in val and val['perturbed'] is not None:
                    # For dictionary columns, use the perturbed value
                    snapshot[col] = val['perturbed']
                elif 'ground_truth' in val and val['ground_truth'] is not None:
                    # If perturbed is None, use ground truth as a fallback
                    snapshot[col] = val['ground_truth']
                else:
                    # If both are None or invalid, skip this value
                    print(f"Error: no value at all for {col}")
    return snapshot
        
if __name__ == "__main__":
    main()
    