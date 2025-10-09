"""
Self-contained network interface telemetry repair using Hodor algorithm.

This implementation contains the complete Hodor repair system in a single file:
- Democratic Trust Propagation (DTP) for repair
- Network topology-aware corrections
- All dependencies inlined for OpenEvolve evolution

Combines the key components from the reference implementation.
"""
# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import copy
import random
from dataclasses import dataclass


# ============================================================================
# NETWORK TOPOLOGY UTILITIES (from common_utils.py)
# ============================================================================

class NetworkTopology:
    """Network topology representation for repair algorithms."""
    
    def __init__(self, topology_dict: Dict):
        """Initialize with network topology JSON structure."""
        self.topology = topology_dict
        self.nodes = {node['id']: node for node in topology_dict['nodes']}
        self.node_name_to_id = {node['name']: node['id'] for node in topology_dict['nodes']}
        
        # Build adjacency lists
        self.adjacency = {node_id: [] for node_id in self.nodes}
        
        # Add regular interfaces from topology links
        for link in topology_dict['links']:
            src, dst = link['source'], link['target']
            self.adjacency[src].append(dst)
            self.adjacency[dst].append(src)
        
        # Add external interfaces (represented as -1)
        for node_id in self.nodes:
            self.adjacency[node_id].append(-1)  # External interface
        self.adjacency[-1] = list(self.nodes.keys())  # External node connections

    def get_all_node_ids(self) -> List[int]:
        """Get all node IDs including external (-1)."""
        return list(self.nodes.keys()) + [-1]

    def get_node_name(self, node_id: int) -> str:
        """Get node name by ID."""
        if node_id == -1:
            return "external"
        return self.nodes[node_id]['name']

    def get_snapshot_keys_for_link(self, link_id: Tuple[int, int]) -> Tuple[Optional[str], Optional[str]]:
        """Get the snapshot keys for looking up (in_value, out_value) for a link."""
        src_id, dst_id = link_id
        
        # Handle external links
        if dst_id == -1:  # Internal -> external
            node_name = self.get_node_name(src_id)
            in_key = f"low_{node_name}_termination"
            out_key = None
            return (in_key, out_key)
        elif src_id == -1:  # External -> internal
            node_name = self.get_node_name(dst_id)
            in_key = None
            out_key = f"low_{node_name}_origination"
            return (in_key, out_key)
        else:  # Regular internal link
            src_name = self.get_node_name(src_id)
            dst_name = self.get_node_name(dst_id)
            in_key = f"low_{src_name}_egress_to_{dst_name}"
            out_key = f"low_{dst_name}_ingress_from_{src_name}"
            return (in_key, out_key)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get all neighbor IDs for a node."""
        return self.adjacency[node_id]

    def get_interface_name(self, node_id: int, neighbor_id: int) -> str:
        """Get the interface name for a link between two nodes."""
        if neighbor_id == -1:
            return f"{self.get_node_name(node_id)}_external"
        elif node_id == -1:
            return f"external_{self.get_node_name(neighbor_id)}"
        else:
            return f"{self.get_node_name(node_id)}_to_{self.get_node_name(neighbor_id)}"


def extract_perturbed_snapshot(row: pd.Series) -> Dict:
    """Create a snapshot with perturbed values from a row."""
    snapshot = {}
    
    # Copy non-dictionary columns
    snapshot['timestamp'] = row['timestamp']
    snapshot['telemetry_perturbed_type'] = row['telemetry_perturbed_type']
    snapshot['input_perturbed_type'] = row['input_perturbed_type']
    snapshot['true_detect_inconsistent'] = row['true_detect_inconsistent']
    
    # Extract perturbed values from dictionary columns
    for col in row.index:
        if col not in snapshot:
            val = row[col]
            if not isinstance(val, dict):
                snapshot[col] = val
            else:
                if 'perturbed' in val and val['perturbed'] is not None:
                    snapshot[col] = val['perturbed']
                else:
                    snapshot[col] = val['ground_truth']
    
    return snapshot


# ============================================================================
# DTP REPAIR ALGORITHM (from dtp.py)
# ============================================================================

@dataclass
class RepairConfig:
    """Configuration for DTP repair algorithm."""
    disable_cache: bool = False
    interface_threshold: float = 0.03
    num_trials: int = 30
    similarity_threshold: float = 0.05
    seed: int = 42


class LinkValue:
    """Represents a link with in/out values and repair information."""
    
    def __init__(self, link_id: Tuple[int, int], in_val: Optional[float], out_val: Optional[float], 
                 demand_based_prediction: Optional[float] = None, name: str = ""):
        self.link_id = link_id
        self.in_val = in_val
        self.out_val = out_val
        self.demand_based_prediction = demand_based_prediction
        self.name = name
        self.confidence = 0.5  # Default confidence


class DemocraticTrustPropagationRepair:
    """Democratic Trust Propagation repair algorithm."""
    
    def __init__(self, topology: Dict, config: RepairConfig = RepairConfig()):
        self.network_topology = NetworkTopology(topology)
        self.config = config
        random.seed(config.seed)

    def process_row(self, row: pd.Series, paths: Dict = None) -> pd.Series:
        """Process a single row of network data."""
        snapshot = extract_perturbed_snapshot(row)
        repair_results = self.repair_snapshot(snapshot, paths)
        return self._update_row_with_repairs(row, repair_results)

    def repair_snapshot(self, snapshot: Dict, paths: Dict = None) -> Dict:
        """Repair a network snapshot using DTP algorithm."""
        # Initialize links from snapshot
        links = self._initialize_links(snapshot, paths)
        
        # Run consensus algorithm
        repaired_links = self._run_consensus_algorithm(links)
        
        # Convert back to corrected values format
        corrected_values = {}
        for link_id, link in repaired_links.items():
            if link.in_val is not None and link.out_val is not None:
                interface_name = self.network_topology.get_interface_name(link_id[0], link_id[1])
                corrected_values[interface_name] = (link.in_val, link.out_val)
        
        return {
            'corrected_values': corrected_values,
            'method': 'DTP'
        }

    def _initialize_links(self, snapshot: Dict, paths: Dict = None) -> Dict[Tuple[int, int], LinkValue]:
        """Initialize link values from snapshot data."""
        links = {}
        
        for src_id in self.network_topology.get_all_node_ids():
            for dst_id in self.network_topology.get_neighbors(src_id):
                if src_id == dst_id:
                    continue
                    
                link_id = (src_id, dst_id)
                if link_id in links:
                    continue
                    
                links[link_id] = self._get_link_value(snapshot, link_id, None)
        
        return links

    def _get_link_value(self, snapshot: Dict, link_id: Tuple[int, int], 
                       demand_based_predictions: Dict = None) -> LinkValue:
        """Get link value from snapshot data."""
        src_id, dst_id = link_id
        
        # Get the snapshot keys from topology
        in_key, out_key = self.network_topology.get_snapshot_keys_for_link(link_id)
        interface_name = self.network_topology.get_interface_name(src_id, dst_id)
        
        # Look up values in snapshot
        in_val = snapshot[in_key] if in_key is not None and in_key in snapshot else None
        out_val = snapshot[out_key] if out_key is not None and out_key in snapshot else None
        
        return LinkValue(link_id, in_val, out_val, name=interface_name)

    def _run_consensus_algorithm(self, links: Dict[Tuple[int, int], LinkValue]) -> Dict[Tuple[int, int], LinkValue]:
        """Run the democratic consensus algorithm."""
        # Collect votes from all routers
        votes = self._collect_votes(links)
        
        # Apply consensus to each link
        repaired_links = copy.deepcopy(links)
        
        for link_id, link in repaired_links.items():
            if link_id in votes and len(votes[link_id]) > 0:
                # Use average of votes as repaired value
                avg_vote = sum(votes[link_id]) / len(votes[link_id])
                
                # Apply repair if values exist
                if link.in_val is not None:
                    link.in_val = avg_vote
                if link.out_val is not None:
                    link.out_val = avg_vote
                    
                # Set confidence based on vote consistency
                if len(votes[link_id]) > 1:
                    vote_std = (sum((v - avg_vote) ** 2 for v in votes[link_id]) / len(votes[link_id])) ** 0.5
                    link.confidence = max(0.1, 1.0 - (vote_std / (avg_vote + 1e-6)))
                else:
                    link.confidence = 0.5
        
        return repaired_links

    def _collect_votes(self, links: Dict[Tuple[int, int], LinkValue]) -> Dict[Tuple[int, int], List[float]]:
        """Collect votes from each router about link values."""
        votes = {}
        
        for node_id in self.network_topology.get_all_node_ids():
            if node_id == -1:  # Skip external node
                continue
                
            # Try different assignments for this router
            router_beliefs = self._try_router_assignments(node_id, links)
            
            # Add beliefs as votes
            for link_id, belief_values in router_beliefs.items():
                if link_id not in votes:
                    votes[link_id] = []
                
                # Use the average of in/out values as the vote
                if len(belief_values) == 2:
                    vote = (belief_values[0] + belief_values[1]) / 2
                    votes[link_id].append(vote)
        
        return votes

    def _try_router_assignments(self, node_id: int, 
                               link_states: Dict[Tuple[int, int], LinkValue]) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Try different assignments of interface values for a router."""
        router_links = []
        
        # Find all links connected to this router
        for link_id, link in link_states.items():
            if link_id[0] == node_id or link_id[1] == node_id:
                router_links.append((link_id, link))
        
        if not router_links:
            return {}
        
        # Simple assignment: use flow conservation principle
        beliefs = {}
        total_in = 0
        total_out = 0
        valid_links = []
        
        for link_id, link in router_links:
            if link.in_val is not None and link.out_val is not None:
                valid_links.append((link_id, link))
                if link_id[1] == node_id:  # Incoming to this node
                    total_in += link.in_val
                else:  # Outgoing from this node
                    total_out += link.out_val
        
        # Apply flow conservation: total_in should approximately equal total_out
        if len(valid_links) > 1 and total_in > 0 and total_out > 0:
            balance_factor = (total_in + total_out) / (2 * len(valid_links))
            
            for link_id, link in valid_links:
                # Adjust values toward balanced flow
                adjusted_in = (link.in_val + balance_factor) / 2
                adjusted_out = (link.out_val + balance_factor) / 2
                beliefs[link_id] = (adjusted_in, adjusted_out)
        
        return beliefs

    def _update_row_with_repairs(self, row: pd.Series, repair_results: Dict) -> pd.Series:
        """Update a row with repaired values and confidences."""
        output_row = copy.deepcopy(row)
        
        # Set repair type
        output_row['repair_type'] = repair_results.get('method', 'DTP')
        
        # Update the row with repaired values
        for interface_name, (in_val, out_val) in repair_results['corrected_values'].items():
            if interface_name.endswith('_external'):
                # For external interfaces
                node_name = interface_name[:-9]
                if f"low_{node_name}_origination" in output_row:
                    if isinstance(output_row[f"low_{node_name}_origination"], dict):
                        output_row[f"low_{node_name}_origination"]['corrected'] = in_val
                        output_row[f"low_{node_name}_origination"]['confidence'] = 0.6
                if f"low_{node_name}_termination" in output_row:
                    if isinstance(output_row[f"low_{node_name}_termination"], dict):
                        output_row[f"low_{node_name}_termination"]['corrected'] = out_val
                        output_row[f"low_{node_name}_termination"]['confidence'] = 0.6
            else:
                # For regular interfaces
                if '_egress_to_' in interface_name:
                    parts = interface_name.split('_to_')
                    if len(parts) == 2:
                        src, dst = parts[0], parts[1]
                        egress_key = f"low_{src}_egress_to_{dst}"
                        ingress_key = f"low_{dst}_ingress_from_{src}"
                        
                        if egress_key in output_row and isinstance(output_row[egress_key], dict):
                            output_row[egress_key]['corrected'] = out_val
                            output_row[egress_key]['confidence'] = 0.6
                        if ingress_key in output_row and isinstance(output_row[ingress_key], dict):
                            output_row[ingress_key]['corrected'] = in_val
                            output_row[ingress_key]['confidence'] = 0.6
        
        return output_row


# ============================================================================
# HODOR ORCHESTRATION (from hodor.py)
# ============================================================================

@dataclass
class HodorConfig:
    """Configuration for the Hodor pipeline."""
    repair_config: RepairConfig = None
    disable_cache: bool = False
    
    def __post_init__(self):
        if self.repair_config is None:
            self.repair_config = RepairConfig(disable_cache=self.disable_cache)


class Hodor:
    """Hodor: The unified repair pipeline."""
    
    def __init__(self, topology: Dict, config: HodorConfig = HodorConfig()):
        self.network_topology = NetworkTopology(topology)
        self.config = config
        self.repairer = DemocraticTrustPropagationRepair(topology, config.repair_config)
        print(f"🛡️  Hodor initialized with {len(topology.get('nodes', []))} nodes")

    def repair_row(self, row: pd.Series, paths: Dict = None) -> pd.Series:
        """Repair a single row using the DTP algorithm."""
        print("🔧 Hodor repairing single row...")
        repaired_row = self.repairer.process_row(row, paths)
        print("✅ Single row repair complete")
        return repaired_row


# ============================================================================
# OPENEVOLVE INTERFACE
# ============================================================================

def convert_interfaces_to_csv_row(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Convert OpenEvolve interface format to CSV row format expected by Hodor."""
    csv_row = {
        'timestamp': '2024/01/01 12:00 UTC',
        'telemetry_perturbed_type': 'unknown',
        'input_perturbed_type': 'unknown',
        'true_detect_inconsistent': False
    }
    
    # Track router-level aggregates for termination/origination
    router_termination = {}
    router_origination = {}
    
    for interface_id, data in telemetry.items():
        if '_to_' not in interface_id:
            continue
            
        parts = interface_id.split('_to_')
        if len(parts) != 2:
            continue
            
        source, dest = parts
        
        # Create egress (TX) column
        egress_col = f"low_{source}_egress_to_{dest}"
        tx_rate = data.get('tx_rate', 0.0)
        csv_row[egress_col] = {
            'ground_truth': tx_rate,
            'perturbed': tx_rate,
            'corrected': None,
            'confidence': None
        }
        
        # Create ingress (RX) column
        ingress_col = f"low_{dest}_ingress_from_{source}"
        rx_rate = data.get('rx_rate', 0.0)
        csv_row[ingress_col] = {
            'ground_truth': rx_rate,
            'perturbed': rx_rate,
            'corrected': None,
            'confidence': None
        }
        
        # Accumulate router-level traffic
        router_origination[source] = router_origination.get(source, 0.0) + tx_rate
        router_termination[dest] = router_termination.get(dest, 0.0) + rx_rate
    
    # Add router-level termination/origination fields
    all_routers = set(router_termination.keys()) | set(router_origination.keys())
    for router in all_routers:
        # Termination
        term_rate = router_termination.get(router, 0.0)
        csv_row[f"low_{router}_termination"] = {
            'ground_truth': term_rate,
            'perturbed': term_rate,
            'corrected': None,
            'confidence': None
        }
        
        # Origination
        orig_rate = router_origination.get(router, 0.0)
        csv_row[f"low_{router}_origination"] = {
            'ground_truth': orig_rate,
            'perturbed': orig_rate,
            'corrected': None,
            'confidence': None
        }
    
    return csv_row


def convert_csv_row_to_interface_results(csv_row: pd.Series, original_telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple]]:
    """Convert repaired CSV row results back to OpenEvolve interface format."""
    result = {}
    
    for interface_id, original_data in original_telemetry.items():
        if '_to_' not in interface_id:
            continue
            
        parts = interface_id.split('_to_')
        if len(parts) != 2:
            continue
            
        source, dest = parts
        repaired_data = {}
        
        # Process TX data (egress)
        egress_col = f"low_{source}_egress_to_{dest}"
        if egress_col in csv_row:
            egress_info = csv_row[egress_col]
            if isinstance(egress_info, dict):
                original_tx = float(egress_info.get('perturbed', original_data.get('tx_rate', 0.0)))
                repaired_tx = float(egress_info.get('corrected', original_tx)) if egress_info.get('corrected') is not None else original_tx
                confidence_tx = float(egress_info.get('confidence', 0.5)) if egress_info.get('confidence') is not None else 0.5
                repaired_data['tx_rate'] = (original_tx, repaired_tx, confidence_tx)
            else:
                original_tx = original_data.get('tx_rate', 0.0)
                repaired_data['tx_rate'] = (original_tx, original_tx, 0.5)
        else:
            original_tx = original_data.get('tx_rate', 0.0)
            repaired_data['tx_rate'] = (original_tx, original_tx, 0.5)
        
        # Process RX data (ingress)
        ingress_col = f"low_{dest}_ingress_from_{source}"
        if ingress_col in csv_row:
            ingress_info = csv_row[ingress_col]
            if isinstance(ingress_info, dict):
                original_rx = float(ingress_info.get('perturbed', original_data.get('rx_rate', 0.0)))
                repaired_rx = float(ingress_info.get('corrected', original_rx)) if ingress_info.get('corrected') is not None else original_rx
                confidence_rx = float(ingress_info.get('confidence', 0.5)) if ingress_info.get('confidence') is not None else 0.5
                repaired_data['rx_rate'] = (original_rx, repaired_rx, confidence_rx)
            else:
                original_rx = original_data.get('rx_rate', 0.0)
                repaired_data['rx_rate'] = (original_rx, original_rx, 0.5)
        else:
            original_rx = original_data.get('rx_rate', 0.0)
            repaired_data['rx_rate'] = (original_rx, original_rx, 0.5)
        
        # Handle interface status
        original_status = original_data.get('interface_status', 'up')
        repaired_data['interface_status'] = (original_status, original_status, 1.0)
        
        # Copy metadata unchanged
        repaired_data['connected_to'] = original_data.get('connected_to')
        repaired_data['local_router'] = original_data.get('local_router')
        repaired_data['remote_router'] = original_data.get('remote_router')
        
        result[interface_id] = repaired_data
    
    return result


def build_topology_dict(telemetry: Dict[str, Dict[str, Any]]) -> Dict:
    """Build topology dictionary from telemetry data for Hodor."""
    # Extract unique routers and their connections
    routers = set()
    connections = set()
    
    for interface_id, data in telemetry.items():
        local_router = data.get('local_router')
        remote_router = data.get('remote_router')
        
        if local_router and remote_router:
            routers.add(local_router)
            routers.add(remote_router)
            connection = tuple(sorted([local_router, remote_router]))
            connections.add(connection)
    
    # Build topology in the format expected by NetworkTopology
    router_list = sorted(list(routers))
    router_to_id = {router: i for i, router in enumerate(router_list)}
    
    topology = {
        'nodes': [
            {'id': router_to_id[router], 'name': router}
            for router in router_list
        ],
        'links': [
            {
                'source': router_to_id[router1], 
                'target': router_to_id[router2]
            }
            for router1, router2 in connections
        ]
    }
    
    return topology


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry using the self-contained Hodor algorithm.
    
    This function contains the complete Democratic Trust Propagation repair system.
    OpenEvolve can modify this entire implementation to evolve better algorithms.
    
    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down" 
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids
        
    Returns:
        Dictionary with same structure but values become tuples of:
        (original_value, repaired_value, confidence_score)
    """
    
    # Convert interface format to CSV row format
    csv_row_data = convert_interfaces_to_csv_row(telemetry)
    csv_row = pd.Series(csv_row_data)
    
    # Build topology for Hodor
    hodor_topology = build_topology_dict(telemetry)
    
    # Configure Hodor with reasonable settings
    repair_config = RepairConfig(
        disable_cache=True,
        interface_threshold=0.03,
        num_trials=10,
        similarity_threshold=0.05,
        seed=42
    )
    
    hodor_config = HodorConfig(
        repair_config=repair_config,
        disable_cache=True
    )
    
    # Initialize Hodor
    hodor = Hodor(hodor_topology, hodor_config)
    
    # Run repair only (skip validation)
    repaired_row = hodor.repair_row(csv_row)
    
    # Convert results back to interface format
    result = convert_csv_row_to_interface_results(repaired_row, telemetry)
    
    return result

# EVOLVE-BLOCK-END


def run_repair(telemetry: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Main entry point called by the OpenEvolve evaluator.
    """
    print("🛡️  Running self-contained Hodor repair algorithm...")
    
    repaired_interfaces = repair_network_telemetry(telemetry, topology)
    
    print("✅  Self-contained Hodor repair complete")
    
    return repaired_interfaces 