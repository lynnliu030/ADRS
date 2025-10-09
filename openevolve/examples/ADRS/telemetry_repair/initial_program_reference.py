"""
Network interface telemetry repair using hand-crafted Hodor algorithm.

This implementation uses the user's reference Hodor system for repair only:
- Democratic Trust Propagation (DTP) for repair
- Network topology-aware corrections
- Skips validation step for simplicity

Adapts between OpenEvolve's interface format and the CSV/DataFrame format
used by the reference implementation.
"""
# EVOLVE-BLOCK-START
import sys
import os
from typing import Dict, Any, Tuple, List
import pandas as pd
import ast
import copy

# Add ref_impl to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'ref_impl'))

try:
    from hodor import Hodor, HodorConfig
    from dtp import RepairConfig
    from validate import ValidatorConfig
    from common_utils import NetworkTopology
    print("✅ Successfully imported Hodor reference implementation")
except ImportError as e:
    print(f"❌ Could not import reference implementation: {e}")
    Hodor = None


def convert_interfaces_to_csv_row(telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Convert OpenEvolve interface format to CSV row format expected by Hodor.
    
    Args:
        telemetry: Interface telemetry in OpenEvolve format
        
    Returns:
        Dictionary in CSV format with low_SOURCE_egress_to_DEST columns
    """
    csv_row = {
        'timestamp': '2024/01/01 12:00 UTC',  # Dummy timestamp
        'telemetry_perturbed_type': 'unknown',  # Required by Hodor
        'input_perturbed_type': 'unknown',      # Required by Hodor
        'true_detect_inconsistent': False       # Required by Hodor
    }
    
    # Track router-level aggregates for termination/origination
    router_termination = {}  # Total incoming traffic per router
    router_origination = {}  # Total outgoing traffic per router
    
    for interface_id, data in telemetry.items():
        # Extract router names from interface_id (format: RouterA_to_RouterB)
        if '_to_' not in interface_id:
            continue
            
        parts = interface_id.split('_to_')
        if len(parts) != 2:
            continue
            
        source, dest = parts
        
        # Create egress (TX) column: low_SOURCE_egress_to_DEST
        egress_col = f"low_{source}_egress_to_{dest}"
        tx_rate = data.get('tx_rate', 0.0)
        egress_data = {
            'ground_truth': tx_rate,
            'perturbed': tx_rate,  # Use current value as "perturbed"
            'corrected': None,
            'confidence': None
        }
        csv_row[egress_col] = egress_data  # Store as dict, not string
        
        # Create ingress (RX) column: low_DEST_ingress_from_SOURCE  
        ingress_col = f"low_{dest}_ingress_from_{source}"
        rx_rate = data.get('rx_rate', 0.0)
        ingress_data = {
            'ground_truth': rx_rate,
            'perturbed': rx_rate,  # Use current value as "perturbed"
            'corrected': None,
            'confidence': None
        }
        csv_row[ingress_col] = ingress_data  # Store as dict, not string
        
        # Accumulate router-level traffic
        router_origination[source] = router_origination.get(source, 0.0) + tx_rate
        router_termination[dest] = router_termination.get(dest, 0.0) + rx_rate
    
    # Add router-level termination/origination fields (required for external traffic)
    all_routers = set(router_termination.keys()) | set(router_origination.keys())
    for router in all_routers:
        # Termination: total incoming traffic to this router
        term_rate = router_termination.get(router, 0.0)
        term_col = f"low_{router}_termination"
        term_data = {
            'ground_truth': term_rate,
            'perturbed': term_rate,
            'corrected': None,
            'confidence': None
        }
        csv_row[term_col] = term_data  # Store as dict, not string
        
        # Origination: total outgoing traffic from this router
        orig_rate = router_origination.get(router, 0.0)
        orig_col = f"low_{router}_origination"
        orig_data = {
            'ground_truth': orig_rate,
            'perturbed': orig_rate,
            'corrected': None,
            'confidence': None
        }
        csv_row[orig_col] = orig_data  # Store as dict, not string
    
    return csv_row


def convert_csv_row_to_interface_results(csv_row: pd.Series, original_telemetry: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Convert repaired CSV row results back to OpenEvolve interface format.
    
    Args:
        csv_row: Repaired pandas Series from Hodor
        original_telemetry: Original telemetry data for reference
        
    Returns:
        Dictionary with (original, repaired, confidence) tuples
    """
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
            egress_info = csv_row[egress_col]  # Now it's already a dict
            if isinstance(egress_info, dict):
                original_tx = float(egress_info.get('perturbed', original_data.get('tx_rate', 0.0)))
                repaired_tx = float(egress_info.get('corrected', original_tx)) if egress_info.get('corrected') is not None else original_tx
                confidence_tx = float(egress_info.get('confidence', 0.5)) if egress_info.get('confidence') is not None else 0.5
                repaired_data['tx_rate'] = (original_tx, repaired_tx, confidence_tx)
            else:
                # Fallback if not a dict
                original_tx = original_data.get('tx_rate', 0.0)
                repaired_data['tx_rate'] = (original_tx, original_tx, 0.5)
        else:
            original_tx = original_data.get('tx_rate', 0.0)
            repaired_data['tx_rate'] = (original_tx, original_tx, 0.5)
        
        # Process RX data (ingress)
        ingress_col = f"low_{dest}_ingress_from_{source}"
        if ingress_col in csv_row:
            ingress_info = csv_row[ingress_col]  # Now it's already a dict
            if isinstance(ingress_info, dict):
                original_rx = float(ingress_info.get('perturbed', original_data.get('rx_rate', 0.0)))
                repaired_rx = float(ingress_info.get('corrected', original_rx)) if ingress_info.get('corrected') is not None else original_rx
                confidence_rx = float(ingress_info.get('confidence', 0.5)) if ingress_info.get('confidence') is not None else 0.5
                repaired_data['rx_rate'] = (original_rx, repaired_rx, confidence_rx)
            else:
                # Fallback if not a dict
                original_rx = original_data.get('rx_rate', 0.0)
                repaired_data['rx_rate'] = (original_rx, original_rx, 0.5)
        else:
            original_rx = original_data.get('rx_rate', 0.0)
            repaired_data['rx_rate'] = (original_rx, original_rx, 0.5)
        
        # Handle interface status (assume always up with high confidence)
        original_status = original_data.get('interface_status', 'up')
        repaired_data['interface_status'] = (original_status, original_status, 1.0)
        
        # Copy metadata unchanged
        repaired_data['connected_to'] = original_data.get('connected_to')
        repaired_data['local_router'] = original_data.get('local_router')
        repaired_data['remote_router'] = original_data.get('remote_router')
        
        result[interface_id] = repaired_data
    
    return result


def build_topology_dict(telemetry: Dict[str, Dict[str, Any]]) -> Dict:
    """
    Build topology dictionary from telemetry data for Hodor.
    
    Args:
        telemetry: Interface telemetry data
        
    Returns:
        Topology dictionary suitable for NetworkTopology class
    """
    # Extract unique routers and their connections
    routers = set()
    connections = set()  # Use set to avoid duplicates
    
    for interface_id, data in telemetry.items():
        local_router = data.get('local_router')
        remote_router = data.get('remote_router')
        
        if local_router and remote_router:
            routers.add(local_router)
            routers.add(remote_router)
            # Add connection tuple (sorted to avoid duplicates like A->B and B->A)
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
    Repair network interface telemetry using the hand-crafted Hodor algorithm.
    
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
    
    # Check if Hodor is available
    if Hodor is None:
        raise ImportError("Reference Hodor implementation not available! Check ref_impl imports.")
    
    try:
        # Convert interface format to CSV row format
        csv_row_data = convert_interfaces_to_csv_row(telemetry)
        csv_row = pd.Series(csv_row_data)
        
        # Build topology for Hodor
        hodor_topology = build_topology_dict(telemetry)
        
        # Configure Hodor with reasonable settings
        repair_config = RepairConfig(
            disable_cache=True,  # Disable caching for single-row processing
            interface_threshold=0.03,
            num_trials=10,  # Reduced for performance
            similarity_threshold=0.05,
            seed=42
        )
        
        validator_config = ValidatorConfig(
            disable_cache=True,
            confidence_cutoff=0.0
        )
        
        hodor_config = HodorConfig(
            repair_config=repair_config,
            validator_config=validator_config,
            disable_cache=True
        )
        
        # Initialize Hodor
        hodor = Hodor(hodor_topology, hodor_config)
        
        # Run repair only (skip validation)
        repaired_row = hodor.repair_row(csv_row)
        
        # Convert results back to interface format
        result = convert_csv_row_to_interface_results(repaired_row, telemetry)
        
        return result
        
    except Exception as e:
        print(f"❌ Hodor processing failed: {e}")
        raise RuntimeError(f"Hodor algorithm failed: {e}") from e

# EVOLVE-BLOCK-END


def run_repair(telemetry: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Main entry point that will be called by the OpenEvolve evaluator.
    
    Args:
        telemetry: Network interface telemetry data
        topology: Dictionary where key is router_id and value contains a list of interface_ids
    Returns:
        Dictionary containing repaired results using hand-crafted Hodor algorithm
    """
    
    print("🛡️  Running hand-crafted Hodor repair algorithm (repair only)...")
    
    # Run repair using the reference implementation
    repaired_interfaces = repair_network_telemetry(telemetry, topology)
    
    print("✅  Hodor repair complete")
    
    return repaired_interfaces 