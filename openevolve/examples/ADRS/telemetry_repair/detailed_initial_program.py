"""
Simple network interface telemetry repair. 

Takes interface telemetry data and then detects and repairs any inconsistencies.
Returns the same data structure with repairs and confidence scores for each measurement.
"""
# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple


def repair_network_telemetry(interfaces: Dict[str, Dict[str, Any]], 
                              tolerance_percent: float = 10.0) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting internal inconsistencies.
    
    Args:
        interfaces: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down" 
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - capacity: interface capacity in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
            
        tolerance_percent: Allowed percentage difference for rx/tx validation
        
    Returns:
        Dictionary with same structure but values become tuples of:
        (original_value, repaired_value, confidence_score)
    """
    
    result = {}
    
    for interface_id, telemetry in interfaces.items():
        repaired_telemetry = {}
        
        # Get basic telemetry values
        interface_status = telemetry.get('interface_status', 'unknown')
        rx_rate = telemetry.get('rx_rate', 0.0)
        tx_rate = telemetry.get('tx_rate', 0.0) 
        capacity = telemetry.get('capacity', 0.0)
        connected_to = telemetry.get('connected_to')
        
        # Super naive repair algorithm – if there is disagreement between two interfaces, take the lower one. 
        # It'd be nice if we had some better way of telling which one is the correct one.
        
        # Compare own rx_rate to neighbor tx_rate:
        neighbor_tx_rate = interfaces[connected_to].get('tx_rate', 0.0)
        repair_rx_rate = min(rx_rate, neighbor_tx_rate)
        repaired_telemetry['rx_rate'] = (rx_rate, repair_rx_rate, 1.0)
        
        # Compare own tx_rate to neighbor rx_rate:
        neighbor_rx_rate = interfaces[connected_to].get('rx_rate', 0.0)
        repair_tx_rate = min(tx_rate, neighbor_rx_rate)
        repaired_telemetry['tx_rate'] = (tx_rate, repair_tx_rate, 1.0)
        
        # No repair for status currently, but we should do something...
        repaired_telemetry['interface_status'] = (interface_status, interface_status, 1.0)
        
        # Copy metadata unchanged
        repaired_telemetry['capacity'] = capacity
        repaired_telemetry['connected_to'] = connected_to
        repaired_telemetry['local_router'] = telemetry.get('local_router')
        repaired_telemetry['remote_router'] = telemetry.get('remote_router')
        
        result[interface_id] = repaired_telemetry
    
    return result

# EVOLVE-BLOCK-END


def run_repair(interfaces: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main entry point that will be called by the OpenEvolve evaluator.
    
    Args:
        interfaces: Network interface telemetry data
        
    Returns:
        Dictionary containing validated results and summary stats
    """
    
    # Run validation with 10% tolerance
    repaired_interfaces = repair_network_telemetry(interfaces, tolerance_percent=10.0)
    
    return repaired_interfaces


if __name__ == "__main__":
    # Test with simple example data
    print("Testing network interface validation...")
    
    test_interfaces = {
        'if1': {
            'interface_status': 'up',
            'rx_rate': 100.0,
            'tx_rate': 95.0,
            'capacity': 1000.0,
            'connected_to': 'if2',
            'local_router': 'router1',
            'remote_router': 'router2'
        },
        'if2': {
            'interface_status': 'up', 
            'rx_rate': 94.0,  # Should be close to if1's tx_rate (95.0)
            'tx_rate': 102.0,  # Should be close to if1's rx_rate (100.0)
            'capacity': 1000.0,
            'connected_to': 'if1',
            'local_router': 'router2', 
            'remote_router': 'router1'
        },
        'if3': {
            'interface_status': 'down',
            'rx_rate': 0.0,
            'tx_rate': 0.0,
            'capacity': 1000.0,
            'connected_to': 'if4',
            'local_router': 'router1',
            'remote_router': 'router3'  
        },
        'if4': {
            'interface_status': 'down',
            'rx_rate': 0.0,
            'tx_rate': 0.0, 
            'capacity': 1000.0,
            'connected_to': 'if3',
            'local_router': 'router3',
            'remote_router': 'router1'
        }
    }
    
    result = run_repair(test_interfaces)
    
    print("\nValidation Results:")
    for interface_id, telemetry in result.items():
        print(f"\n{interface_id}:")
        for key, value in telemetry.items():
            if isinstance(value, tuple) and len(value) == 3:
                orig, repaired, confidence = value
                print(f"  {key}: {orig} -> {repaired} (confidence: {confidence:.2f})")
            else:
                print(f"  {key}: {value}")