"""
Unit tests for the network input validation evaluator.

Tests the data conversion and evaluation logic with simple mock data.
"""

import unittest
from evaluator import convert_csv_row_to_interfaces, apply_test_perturbations


class TestDataConversion(unittest.TestCase):
    """Test the CSV-to-interfaces conversion logic."""
    
    def test_simple_bidirectional_interface(self):
        """Test converting a simple bidirectional link between two routers."""
        
        # Mock CSV row with simple bidirectional interface data
        csv_row = {
            'timestamp': '2024/01/01 12:00 UTC',
            'low_RouterA_egress_to_RouterB': "{'ground_truth': 100.0, 'perturbed': 100.0, 'corrected': None, 'confidence': None}",
            'low_RouterA_ingress_from_RouterB': "{'ground_truth': 95.0, 'perturbed': 95.0, 'corrected': None, 'confidence': None}",
            'low_RouterB_egress_to_RouterA': "{'ground_truth': 95.0, 'perturbed': 95.0, 'corrected': None, 'confidence': None}",
            'low_RouterB_ingress_from_RouterA': "{'ground_truth': 100.0, 'perturbed': 100.0, 'corrected': None, 'confidence': None}",
            # Non-telemetry columns should be ignored
            'some_other_column': 'ignored_value'
        }
        
        interfaces = convert_csv_row_to_interfaces(csv_row)
        
        # Should create exactly 2 interfaces (one for each router)
        self.assertEqual(len(interfaces), 2)
        
        # Check RouterA -> RouterB interface
        self.assertIn('RouterA_to_RouterB', interfaces)
        routerA_if = interfaces['RouterA_to_RouterB']
        self.assertEqual(routerA_if['tx_rate'], 100.0)      # From egress_to
        self.assertEqual(routerA_if['rx_rate'], 95.0)       # From ingress_from 
        self.assertEqual(routerA_if['connected_to'], 'RouterB_to_RouterA')
        self.assertEqual(routerA_if['local_router'], 'RouterA')
        self.assertEqual(routerA_if['remote_router'], 'RouterB')
        
        # Check RouterB -> RouterA interface
        self.assertIn('RouterB_to_RouterA', interfaces)
        routerB_if = interfaces['RouterB_to_RouterA']
        self.assertEqual(routerB_if['tx_rate'], 95.0)       # From egress_to
        self.assertEqual(routerB_if['rx_rate'], 100.0)      # From ingress_from (RouterA sent 100)
        self.assertEqual(routerB_if['connected_to'], 'RouterA_to_RouterB')
        self.assertEqual(routerB_if['local_router'], 'RouterB')
        self.assertEqual(routerB_if['remote_router'], 'RouterA')
    
    def test_perturbed_vs_ground_truth(self):
        """Test that perturbed values are used when available, ground truth as fallback."""
        
        csv_row = {
            # Perturbed data available
            'low_RouterA_egress_to_RouterB': "{'ground_truth': 100.0, 'perturbed': 120.0, 'corrected': None, 'confidence': None}",
            # No perturbed data - should use ground truth
            'low_RouterA_ingress_from_RouterB': "{'ground_truth': 95.0, 'perturbed': None, 'corrected': None, 'confidence': None}",
        }
        
        interfaces = convert_csv_row_to_interfaces(csv_row)
        
        self.assertEqual(len(interfaces), 1)
        routerA_if = interfaces['RouterA_to_RouterB']
        
        # Should use perturbed value when available
        self.assertEqual(routerA_if['tx_rate'], 120.0)  # Used perturbed value
        
        # Should use ground truth when perturbed is None
        self.assertEqual(routerA_if['rx_rate'], 95.0)   # Used ground truth (perturbed was None)
        
        # Should store ground truth separately
        self.assertEqual(routerA_if['_ground_truth_tx'], 100.0)
        self.assertEqual(routerA_if['_ground_truth_rx'], 95.0)
    
    def test_zero_rates_included(self):
        """Test that interfaces with zero rates are now included (after user's change)."""
        
        csv_row = {
            'low_RouterA_egress_to_RouterB': "{'ground_truth': 0.0, 'perturbed': 0.0, 'corrected': None, 'confidence': None}",
            'low_RouterA_ingress_from_RouterB': "{'ground_truth': 0.0, 'perturbed': 0.0, 'corrected': None, 'confidence': None}",
        }
        
        interfaces = convert_csv_row_to_interfaces(csv_row)
        
        # Should still create the interface even with zero rates
        self.assertEqual(len(interfaces), 1)
        routerA_if = interfaces['RouterA_to_RouterB']
        self.assertEqual(routerA_if['tx_rate'], 0.0)
        self.assertEqual(routerA_if['rx_rate'], 0.0)
    
    def test_malformed_data_skipped(self):
        """Test that malformed telemetry data is skipped gracefully."""
        
        csv_row = {
            # Valid data
            'low_RouterA_egress_to_RouterB': "{'ground_truth': 100.0, 'perturbed': 100.0, 'corrected': None, 'confidence': None}",
            # Malformed data - should be skipped
            'low_RouterB_egress_to_RouterC': "invalid_json_data",
            'low_RouterD_egress_to_RouterE': "{'missing_ground_truth': 50.0}",
        }
        
        interfaces = convert_csv_row_to_interfaces(csv_row)
        
        # Should only get the valid interface
        self.assertEqual(len(interfaces), 1)
        self.assertIn('RouterA_to_RouterB', interfaces)
    
    def test_complex_router_names(self):
        """Test that complex router names with numbers and special chars work."""
        
        csv_row = {
            'low_ATLAM5_egress_to_ATLAng': "{'ground_truth': 12.7, 'perturbed': 12.7, 'corrected': None, 'confidence': None}",
            'low_ATLAng_ingress_from_ATLAM5': "{'ground_truth': 12.5, 'perturbed': 12.5, 'corrected': None, 'confidence': None}",
        }
        
        interfaces = convert_csv_row_to_interfaces(csv_row)
        
        self.assertEqual(len(interfaces), 2)
        self.assertIn('ATLAM5_to_ATLAng', interfaces)
        self.assertIn('ATLAng_to_ATLAM5', interfaces)
        
        # Check the values are parsed correctly
        atlam5_if = interfaces['ATLAM5_to_ATLAng']
        self.assertEqual(atlam5_if['tx_rate'], 12.7)
        self.assertEqual(atlam5_if['local_router'], 'ATLAM5')
        self.assertEqual(atlam5_if['remote_router'], 'ATLAng')


class TestPerturbationLogic(unittest.TestCase):
    """Test the perturbation and ground truth extraction logic."""
    
    def test_ground_truth_extraction(self):
        """Test that ground truth is extracted correctly from interface metadata."""
        
        interfaces = {
            'RouterA_to_RouterB': {
                'interface_status': 'up',
                'rx_rate': 95.0,
                'tx_rate': 100.0,
                'capacity': 1000.0,
                'connected_to': 'RouterB_to_RouterA',
                'local_router': 'RouterA',
                'remote_router': 'RouterB',
                '_ground_truth_rx': 90.0,  # Different from current values
                '_ground_truth_tx': 105.0
            }
        }
        
        perturbed, ground_truth = apply_test_perturbations(interfaces)
        
        # Ground truth should use the _ground_truth values
        self.assertEqual(ground_truth['RouterA_to_RouterB']['rx_rate'], 90.0)
        self.assertEqual(ground_truth['RouterA_to_RouterB']['tx_rate'], 105.0)
        
        # Perturbed should clean up the metadata
        self.assertNotIn('_ground_truth_rx', perturbed['RouterA_to_RouterB'])
        self.assertNotIn('_ground_truth_tx', perturbed['RouterA_to_RouterB'])
    
    def test_fallback_to_current_values(self):
        """Test fallback when no ground truth metadata is available."""
        
        interfaces = {
            'RouterA_to_RouterB': {
                'interface_status': 'up',
                'rx_rate': 95.0,
                'tx_rate': 100.0,
                'capacity': 1000.0,
                # No _ground_truth_* fields
            }
        }
        
        perturbed, ground_truth = apply_test_perturbations(interfaces)
        
        # Should fall back to current values
        self.assertEqual(ground_truth['RouterA_to_RouterB']['rx_rate'], 95.0)
        self.assertEqual(ground_truth['RouterA_to_RouterB']['tx_rate'], 100.0)


def run_simple_conversion_examples():
    """Run some simple examples to show the conversion in action."""
    
    print("=== Simple Conversion Examples ===")
    
    # Example 1: Perfect network
    print("\n1. Perfect Network (rates match):")
    csv_row = {
        'low_R1_egress_to_R2': "{'ground_truth': 50.0, 'perturbed': 50.0, 'corrected': None, 'confidence': None}",
        'low_R1_ingress_from_R2': "{'ground_truth': 30.0, 'perturbed': 30.0, 'corrected': None, 'confidence': None}",
        'low_R2_egress_to_R1': "{'ground_truth': 30.0, 'perturbed': 30.0, 'corrected': None, 'confidence': None}",
        'low_R2_ingress_from_R1': "{'ground_truth': 50.0, 'perturbed': 50.0, 'corrected': None, 'confidence': None}",
    }
    
    interfaces = convert_csv_row_to_interfaces(csv_row)
    for if_id, if_data in interfaces.items():
        print(f"  {if_id}: TX={if_data['tx_rate']}, RX={if_data['rx_rate']}")
    
    # Example 2: Perturbed network
    print("\n2. Perturbed Network (some corruption):")
    csv_row = {
        'low_R1_egress_to_R2': "{'ground_truth': 50.0, 'perturbed': 60.0, 'corrected': None, 'confidence': None}",  # Corrupted!
        'low_R1_ingress_from_R2': "{'ground_truth': 30.0, 'perturbed': 30.0, 'corrected': None, 'confidence': None}",
        'low_R2_egress_to_R1': "{'ground_truth': 30.0, 'perturbed': 25.0, 'corrected': None, 'confidence': None}",  # Corrupted!
        'low_R2_ingress_from_R1': "{'ground_truth': 50.0, 'perturbed': 50.0, 'corrected': None, 'confidence': None}",
    }
    
    interfaces = convert_csv_row_to_interfaces(csv_row)
    for if_id, if_data in interfaces.items():
        gt_tx = if_data.get('_ground_truth_tx', 'N/A')
        gt_rx = if_data.get('_ground_truth_rx', 'N/A')
        print(f"  {if_id}: TX={if_data['tx_rate']} (GT: {gt_tx}), RX={if_data['rx_rate']} (GT: {gt_rx})")


if __name__ == '__main__':
    # Run the simple examples first
    run_simple_conversion_examples()
    
    print("\n" + "="*50)
    
    # Run the unit tests
    unittest.main(verbosity=2) 