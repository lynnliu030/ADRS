"""
Tests for the DTP (Democratic Trust Propagation) repair algorithm.
"""
import unittest
import sys
import os
import pandas as pd
from unittest.mock import patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dtp import DemocraticTrustPropagationRepair, RepairConfig
from common_utils import NetworkTopology


class TestNetworkTopology(unittest.TestCase):
    def setUp(self):
        # Create a simple test topology with 3 nodes and 2 links
        self.topology_dict = {
            "nodes": [
                {"id": 0, "name": "NodeA"},
                {"id": 1, "name": "NodeB"},
                {"id": 2, "name": "NodeC"}
            ],
            "links": [
                {"source": 0, "target": 1},  # NodeA <-> NodeB
                {"source": 1, "target": 2}   # NodeB <-> NodeC
            ]
        }
        self.topology = NetworkTopology(self.topology_dict)

    def test_initialization(self):
        """Test that NetworkTopology initializes correctly."""
        # Check nodes dictionary
        self.assertEqual(len(self.topology.nodes), 3)
        self.assertEqual(self.topology.nodes[0]['name'], "NodeA")
        self.assertEqual(self.topology.nodes[1]['name'], "NodeB")
        self.assertEqual(self.topology.nodes[2]['name'], "NodeC")
        
        # Check name to ID mapping
        self.assertEqual(self.topology.node_name_to_id["NodeA"], 0)
        self.assertEqual(self.topology.node_name_to_id["NodeB"], 1)
        self.assertEqual(self.topology.node_name_to_id["NodeC"], 2)
        
        # Check adjacency lists (including external interfaces)
        self.assertEqual(set(self.topology.adjacency[0]), {1, -1})  # NodeA connects to NodeB and external
        self.assertEqual(set(self.topology.adjacency[1]), {0, 2, -1})  # NodeB connects to NodeA, NodeC, and external
        self.assertEqual(set(self.topology.adjacency[2]), {1, -1})  # NodeC connects to NodeB and external
        
        # Check interface naming
        self.assertEqual(self.topology.interfaces[(0, 1)], "NodeA_egress_to_NodeB")
        self.assertEqual(self.topology.interfaces[(1, 0)], "NodeB_egress_to_NodeA")
        self.assertEqual(self.topology.interfaces[(1, 2)], "NodeB_egress_to_NodeC")
        self.assertEqual(self.topology.interfaces[(2, 1)], "NodeC_egress_to_NodeB")
        
        # Check external interfaces
        self.assertEqual(self.topology.interfaces[(0, -1)], "NodeA_external")
        self.assertEqual(self.topology.interfaces[(1, -1)], "NodeB_external")
        self.assertEqual(self.topology.interfaces[(2, -1)], "NodeC_external")


class TestDTPRepair(unittest.TestCase):
    def setUp(self):
        # Create a simple 2-node topology for easier testing
        self.topology_dict = {
            "nodes": [
                {"id": 0, "name": "NodeA"},
                {"id": 1, "name": "NodeB"}
            ],
            "links": [
                {"source": 0, "target": 1}
            ]
        }
        
        # Create a consistent test snapshot (values should match for a perfect network)
        self.perfect_snapshot = {
            "low_NodeA_egress_to_NodeB": 100.0,
            "low_NodeB_ingress_from_NodeA": 100.0,
            "low_NodeB_egress_to_NodeA": 200.0,
            "low_NodeA_ingress_from_NodeB": 200.0,
            "low_NodeA_termination": 50.0,
            "low_NodeB_termination": 75.0,
            "low_NodeA_origination": 25.0,
            "low_NodeB_origination": 30.0
        }
        
        # Create an inconsistent snapshot (some values don't match)
        self.inconsistent_snapshot = {
            "low_NodeA_egress_to_NodeB": 100.0,
            "low_NodeB_ingress_from_NodeA": 95.0,  # Should be 100.0
            "low_NodeB_egress_to_NodeA": 200.0,
            "low_NodeA_ingress_from_NodeB": 210.0,  # Should be 200.0
            "low_NodeA_termination": 50.0,
            "low_NodeB_termination": 75.0,
            "low_NodeA_origination": 25.0,
            "low_NodeB_origination": 30.0
        }
        
        self.config = RepairConfig(seed=42, num_trials=10, similarity_threshold=0.05)
        self.repair = DemocraticTrustPropagationRepair(self.topology_dict, config=self.config)

    def test_initialization(self):
        """Test that DTP initializes correctly."""
        self.assertEqual(len(self.repair.network_topology.nodes), 2)
        self.assertEqual(self.repair.config.seed, 42)
        self.assertEqual(self.repair.config.num_trials, 10)
        
        # Check that topology data is correctly set up
        self.assertEqual(len(self.repair.network_topology.adjacency[0]), 2)  # One neighbor + external
        self.assertEqual(len(self.repair.network_topology.adjacency[1]), 2)  # One neighbor + external

    def test_perfect_network_repair(self):
        """Test repair on a perfect network (should return original values)."""
        result = self.repair.repair_snapshot(self.perfect_snapshot)
        
        # Check that all required keys exist
        self.assertIn('corrected_values', result)
        self.assertIn('confidences', result) 
        self.assertIn('original_values', result)
        
        # Check interface values are present
        self.assertIn("NodeA_egress_to_NodeB", result['corrected_values'])
        self.assertIn("NodeB_egress_to_NodeA", result['corrected_values'])
        self.assertIn("NodeA_external", result['corrected_values'])
        self.assertIn("NodeB_external", result['corrected_values'])

    def test_inconsistent_network_repair(self):
        """Test repair on an inconsistent network."""
        result = self.repair.repair_snapshot(self.inconsistent_snapshot)
        
        # Check that corrected values are different from original for inconsistent links
        nodeA_to_nodeB_original = result['original_values']["NodeA_egress_to_NodeB"]
        nodeA_to_nodeB_corrected = result['corrected_values']["NodeA_egress_to_NodeB"]
        
        # At least one of the directions should be corrected
        # Original: (210.0, 100.0), but ingress should be corrected to ~100
        self.assertEqual(nodeA_to_nodeB_original, (210.0, 100.0))  # (in, out)
        
        # The algorithm should fix the inconsistency
        corrected_in, corrected_out = nodeA_to_nodeB_corrected
        
        # The algorithm should produce more consistent values
        # The corrected values should be closer to each other than the original inconsistent values
        original_diff = abs(210.0 - 100.0)  # 110.0 difference originally
        corrected_diff = abs(corrected_in - corrected_out)  
        self.assertLess(corrected_diff, original_diff)  # Should be more consistent now

    def test_confidence_values(self):
        """Test that confidence values are reasonable."""
        result = self.repair.repair_snapshot(self.perfect_snapshot)
        
        # All confidence values should be between 0 and 1
        for interface_name in result['confidences']:
            in_conf, out_conf = result['confidences'][interface_name]
            self.assertGreaterEqual(in_conf, 0.0)
            self.assertLessEqual(in_conf, 1.0)
            self.assertGreaterEqual(out_conf, 0.0)
            self.assertLessEqual(out_conf, 1.0)

    def test_external_interfaces(self):
        """Test that external interfaces are handled correctly."""
        result = self.repair.repair_snapshot(self.perfect_snapshot)
        
        # Check NodeA external interface
        nodeA_external_original = result['original_values']["NodeA_external"]
        nodeA_external_corrected = result['corrected_values']["NodeA_external"]
        
        # For external interfaces: (in, out) = (origination, termination)
        self.assertEqual(nodeA_external_original, (25.0, 50.0))
        # Should be the same since this is a perfect network
        self.assertEqual(nodeA_external_corrected, (25.0, 50.0))

    def test_with_paths_predictions(self):
        """Test repair with paths-based predictions."""
        # Create simple paths data
        paths = {
            "NodeA": {
                "NodeB": [
                    {"path": ["NodeA", "NodeB"], "weight": 1.0}
                ]
            },
            "NodeB": {
                "NodeA": [
                    {"path": ["NodeB", "NodeA"], "weight": 1.0}
                ]
            }
        }
        
        # Add demand data to snapshot
        snapshot_with_demands = self.perfect_snapshot.copy()
        snapshot_with_demands.update({
            "high_NodeA_NodeB": 80.0,
            "high_NodeB_NodeA": 150.0
        })
        
        repair_with_paths = DemocraticTrustPropagationRepair(
            self.topology_dict, 
            config=self.config
        )
        
        result = repair_with_paths.repair_snapshot(snapshot_with_demands, paths=paths)
        
        # Should complete without error
        self.assertIn('corrected_values', result)

    def test_config_parameters(self):
        """Test that config parameters are used correctly."""
        custom_config = RepairConfig(
            num_trials=5,
            similarity_threshold=0.1,
            seed=123
        )
        
        repair_custom = DemocraticTrustPropagationRepair(
            self.topology_dict,
            config=custom_config
        )
        
        self.assertEqual(repair_custom.config.num_trials, 5)
        self.assertEqual(repair_custom.config.similarity_threshold, 0.1)
        self.assertEqual(repair_custom.config.seed, 123)


class TestLinkValue(unittest.TestCase):
    def setUp(self):
        from dtp import LinkValue
        self.link = LinkValue(
            link_id=(0, 1),
            in_value=100.0,
            out_value=200.0,
            demand_based_prediction=150.0,
            name="test_interface"
        )

    def test_initialization(self):
        """Test LinkValue initialization."""
        self.assertEqual(self.link.link_id, (0, 1))
        self.assertEqual(self.link.measured_in_value, 100.0)
        self.assertEqual(self.link.measured_out_value, 200.0)
        self.assertEqual(self.link.demand_based_predicton, 150.0)
        self.assertIsNone(self.link.locked_value)
        self.assertIsNone(self.link.confidence)

    def test_is_locked_property(self):
        """Test the is_locked property."""
        self.assertFalse(self.link.is_locked)
        
        self.link.lock_val(175.0, 0.8)
        self.assertTrue(self.link.is_locked)

    def test_possible_values(self):
        """Test possible_values property."""
        expected = [100.0, 200.0, 150.0]
        self.assertEqual(self.link.possible_values, expected)

    def test_lock_val(self):
        """Test locking a value."""
        self.link.lock_val(175.0, 0.8)
        self.assertEqual(self.link.locked_value, 175.0)
        self.assertEqual(self.link.confidence, 0.8)
        self.assertTrue(self.link.is_locked)


class TestDTPDataFrame(unittest.TestCase):
    def setUp(self):
        """Set up test data for DataFrame repair tests."""
        self.topology_dict = {
            'nodes': [{'id': 0, 'name': 'NodeA'}, {'id': 1, 'name': 'NodeB'}],
            'links': [{'source': 0, 'target': 1}]
        }
        
        self.paths = {
            'NodeA': {
                'NodeB': [{'path': ['NodeA', 'NodeB'], 'weight': 1.0}]
            },
            'NodeB': {
                'NodeA': [{'path': ['NodeB', 'NodeA'], 'weight': 1.0}]
            }
        }
        
        # Create test DataFrame with multiple rows with perturbed data
        self.test_data = []
        for i in range(3):
            self.test_data.append({
                'timestamp': f'2024/01/01 {12+i}:00 UTC',
                'telemetry_perturbed_type': 'scaled',
                'input_perturbed_type': 'none', 
                'true_detect_inconsistent': True,  # These need repair
                'high_NodeA_NodeB': 100.0 + i*10,
                'high_NodeB_NodeA': 200.0 + i*20,
                # Inconsistent counter values that need repair
                'low_NodeA_egress_to_NodeB': {'perturbed': 90.0 + i*10, 'ground_truth': 100.0 + i*10},
                'low_NodeB_ingress_from_NodeA': {'perturbed': 110.0 + i*10, 'ground_truth': 100.0 + i*10},
                'low_NodeB_egress_to_NodeA': {'perturbed': 180.0 + i*20, 'ground_truth': 200.0 + i*20},
                'low_NodeA_ingress_from_NodeB': {'perturbed': 220.0 + i*20, 'ground_truth': 200.0 + i*20},
                'low_NodeA_origination': {'perturbed': 95.0 + i*10, 'ground_truth': 100.0 + i*10},
                'low_NodeA_termination': {'perturbed': 210.0 + i*20, 'ground_truth': 200.0 + i*20},
                'low_NodeB_origination': {'perturbed': 190.0 + i*20, 'ground_truth': 200.0 + i*20},
                'low_NodeB_termination': {'perturbed': 105.0 + i*10, 'ground_truth': 100.0 + i*10}
            })
        
        self.test_df = pd.DataFrame(self.test_data)
        self.config = RepairConfig(disable_cache=True, num_trials=10, seed=42)
        self.repairer = DemocraticTrustPropagationRepair(self.topology_dict, self.config)

    @patch('dtp.load_paths_for_timestamp')
    def test_repair_df_with_paths(self, mock_load):
        """Test DataFrame repair with paths."""
        # Mock the paths loading to return our test paths
        mock_load.return_value = self.paths
        
        result_df = self.repairer.repair_df(self.test_df, "/mock/paths/dir")
        
        # Check that all rows were processed
        self.assertEqual(len(result_df), 3)
        
        # Check that load_paths_for_timestamp was called for each row
        self.assertEqual(mock_load.call_count, 3)
        
        # Check that all rows have repair information
        for idx, row in result_df.iterrows():
            self.assertEqual(row['repair_type'], 'DTP')
            self.assertIsNotNone(row['repair_confidence'])
            # Check that corrected values were added
            self.assertIn('corrected', row['low_NodeA_egress_to_NodeB'])
            self.assertIn('confidence', row['low_NodeA_egress_to_NodeB'])
        
        # Check DataFrame attributes
        self.assertEqual(result_df.attrs['Repair strategy'], "DTP w/ paths")
        self.assertEqual(result_df.attrs['config'], self.config)

    def test_repair_df_without_paths(self):
        """Test DataFrame repair without paths."""
        result_df = self.repairer.repair_df(self.test_df, None)
        
        # Check that all rows were processed
        self.assertEqual(len(result_df), 3)
        
        # Check that all rows have repair information
        for idx, row in result_df.iterrows():
            self.assertEqual(row['repair_type'], 'DTP')
            self.assertIsNotNone(row['repair_confidence'])
            # Check that corrected values were added
            self.assertIn('corrected', row['low_NodeA_egress_to_NodeB'])
            self.assertIn('confidence', row['low_NodeA_egress_to_NodeB'])
        
        # Check DataFrame attributes
        self.assertEqual(result_df.attrs['Repair strategy'], "DTP w/o paths")

    @patch('dtp.load_paths_for_timestamp')
    @patch('builtins.print')
    def test_repair_df_error_handling(self, mock_print, mock_load):
        """Test DataFrame repair error handling."""
        # Mock to raise an exception for the second row
        def side_effect(paths_path, timestamp):
            if "13:00" in timestamp:
                raise ValueError("Paths file not found")
            return self.paths
        
        mock_load.side_effect = side_effect
        
        result_df = self.repairer.repair_df(self.test_df, "/mock/paths/dir")
        
        # Should have 2 rows (one skipped due to error)
        self.assertEqual(len(result_df), 2)
        
        # Check that warning was printed
        mock_print.assert_called()
        warning_calls = [call for call in mock_print.call_args_list if 'Warning:' in str(call)]
        self.assertTrue(len(warning_calls) > 0)

    def test_repair_df_consistency(self):
        """Test that repair_df produces consistent results."""
        # Create repairer and run twice on same data
        result_df1 = self.repairer.repair_df(self.test_df, None)
        result_df2 = self.repairer.repair_df(self.test_df, None)
        
        # Results should have same structure
        self.assertEqual(len(result_df1), len(result_df2))
        self.assertEqual(result_df1.attrs['Repair strategy'], result_df2.attrs['Repair strategy'])
        
        # Check that repair was performed on both runs
        for i in range(len(result_df1)):
            row1 = result_df1.iloc[i]
            row2 = result_df2.iloc[i]
            self.assertEqual(row1['repair_type'], row2['repair_type'])
            # Both should have confidence values
            self.assertIsNotNone(row1['repair_confidence'])
            self.assertIsNotNone(row2['repair_confidence'])

    def test_process_row_individual(self):
        """Test the process_row method individually."""
        row = self.test_df.iloc[0]
        
        # Test with paths
        repaired_row = self.repairer.process_row(row, self.paths)
        
        # Check that repair was applied
        self.assertEqual(repaired_row['repair_type'], 'DTP')
        self.assertIsNotNone(repaired_row['repair_confidence'])
        self.assertIn('corrected', repaired_row['low_NodeA_egress_to_NodeB'])
        
        # Test without paths
        repaired_row_no_paths = self.repairer.process_row(row, None)
        
        # Check that repair was applied
        self.assertEqual(repaired_row_no_paths['repair_type'], 'DTP')
        self.assertIsNotNone(repaired_row_no_paths['repair_confidence'])

    def test_repair_df_config_usage(self):
        """Test that repair_df uses the configuration correctly."""
        # Create repairer with specific config
        custom_config = RepairConfig(
            disable_cache=True,
            num_trials=5,
            similarity_threshold=0.1,
            seed=123
        )
        custom_repairer = DemocraticTrustPropagationRepair(self.topology_dict, custom_config)
        
        result_df = custom_repairer.repair_df(self.test_df, None)
        
        # Check that config is preserved in attributes
        self.assertEqual(result_df.attrs['config'], custom_config)
        
        # Check that repair was performed
        self.assertEqual(len(result_df), 3)
        for idx, row in result_df.iterrows():
            self.assertEqual(row['repair_type'], 'DTP')


if __name__ == '__main__':
    # Run specific test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkTopology))
    suite.addTests(loader.loadTestsFromTestCase(TestDTPRepair))
    suite.addTests(loader.loadTestsFromTestCase(TestLinkValue))
    suite.addTests(loader.loadTestsFromTestCase(TestDTPDataFrame))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite) 