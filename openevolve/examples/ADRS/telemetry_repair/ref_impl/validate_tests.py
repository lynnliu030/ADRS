"""
Tests for the validate.py validation logic.
"""
import unittest
import sys
import os
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we want to test  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from validate import (
    Validator,
    ValidatorConfig
)
from common_utils import load_paths_for_timestamp
from common_utils import NetworkTopology, perc_equal, compute_predicted_counters


class TestCreateSnapshot(unittest.TestCase):
    def setUp(self):
        # Create a test DataFrame row with various data types
        self.test_row = pd.Series({
            'timestamp': '2024/01/01 12:00 UTC',
            'telemetry_perturbed_type': 'scaled',
            'input_perturbed_type': 'none',
            'true_detect_inconsistent': False,
            'low_NodeA_egress_to_NodeB': {
                'corrected': 100.0,
                'confidence': 0.8,
                'perturbed': 95.0,
                'ground_truth': 98.0
            },
            'low_NodeB_ingress_from_NodeA': {
                'corrected': None,
                'confidence': 0.0,
                'perturbed': 105.0,
                'ground_truth': 98.0
            },
            'low_NodeA_termination': {
                'corrected': 50.0,
                'confidence': 0.9,
                'perturbed': 45.0,
                'ground_truth': 48.0
            },
            'high_NodeA_NodeB': 75.5  # High-level demand
        })
        
        # Create a validator instance for testing the create_snapshot functionality
        topology_dict = {'nodes': [{'id': 0, 'name': 'NodeA'}, {'id': 1, 'name': 'NodeB'}], 'links': []}
        self.validator = Validator(topology_dict, ValidatorConfig(disable_cache=True))

    def test_prefers_corrected_values(self):
        """Test that corrected values are preferred over perturbed values."""
        snapshot, confidences, corrected_count, perturbed_count = self.validator._create_corrected_snapshot(self.test_row)
        
        # Should use corrected value when available
        self.assertEqual(snapshot['low_NodeA_egress_to_NodeB'], 100.0)
        self.assertEqual(confidences['low_NodeA_egress_to_NodeB'], 0.8)
        
        # Should use perturbed when corrected is None
        self.assertEqual(snapshot['low_NodeB_ingress_from_NodeA'], 105.0)
        self.assertEqual(confidences['low_NodeB_ingress_from_NodeA'], 0.0)
        
        # Check counts
        self.assertEqual(corrected_count, 2)  # NodeA_egress and NodeA_termination
        self.assertEqual(perturbed_count, 1)  # NodeB_ingress

    def test_handles_high_level_demands(self):
        """Test that high-level demands are included correctly."""
        snapshot, confidences, _, _ = self.validator._create_corrected_snapshot(self.test_row)
        
        self.assertEqual(snapshot['high_NodeA_NodeB'], 75.5)
        self.assertIn('high_NodeA_NodeB', snapshot)

    def test_metadata_preserved(self):
        """Test that metadata columns are preserved."""
        snapshot, _, _, _ = self.validator._create_corrected_snapshot(self.test_row)
        
        self.assertEqual(snapshot['timestamp'], '2024/01/01 12:00 UTC')
        self.assertEqual(snapshot['telemetry_perturbed_type'], 'scaled')
        self.assertEqual(snapshot['input_perturbed_type'], 'none')
        self.assertEqual(snapshot['true_detect_inconsistent'], False)


class TestComputePredictedCounters(unittest.TestCase):
    def setUp(self):
        # Create simple 2-node topology
        self.topology_dict = {
            "nodes": [
                {"id": 0, "name": "NodeA"},
                {"id": 1, "name": "NodeB"}
            ],
            "links": [
                {"source": 0, "target": 1}
            ]
        }
        self.topology = NetworkTopology(self.topology_dict)
        
        # Create simple paths
        self.paths = {
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
        
        # Create test snapshot
        self.snapshot = {
            'high_NodeA_NodeB': 100.0,
            'high_NodeB_NodeA': 200.0
        }

    def test_basic_prediction_computation(self):
        """Test basic prediction computation without perturbation."""
        predictions = compute_predicted_counters(
            self.snapshot, self.paths, self.topology, paths_perturbed_num_nodes=0
        )
        
        # Check external interface predictions
        self.assertEqual(predictions[(-1, 0)], 100.0)  # NodeA ingress (origination)
        self.assertEqual(predictions[(1, -1)], 100.0)  # NodeB egress (termination)
        self.assertEqual(predictions[(-1, 1)], 200.0)  # NodeB ingress (origination)
        self.assertEqual(predictions[(0, -1)], 200.0)  # NodeA egress (termination)
        
        # Check internal link predictions
        self.assertEqual(predictions[(0, 1)], 100.0)  # NodeA -> NodeB
        self.assertEqual(predictions[(1, 0)], 200.0)  # NodeB -> NodeA

    def test_with_perturbation(self):
        """Test prediction computation with node perturbation."""
        # This should reduce some predictions to 0 due to path skipping
        predictions = compute_predicted_counters(
            self.snapshot, self.paths, self.topology, paths_perturbed_num_nodes=1
        )
        
        # External interfaces should still get traffic (they're not skipped)
        self.assertGreaterEqual(predictions[(-1, 0)], 0.0)
        self.assertGreaterEqual(predictions[(1, -1)], 0.0)
        
        # Some internal links might be reduced due to perturbation
        # (depends on which node was randomly selected)
        self.assertGreaterEqual(predictions[(0, 1)], 0.0)
        self.assertGreaterEqual(predictions[(1, 0)], 0.0)

    def test_missing_demand_raises_error(self):
        """Test that missing demand keys raise appropriate errors."""
        incomplete_snapshot = {'high_NodeA_NodeB': 100.0}  # Missing NodeB_NodeA
        
        with self.assertRaises(AssertionError):
            compute_predicted_counters(
                incomplete_snapshot, self.paths, self.topology
            )


class TestValidateSnapshotWithPaths(unittest.TestCase):
    def setUp(self):
        # Create test topology
        self.topology_dict = {
            "nodes": [
                {"id": 0, "name": "NodeA"},
                {"id": 1, "name": "NodeB"}
            ],
            "links": [
                {"source": 0, "target": 1}
            ]
        }
        
        # Create validator with default config
        self.validator = Validator(self.topology_dict, ValidatorConfig())
        
        # Create test paths
        self.paths = {
            "NodeA": {
                "NodeB": [{"path": ["NodeA", "NodeB"], "weight": 1.0}]
            },
            "NodeB": {
                "NodeA": [{"path": ["NodeB", "NodeA"], "weight": 1.0}]
            }
        }
        
        # Create perfect test row (predictions should match counters exactly)
        self.perfect_row = pd.Series({
            'timestamp': '2024/01/01 12:00 UTC',
            'telemetry_perturbed_type': 'none',
            'input_perturbed_type': 'none', 
            'true_detect_inconsistent': False,
            'high_NodeA_NodeB': 100.0,
            'high_NodeB_NodeA': 200.0,
            # Counter values that match predictions exactly
            'low_NodeA_egress_to_NodeB': {'corrected': 100.0, 'confidence': 1.0},
            'low_NodeB_ingress_from_NodeA': {'corrected': 100.0, 'confidence': 1.0},
            'low_NodeB_egress_to_NodeA': {'corrected': 200.0, 'confidence': 1.0},
            'low_NodeA_ingress_from_NodeB': {'corrected': 200.0, 'confidence': 1.0},
            'low_NodeA_origination': {'corrected': 100.0, 'confidence': 1.0},
            'low_NodeA_termination': {'corrected': 200.0, 'confidence': 1.0},
            'low_NodeB_origination': {'corrected': 200.0, 'confidence': 1.0},
            'low_NodeB_termination': {'corrected': 100.0, 'confidence': 1.0}
        })

    def test_perfect_validation(self):
        """Test validation with perfect counter values."""
        result_row = self.validator.validate_row(
            self.perfect_row, self.paths
        )
        
        # Should have 100% satisfaction rate
        self.assertEqual(result_row['validation_confidence'], 1.0)
        self.assertEqual(result_row['validation_type'], "w/ paths")
        self.assertIsNone(result_row['validation_result'])

    def test_confidence_cutoff_filtering(self):
        """Test that low-confidence counters are filtered out."""
        # Create validator with high confidence cutoff
        high_cutoff_validator = Validator(self.topology_dict, ValidatorConfig(confidence_cutoff=0.5))
        
        # Create row with some low-confidence values
        low_conf_row = self.perfect_row.copy()
        low_conf_row['low_NodeA_egress_to_NodeB'] = {'corrected': 100.0, 'confidence': 0.1}
        
        result_row = high_cutoff_validator.validate_row(
            low_conf_row, self.paths
        )
        
        # Should still work but with fewer checks (low confidence counter excluded)
        # Satisfaction rate might be different due to fewer total checks
        self.assertIsNotNone(result_row['validation_confidence'])
        self.assertGreaterEqual(result_row['validation_confidence'], 0.0)
        self.assertLessEqual(result_row['validation_confidence'], 1.0)


class TestValidateSnapshot(unittest.TestCase):
    def setUp(self):
        # Create simple 2-node topology for demand invariant testing
        self.topology_dict = {
            "nodes": [
                {"id": 0, "name": "NodeA"},
                {"id": 1, "name": "NodeB"}
            ],
            "links": [
                {"source": 0, "target": 1}
            ]
        }
        
        # Create validator with default config
        self.validator = Validator(self.topology_dict, ValidatorConfig())
        
        # Create row that should satisfy demand invariants
        self.good_row = pd.Series({
            'timestamp': '2024/01/01 12:00 UTC',
            'telemetry_perturbed_type': 'none',
            'input_perturbed_type': 'none',
            'true_detect_inconsistent': False,
            # Demands and external counters that match
            'high_NodeA_NodeB': 100.0,
            'high_NodeB_NodeA': 200.0,
            'low_NodeA_origination': {'corrected': 100.0, 'confidence': 1.0},
            'low_NodeA_termination': {'corrected': 200.0, 'confidence': 1.0},
            'low_NodeB_origination': {'corrected': 200.0, 'confidence': 1.0},
            'low_NodeB_termination': {'corrected': 100.0, 'confidence': 1.0}
        })

    def test_demand_invariant_validation(self):
        """Test demand invariant validation logic."""
        result_row = self.validator.validate_row(self.good_row, paths=None)
        
        # Check results
        self.assertEqual(result_row['validation_type'], "w/o paths")
        self.assertIsNone(result_row['validation_result'])
        # Note: Actual validation confidence will depend on the real DemandInvariantChecker
        self.assertIsNotNone(result_row['validation_confidence'])


class TestValidatorDataFrame(unittest.TestCase):
    def setUp(self):
        """Set up test data for DataFrame validation tests."""
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
        
        # Create test DataFrame with multiple rows
        self.test_data = []
        for i in range(3):
            self.test_data.append({
                'timestamp': f'2024/01/01 {12+i}:00 UTC',
                'telemetry_perturbed_type': 'none',
                'input_perturbed_type': 'none', 
                'true_detect_inconsistent': False,
                'high_NodeA_NodeB': 100.0 + i*10,
                'high_NodeB_NodeA': 200.0 + i*20,
                'low_NodeA_egress_to_NodeB': {'corrected': 100.0 + i*10, 'confidence': 1.0},
                'low_NodeB_ingress_from_NodeA': {'corrected': 100.0 + i*10, 'confidence': 1.0},
                'low_NodeB_egress_to_NodeA': {'corrected': 200.0 + i*20, 'confidence': 1.0},
                'low_NodeA_ingress_from_NodeB': {'corrected': 200.0 + i*20, 'confidence': 1.0},
                'low_NodeA_origination': {'corrected': 100.0 + i*10, 'confidence': 1.0},
                'low_NodeA_termination': {'corrected': 200.0 + i*20, 'confidence': 1.0},
                'low_NodeB_origination': {'corrected': 200.0 + i*20, 'confidence': 1.0},
                'low_NodeB_termination': {'corrected': 100.0 + i*10, 'confidence': 1.0}
            })
        
        self.test_df = pd.DataFrame(self.test_data)
        self.validator = Validator(self.topology_dict, ValidatorConfig(confidence_cutoff=0.0, disable_cache=True))

    def test_validate_df_with_paths_dict(self):
        """Test DataFrame validation with a paths dictionary."""
        result_df = self.validator.validate_df(self.test_df, paths_dict=self.paths)
        
        # Check that all rows were processed
        self.assertEqual(len(result_df), 3)
        
        # Check that all rows have path-based validation type
        for _, row in result_df.iterrows():
            self.assertEqual(row['validation_type'], "w/ paths")
            self.assertIsNotNone(row['validation_confidence'])
            self.assertIsNone(row['validation_result'])
            
        # With perfect data, should have high confidence
        for _, row in result_df.iterrows():
            self.assertGreaterEqual(row['validation_confidence'], 0.8)

    def test_validate_df_without_paths(self):
        """Test DataFrame validation without paths (demand invariants)."""
        result_df = self.validator.validate_df(self.test_df)
        
        # Check that all rows were processed
        self.assertEqual(len(result_df), 3)
        
        # Check that all rows have demand invariant validation type
        for _, row in result_df.iterrows():
            self.assertEqual(row['validation_type'], "w/o paths")
            self.assertIsNotNone(row['validation_confidence'])
            self.assertIsNone(row['validation_result'])

    def test_validate_df_with_paths_path_simple(self):
        """Test DataFrame validation with paths_path using simple approach."""
        # For this test, just ensure that the method doesn't crash when trying to load paths
        # Since we can't easily mock the multiprocessing calls, we'll test the behavior when paths can't be loaded
        
        result_df = self.validator.validate_df(self.test_df, paths_path="/nonexistent/paths/dir")
        
        # Should handle missing paths gracefully and return empty DataFrame
        self.assertEqual(len(result_df), 0)  # All rows should be skipped due to missing paths

    def test_validate_df_mixed_confidence(self):
        """Test DataFrame validation with mixed confidence values."""
        # Create DataFrame with some low-confidence values
        mixed_data = self.test_data.copy()
        mixed_data[1]['low_NodeA_egress_to_NodeB'] = {'corrected': 110.0, 'confidence': 0.1}
        mixed_df = pd.DataFrame(mixed_data)
        
        # Create validator with higher confidence cutoff
        high_cutoff_validator = Validator(
            self.topology_dict, 
            ValidatorConfig(confidence_cutoff=0.5, disable_cache=True)
        )
        
        result_df = high_cutoff_validator.validate_df(mixed_df, paths_dict=self.paths)
        
        # All rows should still be processed
        self.assertEqual(len(result_df), 3)
        
        # All should have validation results
        for _, row in result_df.iterrows():
            self.assertIsNotNone(row['validation_confidence'])

    def test_validate_df_error_handling_simple(self):
        """Test DataFrame validation error handling with simple approach."""
        # Test error handling by providing invalid paths_path
        result_df = self.validator.validate_df(self.test_df, paths_path="/nonexistent/paths/dir")
        
        # Should handle errors gracefully and return empty DataFrame
        self.assertEqual(len(result_df), 0)  # All rows should be skipped due to errors


class TestUtilityFunctions(unittest.TestCase):
    def test_perc_equal_basic(self):
        """Test perc_equal function with basic cases."""
        # Identical values
        self.assertTrue(perc_equal(100.0, 100.0, threshold=0.03))
        
        # Within default 3% threshold
        self.assertTrue(perc_equal(100.0, 102.0, threshold=0.03))  # 2% difference
        
        # Outside default threshold
        self.assertFalse(perc_equal(100.0, 105.0, threshold=0.03))  # 5% difference
        
        # Zero handling
        self.assertTrue(perc_equal(0.0, 0.0, threshold=0.03))

    @patch('common_utils.os.path.isfile')
    @patch('common_utils.open')
    @patch('common_utils.json.load')
    def test_load_paths_for_timestamp_file(self, mock_json_load, mock_open, mock_isfile):
        """Test loading paths from a single file."""
        mock_isfile.return_value = True
        mock_json_load.return_value = {"paths": {"test": "data"}}
        
        result = load_paths_for_timestamp("/path/to/paths.json", "2024/01/01 12:00 UTC")
        
        self.assertEqual(result, {"test": "data"})
        mock_open.assert_called_once_with("/path/to/paths.json")

    @patch('common_utils.os.path.isfile')
    @patch('common_utils.open')
    @patch('common_utils.json.load')
    def test_load_paths_for_timestamp_directory(self, mock_json_load, mock_open, mock_isfile):
        """Test loading paths from a directory with timestamp-based filename."""
        mock_isfile.return_value = False
        mock_json_load.return_value = {"paths": {"test": "data"}}
        
        result = load_paths_for_timestamp("/path/to/dir", "2024/01/01 12:00 UTC")
        
        expected_path = "/path/to/dir/paths_2024-01-01_12_00_00.json"
        self.assertEqual(result, {"test": "data"})
        mock_open.assert_called_once_with(expected_path)


if __name__ == '__main__':
    # Run specific test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCreateSnapshot))
    suite.addTests(loader.loadTestsFromTestCase(TestComputePredictedCounters))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateSnapshotWithPaths))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateSnapshot))
    suite.addTests(loader.loadTestsFromTestCase(TestValidatorDataFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite) 