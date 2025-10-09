"""
Tests for the Hodor unified repair and validation pipeline.
"""
import unittest
import sys
import os
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hodor import Hodor, HodorConfig
from dtp import RepairConfig
from validate import ValidatorConfig


class TestHodorConfig(unittest.TestCase):
    def test_default_initialization(self):
        """Test HodorConfig with default values."""
        config = HodorConfig()
        
        self.assertIsNotNone(config.repair_config)
        self.assertIsNotNone(config.validator_config)
        self.assertFalse(config.disable_cache)
        self.assertEqual(config.repair_config.disable_cache, config.disable_cache)
        self.assertEqual(config.validator_config.disable_cache, config.disable_cache)

    def test_custom_initialization(self):
        """Test HodorConfig with custom values."""
        repair_config = RepairConfig(num_trials=50, seed=123)
        validator_config = ValidatorConfig(confidence_cutoff=0.5)
        
        config = HodorConfig(
            repair_config=repair_config,
            validator_config=validator_config,
            disable_cache=True
        )
        
        self.assertEqual(config.repair_config, repair_config)
        self.assertEqual(config.validator_config, validator_config)
        self.assertTrue(config.disable_cache)

    def test_cache_propagation(self):
        """Test that cache settings propagate correctly."""
        config = HodorConfig(disable_cache=True)
        
        self.assertTrue(config.repair_config.disable_cache)
        self.assertTrue(config.validator_config.disable_cache)


class TestHodor(unittest.TestCase):
    def setUp(self):
        """Set up test data for Hodor tests."""
        self.topology_dict = {
            'nodes': [
                {'id': 0, 'name': 'NodeA'}, 
                {'id': 1, 'name': 'NodeB'},
                {'id': 2, 'name': 'NodeC'}
            ],
            'links': [
                {'source': 0, 'target': 1},
                {'source': 1, 'target': 2}
            ]
        }
        
        # Create test DataFrame with problematic data
        self.test_data = []
        for i in range(3):
            self.test_data.append({
                'timestamp': f'2024/01/01 {12+i}:00 UTC',
                'telemetry_perturbed_type': 'scaled',
                'input_perturbed_type': 'none',
                'true_detect_inconsistent': True,
                # High-level demands (must include all node pairs)
                'high_NodeA_NodeB': 100.0 + i*10,
                'high_NodeB_NodeA': 50.0 + i*5,
                'high_NodeB_NodeC': 200.0 + i*20,
                'high_NodeC_NodeB': 75.0 + i*7,
                'high_NodeA_NodeC': 150.0 + i*15,
                'high_NodeC_NodeA': 80.0 + i*8,
                # Inconsistent counter values that need repair
                'low_NodeA_egress_to_NodeB': {'perturbed': 90.0 + i*10, 'ground_truth': 100.0 + i*10},
                'low_NodeB_ingress_from_NodeA': {'perturbed': 110.0 + i*10, 'ground_truth': 100.0 + i*10},
                'low_NodeB_egress_to_NodeA': {'perturbed': 45.0 + i*5, 'ground_truth': 50.0 + i*5},
                'low_NodeA_ingress_from_NodeB': {'perturbed': 55.0 + i*5, 'ground_truth': 50.0 + i*5},
                'low_NodeB_egress_to_NodeC': {'perturbed': 180.0 + i*20, 'ground_truth': 200.0 + i*20},
                'low_NodeC_ingress_from_NodeB': {'perturbed': 220.0 + i*20, 'ground_truth': 200.0 + i*20},
                'low_NodeC_egress_to_NodeB': {'perturbed': 70.0 + i*7, 'ground_truth': 75.0 + i*7},
                'low_NodeB_ingress_from_NodeC': {'perturbed': 80.0 + i*7, 'ground_truth': 75.0 + i*7},
                'low_NodeA_egress_to_NodeC': {'perturbed': 140.0 + i*15, 'ground_truth': 150.0 + i*15},
                'low_NodeC_ingress_from_NodeA': {'perturbed': 160.0 + i*15, 'ground_truth': 150.0 + i*15},
                'low_NodeC_egress_to_NodeA': {'perturbed': 75.0 + i*8, 'ground_truth': 80.0 + i*8},
                'low_NodeA_ingress_from_NodeC': {'perturbed': 85.0 + i*8, 'ground_truth': 80.0 + i*8},
                # External interfaces
                'low_NodeA_origination': {'perturbed': 240.0 + i*25, 'ground_truth': 250.0 + i*25},
                'low_NodeA_termination': {'perturbed': 125.0 + i*13, 'ground_truth': 130.0 + i*13},
                'low_NodeB_origination': {'perturbed': 140.0 + i*15, 'ground_truth': 150.0 + i*15},
                'low_NodeB_termination': {'perturbed': 290.0 + i*30, 'ground_truth': 300.0 + i*30},
                'low_NodeC_origination': {'perturbed': 70.0 + i*7, 'ground_truth': 75.0 + i*7},
                'low_NodeC_termination': {'perturbed': 375.0 + i*38, 'ground_truth': 380.0 + i*38}
            })
        
        self.test_df = pd.DataFrame(self.test_data)
        
        # Configuration for testing
        self.config = HodorConfig(
            repair_config=RepairConfig(num_trials=10, seed=42, disable_cache=True),
            validator_config=ValidatorConfig(confidence_cutoff=0.0, disable_cache=True)
        )

    @patch('builtins.print')  # Suppress print statements during tests
    def test_initialization(self, mock_print):
        """Test Hodor initialization."""
        hodor = Hodor(self.topology_dict, self.config)
        
        self.assertEqual(hodor.topology, self.topology_dict)
        self.assertEqual(hodor.config, self.config)
        self.assertIsNotNone(hodor.network_topology)
        self.assertIsNotNone(hodor.repairer)
        self.assertIsNotNone(hodor.validator)
        
        # Check that print was called for initialization message
        mock_print.assert_called()

    @patch('builtins.print')  # Suppress print statements during tests
    def test_get_pipeline_info(self, mock_print):
        """Test pipeline information retrieval."""
        hodor = Hodor(self.topology_dict, self.config)
        info = hodor.get_pipeline_info()
        
        self.assertEqual(info['topology_nodes'], 3)
        self.assertEqual(info['repair_config'], self.config.repair_config)
        self.assertEqual(info['validator_config'], self.config.validator_config)
        self.assertEqual(info['repair_trials'], 10)
        self.assertEqual(info['validation_confidence_cutoff'], 0.0)
        self.assertEqual(info['cache_disabled'], self.config.disable_cache)
        self.assertTrue(info['ready'])

    @patch('hodor.Hodor.repair_row')
    @patch('hodor.Hodor.validate_row')
    @patch('builtins.print')  # Suppress print statements during tests
    def test_repair_and_validate_row(self, mock_print, mock_validate, mock_repair):
        """Test combined repair and validation of a single row."""
        hodor = Hodor(self.topology_dict, self.config)
        
        # Mock return values
        mock_repaired_row = pd.Series({'repair_type': 'DTP', 'repair_confidence': 0.8})
        mock_validated_row = pd.Series({'validation_type': 'w/o paths', 'validation_confidence': 0.9})
        mock_repair.return_value = mock_repaired_row
        mock_validate.return_value = mock_validated_row
        
        row = self.test_df.iloc[0]
        repaired_row, validated_row = hodor.repair_and_validate_row(row, paths=None)
        
        pd.testing.assert_series_equal(repaired_row, mock_repaired_row)
        pd.testing.assert_series_equal(validated_row, mock_validated_row)
        mock_repair.assert_called_once_with(row, None)
        mock_validate.assert_called_once_with(mock_repaired_row, None)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_repair_row(self, mock_print):
        """Test single row repair."""
        hodor = Hodor(self.topology_dict, self.config)
        row = self.test_df.iloc[0]
        
        repaired_row = hodor.repair_row(row, paths=None)
        
        # Check that repair was applied
        self.assertIn('repair_type', repaired_row)
        self.assertEqual(repaired_row['repair_type'], 'DTP')
        self.assertIn('repair_confidence', repaired_row)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_validate_row(self, mock_print):
        """Test single row validation."""
        hodor = Hodor(self.topology_dict, self.config)
        row = self.test_df.iloc[0]
        
        validated_row = hodor.validate_row(row, paths=None)
        
        # Check that validation was applied
        self.assertIn('validation_type', validated_row)
        self.assertIn('validation_confidence', validated_row)

    @patch('builtins.print')  # Suppress print statements during tests  
    def test_process_df_without_paths(self, mock_print):
        """Test DataFrame processing without paths."""
        hodor = Hodor(self.topology_dict, self.config)
        
        result_df = hodor.process_df(self.test_df, paths_path=None)
        
        # Check that processing was successful
        self.assertGreater(len(result_df), 0)
        
        # Check DataFrame attributes
        self.assertEqual(result_df.attrs['pipeline'], 'Hodor: DTP Repair + Validation')
        self.assertIn('repair_strategy', result_df.attrs)
        self.assertEqual(result_df.attrs['repair_config'], self.config.repair_config)
        self.assertEqual(result_df.attrs['validator_config'], self.config.validator_config)
        self.assertEqual(result_df.attrs['topology_nodes'], 3)
        self.assertFalse(result_df.attrs['paths_used'])
        
        # Check that rows have both repair and validation data
        if len(result_df) > 0:
            sample_row = result_df.iloc[0]
            self.assertIn('repair_type', sample_row)
            self.assertIn('validation_type', sample_row)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_process_df_with_nonexistent_paths(self, mock_print):
        """Test DataFrame processing with nonexistent paths directory."""
        hodor = Hodor(self.topology_dict, self.config)
        
        # This should handle missing paths gracefully
        result_df = hodor.process_df(self.test_df, paths_path="/nonexistent/paths/dir")
        
        # Should return empty DataFrame due to path loading failures
        self.assertEqual(len(result_df), 0)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_empty_dataframe_handling(self, mock_print):
        """Test handling of empty input DataFrame."""
        hodor = Hodor(self.topology_dict, self.config)
        empty_df = pd.DataFrame()
        
        result_df = hodor.process_df(empty_df, paths_path=None)
        
        # Should return empty DataFrame
        self.assertEqual(len(result_df), 0)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_config_propagation(self, mock_print):
        """Test that configuration is properly propagated to components."""
        custom_repair_config = RepairConfig(num_trials=25, similarity_threshold=0.1, seed=123)
        custom_validator_config = ValidatorConfig(confidence_cutoff=0.7, threshold=0.05)
        
        config = HodorConfig(
            repair_config=custom_repair_config,
            validator_config=custom_validator_config
        )
        
        hodor = Hodor(self.topology_dict, config)
        
        # Check that configurations were properly set
        self.assertEqual(hodor.repairer.config, custom_repair_config)
        self.assertEqual(hodor.validator.config, custom_validator_config)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_warning_mode_initialization(self, mock_print):
        """Test initialization with warning modes (perturbed nodes)."""
        config = HodorConfig(
            repair_config=RepairConfig(num_perturbed_nodes=2, disable_cache=True),
            validator_config=ValidatorConfig(paths_perturbed_num_nodes=1, disable_cache=True)
        )
        
        hodor = Hodor(self.topology_dict, config)
        
        # Should have printed warning messages
        warning_calls = [call for call in mock_print.call_args_list if '⚠️' in str(call)]
        self.assertEqual(len(warning_calls), 2)  # One for repair, one for validation

    @patch('builtins.print')  # Suppress print statements during tests
    def test_pipeline_resilience(self, mock_print):
        """Test that pipeline handles repair failures gracefully."""
        # Create a DataFrame that might cause repair issues
        problematic_data = [{
            'timestamp': '2024/01/01 12:00 UTC',
            'telemetry_perturbed_type': 'none',
            'input_perturbed_type': 'none',
            'true_detect_inconsistent': False,
            'high_NodeA_NodeB': 100.0,
            'high_NodeB_NodeC': 200.0,
            'high_NodeA_NodeC': 150.0,
            # Missing critical interface data
            'low_NodeA_origination': {'perturbed': 95.0},
            'low_NodeA_termination': {'perturbed': 140.0},
        }]
        
        problematic_df = pd.DataFrame(problematic_data)
        hodor = Hodor(self.topology_dict, self.config)
        
        # Should handle problematic data gracefully
        result_df = hodor.process_df(problematic_df, paths_path=None)
        
        # Result might be empty if repair completely fails, which is acceptable
        self.assertIsInstance(result_df, pd.DataFrame)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_metadata_preservation(self, mock_print):
        """Test that important metadata is preserved throughout the pipeline."""
        hodor = Hodor(self.topology_dict, self.config)
        
        result_df = hodor.process_df(self.test_df, paths_path=None)
        
        if len(result_df) > 0:
            # Check that original columns are preserved
            sample_row = result_df.iloc[0]
            self.assertIn('timestamp', sample_row)
            self.assertIn('telemetry_perturbed_type', sample_row)
            self.assertIn('true_detect_inconsistent', sample_row)
            
            # Check that new columns were added
            self.assertIn('repair_type', sample_row)
            self.assertIn('validation_type', sample_row)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_different_topologies(self, mock_print):
        """Test Hodor with different network topologies."""
        # Test with minimal topology
        minimal_topology = {
            'nodes': [{'id': 0, 'name': 'NodeA'}, {'id': 1, 'name': 'NodeB'}],
            'links': [{'source': 0, 'target': 1}]
        }
        
        hodor_minimal = Hodor(minimal_topology, self.config)
        info = hodor_minimal.get_pipeline_info()
        self.assertEqual(info['topology_nodes'], 2)
        
        # Test with larger topology
        larger_topology = {
            'nodes': [
                {'id': 0, 'name': 'NodeA'}, {'id': 1, 'name': 'NodeB'},
                {'id': 2, 'name': 'NodeC'}, {'id': 3, 'name': 'NodeD'}
            ],
            'links': [
                {'source': 0, 'target': 1}, {'source': 1, 'target': 2},
                {'source': 2, 'target': 3}, {'source': 3, 'target': 0}
            ]
        }
        
        hodor_larger = Hodor(larger_topology, self.config)
        info_larger = hodor_larger.get_pipeline_info()
        self.assertEqual(info_larger['topology_nodes'], 4)


class TestHodorIntegration(unittest.TestCase):
    """Integration tests for Hodor pipeline."""
    
    def setUp(self):
        """Set up integration test data."""
        self.topology = {
            'nodes': [{'id': 0, 'name': 'NodeA'}, {'id': 1, 'name': 'NodeB'}],
            'links': [{'source': 0, 'target': 1}]
        }
        
        # Well-formed test data that should work through the entire pipeline
        self.integration_data = [{
            'timestamp': '2024/01/01 12:00 UTC',
            'telemetry_perturbed_type': 'scaled',
            'input_perturbed_type': 'none',
            'true_detect_inconsistent': True,
            'high_NodeA_NodeB': 100.0,
            'high_NodeB_NodeA': 200.0,
            'low_NodeA_egress_to_NodeB': {'perturbed': 95.0, 'ground_truth': 100.0},
            'low_NodeB_ingress_from_NodeA': {'perturbed': 105.0, 'ground_truth': 100.0},
            'low_NodeB_egress_to_NodeA': {'perturbed': 195.0, 'ground_truth': 200.0},
            'low_NodeA_ingress_from_NodeB': {'perturbed': 205.0, 'ground_truth': 200.0},
            'low_NodeA_origination': {'perturbed': 95.0, 'ground_truth': 100.0},
            'low_NodeA_termination': {'perturbed': 205.0, 'ground_truth': 200.0},
            'low_NodeB_origination': {'perturbed': 195.0, 'ground_truth': 200.0},
            'low_NodeB_termination': {'perturbed': 105.0, 'ground_truth': 100.0}
        }]
        
        self.integration_df = pd.DataFrame(self.integration_data)

    @patch('builtins.print')  # Suppress print statements during tests
    def test_end_to_end_pipeline(self, mock_print):
        """Test complete end-to-end pipeline functionality."""
        config = HodorConfig(
            repair_config=RepairConfig(num_trials=5, seed=42, disable_cache=True),
            validator_config=ValidatorConfig(confidence_cutoff=0.0, disable_cache=True)
        )
        
        hodor = Hodor(self.topology, config)
        
        # Process the DataFrame through the complete pipeline
        result_df = hodor.process_df(self.integration_df, paths_path=None)
        
        # Verify pipeline execution
        self.assertGreater(len(result_df), 0, "Pipeline should produce results")
        
        # Verify pipeline attributes
        self.assertIn('pipeline', result_df.attrs)
        self.assertEqual(result_df.attrs['pipeline'], 'Hodor: DTP Repair + Validation')
        
        # Verify data structure
        if len(result_df) > 0:
            row = result_df.iloc[0]
            
            # Should have repair information
            self.assertIn('repair_type', row)
            self.assertIn('repair_confidence', row)
            
            # Should have validation information  
            self.assertIn('validation_type', row)
            self.assertIn('validation_confidence', row)
            
            # Should preserve original metadata
            self.assertIn('timestamp', row)
            self.assertEqual(row['timestamp'], '2024/01/01 12:00 UTC')

    @patch('builtins.print')  # Suppress print statements during tests
    def test_single_row_workflow(self, mock_print):
        """Test single row processing workflow."""
        config = HodorConfig(disable_cache=True)
        hodor = Hodor(self.topology, config)
        
        row = self.integration_df.iloc[0]
        
        # Test individual repair
        repaired_row = hodor.repair_row(row, paths=None)
        self.assertIn('repair_type', repaired_row)
        
        # Test individual validation
        validated_row = hodor.validate_row(repaired_row, paths=None)
        self.assertIn('validation_type', validated_row)
        
        # Test combined workflow
        repaired, validated = hodor.repair_and_validate_row(row, paths=None)
        self.assertIn('repair_type', repaired)
        self.assertIn('validation_type', validated)


if __name__ == '__main__':
    # Run specific test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHodorConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestHodor))
    suite.addTests(loader.loadTestsFromTestCase(TestHodorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite) 