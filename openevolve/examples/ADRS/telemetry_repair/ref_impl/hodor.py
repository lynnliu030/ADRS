"""
Hodor: A comprehensive repair and validation pipeline for network counter data.

This module provides a unified interface that combines the Democratic Trust Propagation (DTP)
repair algorithm with validation capabilities, offering both DataFrame batch processing
and individual row processing functionality.
"""
import sys
import os
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dtp import DemocraticTrustPropagationRepair, RepairConfig
from validate import Validator, ValidatorConfig
from common_utils import NetworkTopology


@dataclass
class HodorConfig:
    """Configuration for the Hodor pipeline."""
    repair_config: RepairConfig = None
    validator_config: ValidatorConfig = None
    disable_cache: bool = False
    
    def __post_init__(self):
        if self.repair_config is None:
            self.repair_config = RepairConfig(disable_cache=self.disable_cache)
        if self.validator_config is None:
            self.validator_config = ValidatorConfig(disable_cache=self.disable_cache)


class Hodor:
    """
    Hodor: The unified repair and validation pipeline.
    
    Hodor holds the door between raw network data and clean, validated measurements.
    This class orchestrates the complete pipeline:
    1. Repairs inconsistent counter values using Democratic Trust Propagation
    2. Validates the repaired data using path-based or demand invariant checking
    3. Returns a comprehensive DataFrame with repair and validation metadata
    """
    
    def __init__(self, topology: Dict, config: HodorConfig = HodorConfig()):
        """
        Initialize Hodor with network topology and configuration.
        
        Args:
            topology: Network topology dictionary
            config: Configuration for repair and validation components
        """
        self.topology = topology
        self.config = config
        self.network_topology = NetworkTopology(topology)
        
        # Initialize the repair and validation components
        self.repairer = DemocraticTrustPropagationRepair(topology, self.config.repair_config)
        self.validator = Validator(topology, self.config.validator_config)
        
        print(f"🛡️  Hodor initialized with {self.network_topology.get_node_count()} nodes")
        if self.config.repair_config.num_perturbed_nodes > 0:
            print(f"⚠️  Repair mode: {self.config.repair_config.num_perturbed_nodes} perturbed nodes")
        if self.config.validator_config.paths_perturbed_num_nodes > 0:
            print(f"⚠️  Validation mode: {self.config.validator_config.paths_perturbed_num_nodes} perturbed nodes")
    
    def process_df(self, df: pd.DataFrame, paths_path: str = None) -> pd.DataFrame:
        """
        Process a complete DataFrame through the repair and validation pipeline.
        
        Args:
            df: DataFrame containing network counter data
            paths_path: Optional path to directory/file containing paths data
            
        Returns:
            DataFrame with repaired values, confidence scores, and validation results
        """
        print(f"🔧 Hodor processing {len(df)} rows...")
        
        # Step 1: Repair the data using DTP
        print("📍 Step 1: Repairing data with Democratic Trust Propagation...")
        repaired_df = self.repairer.repair_df(df, paths_path)
        
        if len(repaired_df) == 0:
            print("⚠️  No rows were successfully repaired")
            return pd.DataFrame()
        
        print(f"✅ Repaired {len(repaired_df)} rows")
        
        # Step 2: Validate the repaired data
        print("📍 Step 2: Validating repaired data...")
        validated_df = self.validator.validate_df(repaired_df, paths_path=paths_path)
        
        if len(validated_df) == 0:
            print("⚠️  No rows were successfully validated")
            return repaired_df  # Return repaired data even if validation fails
        
        print(f"✅ Validated {len(validated_df)} rows")
        
        # Step 3: Combine results and add metadata
        result_df = validated_df.copy()
        
        # Add Hodor metadata
        result_df.attrs['pipeline'] = 'Hodor: DTP Repair + Validation'
        result_df.attrs['repair_strategy'] = repaired_df.attrs.get('Repair strategy', 'Unknown')
        result_df.attrs['repair_config'] = self.config.repair_config
        result_df.attrs['validator_config'] = self.config.validator_config
        result_df.attrs['topology_nodes'] = self.network_topology.get_node_count()
        result_df.attrs['paths_used'] = paths_path is not None
        
        print(f"🎉 Hodor pipeline complete! Processed {len(result_df)} rows")
        return result_df
    
    def repair_row(self, row: pd.Series, paths: Dict = None) -> pd.Series:
        """
        Repair a single row using the DTP algorithm.
        
        Args:
            row: Series containing network counter data for one snapshot
            paths: Optional dictionary containing paths between nodes
            
        Returns:
            Series with repaired counter values and confidence scores
        """
        print("🔧 Hodor repairing single row...")
        repaired_row = self.repairer.process_row(row, paths)
        print("✅ Single row repair complete")
        return repaired_row
    
    def validate_row(self, row: pd.Series, paths: Dict = None) -> pd.Series:
        """
        Validate a single row using path-based or demand invariant checking.
        
        Args:
            row: Series containing network counter data for one snapshot
            paths: Optional dictionary containing paths between nodes
            
        Returns:
            Series with validation results and confidence scores
        """
        print("🔍 Hodor validating single row...")
        validated_row = self.validator.validate_row(row, paths)
        print("✅ Single row validation complete")
        return validated_row
    
    def repair_and_validate_row(self, row: pd.Series, paths: Dict = None) -> Tuple[pd.Series, pd.Series]:
        """
        Repair and validate a single row in sequence.
        
        Args:
            row: Series containing network counter data for one snapshot
            paths: Optional dictionary containing paths between nodes
            
        Returns:
            Tuple of (repaired_row, validated_row)
        """
        print("🛡️  Hodor processing single row through full pipeline...")
        repaired_row = self.repair_row(row, paths)
        validated_row = self.validate_row(repaired_row, paths)
        print("✅ Single row pipeline complete")
        return repaired_row, validated_row
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the configured pipeline.
        
        Returns:
            Dictionary containing pipeline configuration and status
        """
        return {
            'topology_nodes': self.network_topology.get_node_count(),
            'repair_config': self.config.repair_config,
            'validator_config': self.config.validator_config,
            'repair_trials': self.config.repair_config.num_trials,
            'validation_confidence_cutoff': self.config.validator_config.confidence_cutoff,
            'cache_disabled': self.config.disable_cache,
            'ready': True
        }


def main():
    """Example usage of the Hodor pipeline."""
    import json
    
    # Example topology
    topology = {
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
    
    # Create sample problematic data
    sample_data = []
    for i in range(3):
        sample_data.append({
            'timestamp': f'2024/01/01 {12+i}:00 UTC',
            'telemetry_perturbed_type': 'scaled',
            'input_perturbed_type': 'none',
            'true_detect_inconsistent': True,
            # High-level demands (all node pairs)
            'high_NodeA_NodeB': 100.0 + i*10,
            'high_NodeB_NodeA': 50.0 + i*5,
            'high_NodeB_NodeC': 200.0 + i*20,
            'high_NodeC_NodeB': 75.0 + i*7,
            'high_NodeA_NodeC': 150.0 + i*15,
            'high_NodeC_NodeA': 80.0 + i*8,
            # All required low-level counter interfaces
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
    
    df = pd.DataFrame(sample_data)
    
    # Initialize Hodor
    config = HodorConfig(
        repair_config=RepairConfig(num_trials=20, seed=42, disable_cache=True),
        validator_config=ValidatorConfig(confidence_cutoff=0.0, disable_cache=True)
    )
    
    hodor = Hodor(topology, config)
    
    print("\n" + "="*60)
    print("🛡️  HODOR HOLDS THE DOOR TO CLEAN DATA!")
    print("="*60)
    
    # Process the DataFrame
    result_df = hodor.process_df(df)
    
    print(f"\n📊 Pipeline Results:")
    print(f"   • Input rows: {len(df)}")
    print(f"   • Output rows: {len(result_df)}")
    print(f"   • Pipeline: {result_df.attrs.get('pipeline', 'Unknown')}")
    print(f"   • Repair strategy: {result_df.attrs.get('repair_strategy', 'Unknown')}")
    
    if len(result_df) > 0:
        sample_row = result_df.iloc[0]
        print(f"\n🔍 Sample Results:")
        print(f"   • Repair type: {sample_row.get('repair_type', 'N/A')}")
        print(f"   • Repair confidence: {sample_row.get('repair_confidence', 'N/A')}")
        print(f"   • Validation type: {sample_row.get('validation_type', 'N/A')}")
        print(f"   • Validation confidence: {sample_row.get('validation_confidence', 'N/A')}")
    
    # Test single row repair
    print(f"\n🔧 Testing single row repair...")
    single_row = df.iloc[0]
    repaired_single = hodor.repair_row(single_row)
    print(f"   • Single row repair type: {repaired_single.get('repair_type', 'N/A')}")
    
    print(f"\n🎉 Hodor demo complete!")


if __name__ == "__main__":
    main()

