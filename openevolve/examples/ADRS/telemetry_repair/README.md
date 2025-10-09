# Network Input Validation Evolution

This example demonstrates using OpenEvolve to evolve algorithms for validating network interface telemetry data. The goal is to detect inconsistencies in network measurements by checking relationships between connected interfaces.

## Problem Description

Network telemetry data can become corrupted or inconsistent due to:
- Hardware failures
- Measurement errors  
- Configuration mistakes
- Timing issues between measurements

The challenge is to validate this data by exploiting the inherent relationships in network topology - for example, the receive rate on one interface should approximately match the transmit rate on the connected interface.

## Data Format

The algorithm works with interface telemetry data structured as:

```python
interfaces = {
    'if1': {
        'interface_status': 'up',      # 'up' or 'down'
        'rx_rate': 100.0,              # Receive rate in Mbps
        'tx_rate': 95.0,               # Transmit rate in Mbps  
        'capacity': 1000.0,            # Interface capacity in Mbps
        'connected_to': 'if2',         # ID of connected interface
        'local_router': 'router1',     # Router this interface belongs to
        'remote_router': 'router2'     # Router on other end
    },
    'if2': {
        'interface_status': 'up',
        'rx_rate': 95.0,               # Should ≈ if1's tx_rate
        'tx_rate': 100.0,              # Should ≈ if1's rx_rate
        'capacity': 1000.0,
        'connected_to': 'if1',
        'local_router': 'router2',
        'remote_router': 'router1'
    }
}
```

## Validation Logic

The algorithm validates data by checking:

1. **Rate Consistency**: My RX rate should match their TX rate (and vice versa), within a tolerance
2. **Status Consistency**: If one interface is down, the connected interface should also be down
3. **Capacity Constraints**: Rates should not exceed interface capacity
4. **Physical Constraints**: Rates should be non-negative

## Output Format

The algorithm returns the same data structure, but each telemetry value becomes a tuple:
`(original_value, repaired_value, confidence_score)`

Where:
- `original_value`: The input measurement
- `repaired_value`: Corrected value (initially same as original)
- `confidence_score`: Float between 0.0 (definitely wrong) and 1.0 (definitely correct)

## Initial Algorithm

The starting algorithm implements basic validation:
- Compares RX/TX rates between connected interfaces
- Uses 10% tolerance for rate matching
- Assigns confidence 1.0 for consistent measurements, 0.0 for inconsistent
- No repairs attempted initially (just validation)

## Evaluation Metrics

The evaluator tests algorithms against 6 test cases with known ground truth:

1. **Perfect Network**: All measurements consistent
2. **Rate Mismatch**: Large discrepancies in RX/TX rates  
3. **Status Mismatch**: One interface up, connected interface down
4. **Both Down**: Consistent down state (should be valid)
5. **Edge Case**: Small differences within tolerance
6. **Complex Network**: Multiple interface pairs, some problematic

**Metrics:**
- **Accuracy**: Correctly classified measurements (TP+TN)/(TP+FP+TN+FN)
- **Precision**: Of flagged problems, how many are real: TP/(TP+FP)
- **Recall**: Of real problems, how many were caught: TP/(TP+FN)
- **Confidence Calibration**: Do high confidence scores actually indicate reliability?
- **Combined Score**: Weighted combination (40% accuracy, 30% precision, 30% recall)

## Running the Example

1. **Test the initial program:**
```bash
cd examples/network_input_validation
python initial_program.py
python evaluator.py
```

2. **Run evolution:**
```bash
cd ../../  # Back to openevolve root
python openevolve-run.py \
  examples/network_input_validation/initial_program.py \
  examples/network_input_validation/evaluator.py \
  --config examples/network_input_validation/config.yaml
```

## Evolution Strategy

OpenEvolve will evolve the algorithm to:
- **Improve detection accuracy** by finding better ways to identify inconsistencies
- **Reduce false positives** by handling edge cases and tolerances more smartly
- **Add repair capabilities** by inferring correct values from topology relationships
- **Enhance confidence scoring** by considering multiple validation signals
- **Scale to complex networks** by handling multi-hop relationships and aggregation

## Example Improvements

Evolution might discover:
- **Dynamic tolerance adjustment** based on network conditions
- **Multi-factor validation** combining several consistency checks
- **Temporal patterns** using historical data trends
- **Topology-aware repairs** using alternate path information
- **Probabilistic confidence** based on measurement uncertainty

## Advanced Extensions

Once the basic validation works well, you could extend to:
- **Real network data** from your infrastructure
- **Complex topologies** with redundant paths and load balancing
- **Additional telemetry** like latency, packet loss, error rates
- **Integration** with network management systems
- **Real-time validation** with streaming data

This example provides a foundation for evolving sophisticated network data validation algorithms tailored to your specific topology and requirements. 