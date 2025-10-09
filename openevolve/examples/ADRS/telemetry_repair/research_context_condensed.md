# Key Research Context: Validating Inputs in Software-Defined WANs

## Core Problem
Network controllers often receive **incorrect inputs** that don't accurately reflect network state, causing major outages. Over 1/3 of production outages are caused by incorrect inputs to SDN controllers.

## Key Validation Principles

### Three-Step Validation Approach (Hodor System):

1. **Signal Collection**: Gather redundant signals from network devices
2. **Signal Hardening**: Use redundancy to detect and correct faulty measurements  
3. **Dynamic Checking**: Verify inputs against hardened network state

### Critical Network Invariants for Validation:

1. **Link Symmetry (R3)**: `my_tx_rate ≈ their_rx_rate` for connected interfaces
2. **Flow Conservation (R1)**: `Σ(incoming_traffic) = Σ(outgoing_traffic) + dropped_traffic` at each router
3. **Interface Consistency**: Status should be consistent across connected pairs

## Repair Strategy from Paper

### Demand/Traffic Validation:
- **Detection**: Compare outgoing interface count to incoming interface count on each side of links
- **Repair**: Use flow conservation principle - traffic into a router must equal traffic out (plus drops)
- **Confidence**: Higher confidence when multiple redundant signals agree

### Key Equations:
```
∀v∈V, Σ(counter(e_in)) = Σ(counter(e_out)) + dropped(v)
```

### Confidence Calibration Approach:
- Use **multiple redundant signals** to increase confidence
- **Cross-validation** between different measurement sources
- **Hardening threshold** (τh ≈ 2%) to account for measurement timing differences
- **Equality threshold** (τe) for accepting repair invariants

## Practical Implementation Insights:

1. **Spurious Measurements**: Replace measurements differing by more than hardening threshold with "unknown variables"
2. **Flow Vector**: Create equations with constants (known good values) and variables (suspected bad values)
3. **Solve System**: Use flow conservation to solve for unknown/corrupted values
4. **Confidence Scoring**: Base confidence on agreement between multiple independent signals

## Application to Your Problem:
- Your `connected_to` relationships provide the link symmetry constraint
- Topology dictionary enables flow conservation checks at router level
- Multiple interfaces per router allow cross-validation
- Ground truth evaluation matches paper's validation approach

This research validates your approach of using network topology constraints for telemetry repair! 