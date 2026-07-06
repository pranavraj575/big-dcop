# COSPSolver Configuration Files and Usage Guide

## Overview
This directory contains configuration files for running the integrated COSPSolver system with the satellite scheduling problem. The COSPSolver implements three DCOP algorithms (MGM, DSA, MaxSum) and integrates seamlessly with both constraint_generation and iterative_pricing frameworks.

## Configuration Files

### 1. Algorithm Configurations

#### `cosp_algorithm_configs.json`
Contains COSPSolver algorithm configurations for MGM, DSA (variants A/B/C), and MaxSum:

```json
[
  {
    "name": "mgm",                    # Algorithm identifier
    "display_name": "MGM",            # Display name for results
    "algo_params": [
      "max_iterations:100",           # Maximum iterations
      "convergence_threshold:0"       # Convergence threshold
    ]
  },
  ...
]
```

**Supported Algorithms:**
- **mgm**: Maximum Gain Message - Greedy local search algorithm
- **dsa**: Distributed Stochastic Algorithm (variants A, B, C)
- **maxsum**: MaxSum/Belief Propagation algorithm

**Common Parameters:**
- `max_iterations`: Maximum number of iterations (default: 100)
- `convergence_threshold`: Convergence detection threshold (default: 0)
- `stop_cycle`: Early stopping cycle number (optional)

### 2. Scenario Files

#### `test.json` (Large: 38.9K)
Full-scale test scenario with ~1000 satellite requests and 3 satellites. Use for comprehensive testing.

#### `test_large.json` (Very Large: 702K)
Extended scenario for performance evaluation.

#### `test_minimal.json` (Small: 1.5K) - NEW
Minimal test scenario with 8 requests and 3 satellites. Use for quick testing and debugging.

**Scenario Structure:**
```json
{
  "horizon": {
    "start": 69347.0,        # Simulation start time
    "end": 90947.0           # Simulation end time
  },
  "requests": [
    {
      "request_id": "req_id_time1_time2_DESIRED"
    },
    ...
  ],
  "agents": [
    {
      "agent_id": "sat_id",
      "identifier": "sat_id",
      "data_volume_MB": 100000.0,
      "downlinks": [
        {
          "receiver_id": "gs_id",
          "data_volume_Mbps": 100.0,
          "start_time": 69347.0,
          "end_time": 90947.0
        }
      ]
    },
    ...
  ]
}
```

## Usage

### Basic Command Line Usage

```bash
cd /path/to/satellite_scheduling

# Run with COSPSolver + iterative_pricing (default)
python3 main.py \
  --scenario test_minimal.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/cosp_test.json \
  --framework iterative_pricing \
  --max_iterations 4

# Run with COSPSolver + constraint_generation
python3 main.py \
  --scenario test_minimal.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/cosp_test.json \
  --framework constraint_generation \
  --max_iterations 4
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | `satellite_scheduling/test.json` | Input scenario JSON file |
| `--output_json` | `output/test_main.json` | Output results JSON file |
| `--algorithms_json` | `satellite_scheduling/rm_algorithm_configs.json` | Algorithm configs JSON file |
| `--framework` | `iterative_pricing` | Framework: `iterative_pricing` or `constraint_generation` |
| `--pydcop_mode` | `thread` | Execution mode: `thread` or `process` (for compatibility) |
| `--timeout` | `-1` | Timeout in seconds (-1 for no timeout) |
| `--max_iterations` | `4` | Number of iterations for the framework |

### Quick Start Examples

**1. Quick Test (Minimal Scenario, MGM Only)**
```bash
python3 main.py \
  --scenario test_minimal.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/quick_test.json \
  --framework iterative_pricing \
  --max_iterations 2
```

**2. Full Test (Large Scenario, All Algorithms)**
```bash
python3 main.py \
  --scenario test.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/full_test.json \
  --framework constraint_generation \
  --max_iterations 5
```

**3. Performance Test (Very Large Scenario)**
```bash
python3 main.py \
  --scenario test_large.json \
  --algorithms_json cosp_algorithm_configs.json \
  --output_json output/perf_test.json \
  --framework iterative_pricing \
  --max_iterations 3 \
  --timeout 300
```

## Algorithm Configuration Examples

### MGM (Maximum Gain Message)
Conservative approach with early stopping:
```json
{
  "name": "mgm",
  "display_name": "MGM-Conservative",
  "algo_params": [
    "max_iterations:50",
    "break_mode:first",
    "stop_cycle:10"
  ]
}
```

### DSA (Distributed Stochastic Algorithm)
Variant B with medium probability:
```json
{
  "name": "dsa",
  "display_name": "DSA-B-Aggressive",
  "algo_params": [
    "max_iterations:100",
    "variant:B",
    "probability:0.7"
  ]
}
```

### MaxSum (Belief Propagation)
With damping for stability:
```json
{
  "name": "maxsum",
  "display_name": "MaxSum-Stable",
  "algo_params": [
    "max_iterations:100",
    "damping:0.5",
    "stability:0.0001"
  ]
}
```

## Output Format

The output JSON file contains:

```json
{
  "run_info": {
    "scenario": "test_minimal.json",
    "algorithms_json": "cosp_algorithm_configs.json"
  },
  "algorithm_configs": [...],
  "output": {
    "MGM": {
      "data": {...},
      "aux_info": {
        "best_total_scheduled": 8,
        "best_iter": 3
      }
    },
    "DSA-A": {
      "data": {...},
      "aux_info": {
        "best_total_scheduled": 7,
        "best_iter": 2
      }
    },
    ...
  }
}
```

## Integration with Workflows

### Constraint Generation Workflow
- Uses COSPSolver via `run_global_dispatcher_cosp()` in `utils.py`
- Solves constraints added progressively to the DCOP problem
- Custom constraint utilities are automatically evaluated

### Iterative Pricing Workflow
- Uses COSPSolver via `run_global_dispatcher_cosp()` in `utils.py`
- Applies dynamic penalty constraints: `-penalty * var_name`
- Penalty values updated iteratively based on dropped requests
- Custom constraint utilities enable proper penalty evaluation

## Key Features

✅ **Custom Constraint Utilities** - Penalty functions like "-10 * penalty_var" fully evaluated
✅ **Format Transparency** - Both simple lists and full pydcop constraint specs supported
✅ **Safe Expression Evaluation** - Restricted eval() prevents code injection
✅ **All 3 Algorithms** - MGM, DSA, MaxSum all support custom utilities
✅ **In-Process Execution** - No external CLI dependencies
✅ **Backward Compatible** - Existing code unchanged

## Performance Notes

- **Minimal scenario**: ~1-5 seconds per algorithm
- **Test scenario**: ~10-30 seconds per algorithm (depending on max_iterations)
- **Large scenario**: ~1-5 minutes per algorithm
- **Framework**: iterative_pricing typically faster than constraint_generation

## Troubleshooting

### Issue: "FileNotFoundError: scenario file not found"
**Solution**: Verify the scenario file path is correct and file exists
```bash
ls -la satellite_scheduling/test_minimal.json
```

### Issue: "Module not found: ortools"
**Solution**: Install ortools (required for local solver)
```bash
pip install ortools
```

### Issue: "Invalid algorithm name"
**Solution**: Check algorithm name in config is one of: `mgm`, `dsa`, `maxsum`

### Issue: Output file already exists
**Solution**: Delete existing output file or use different output path
```bash
rm output/cosp_test.json
```

## Creating Custom Scenarios

To create your own scenario file:

1. Define horizon (start/end times in seconds)
2. Add requests with unique IDs
3. Define agents (satellites) with:
   - agent_id: unique identifier
   - data_volume_MB: storage capacity
   - downlinks: ground station connections with bandwidth

Example:
```json
{
  "horizon": {
    "start": 0.0,
    "end": 86400.0
  },
  "requests": [
    {"request_id": "my_request_1"},
    {"request_id": "my_request_2"}
  ],
  "agents": [
    {
      "agent_id": "my_satellite",
      "identifier": "my_satellite",
      "data_volume_MB": 50000.0,
      "downlinks": [{
        "receiver_id": "my_ground_station",
        "data_volume_Mbps": 50.0,
        "start_time": 0.0,
        "end_time": 86400.0
      }]
    }
  ]
}
```

## Next Steps

1. **Quick Start**: Run the minimal test to verify installation
2. **Testing**: Run full test with your preferred framework
3. **Tuning**: Adjust algorithm parameters based on results
4. **Production**: Scale to larger scenarios as needed

## References

- COSPSolver Implementation: `cosp_solver.py`
- Constraint Parser: `constraint_parser.py`
- Integration Adapter: `utils.py`
- Main Workflow: `main.py`
- Constraint Generation: `constraint_generation.py`
- Iterative Pricing: `iterative_pricing.py`
