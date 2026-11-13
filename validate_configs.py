#!/usr/bin/env python3
"""Validate all configuration files are in place."""

import json
import os

print("\n" + "=" * 70)
print("CONFIGURATION FILES VALIDATION")
print("=" * 70 + "\n")

# Check all required files exist
files_to_check = [
    "satellite_scheduling/algo_configs/cosp_algorithm_configs.json",
    "satellite_scheduling/scenarios/test_minimal.json",
    "satellite_scheduling/scenarios/test.json",
    "satellite_scheduling/COSPSOLVER_CONFIG_GUIDE.md",
]

scripts_to_check = [
    "satellite_scheduling/hyperparam_tuning.sh",
    "satellite_scheduling/run_experiments.sh",
]

print("Config and Scenario Files:")
print("-" * 70)
for filepath in files_to_check:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print("✓ {:<50} ({} bytes)".format(filepath, f"{size:,}"))
    else:
        print(f"✗ {filepath:<50} MISSING!")

print("\nExecution Scripts:")
print("-" * 70)
for filepath in scripts_to_check:
    if os.path.exists(filepath):
        if os.access(filepath, os.X_OK):
            print(f"✓ {filepath:<50} (executable)")
        else:
            print(f"⚠ {filepath:<50} (not executable)")
    else:
        print(f"✗ {filepath:<50} MISSING!")

# Validate JSON files
print("\nJSON Configuration Validation:")
print("-" * 70)

try:
    with open("satellite_scheduling/algo_configs/cosp_algorithm_configs.json") as f:
        configs = json.load(f)
    print(f"✓ cosp_algorithm_configs.json: {len(configs)} algorithms defined")
    for cfg in configs:
        display_name = cfg.get("display_name", cfg.get("name"))
        print("  - {}: {}".format(display_name, cfg["name"]))
except Exception as e:
    print(f"✗ Error parsing cosp_algorithm_configs.json: {e}")

try:
    with open("satellite_scheduling/scenarios/test_minimal.json") as f:
        scenario = json.load(f)
    requests = len(scenario.get("requests", []))
    agents = len(scenario.get("agents", []))
    print(f"✓ test_minimal.json: {requests} requests, {agents} agents")
except Exception as e:
    print(f"✗ Error parsing test_minimal.json: {e}")

print("\nDocumentation:")
print("-" * 70)
guide_path = "satellite_scheduling/COSPSOLVER_CONFIG_GUIDE.md"
if os.path.exists(guide_path):
    size = os.path.getsize(guide_path)
    with open(guide_path) as f:
        lines = len(f.readlines())
    print("✓ COSPSOLVER_CONFIG_GUIDE.md: {} lines, {} bytes".format(lines, f"{size:,}"))
else:
    print("✗ COSPSOLVER_CONFIG_GUIDE.md: MISSING!")

print("\n" + "=" * 70)
print("READY TO RUN")
print("=" * 70 + "\n")

print("Quick Start Commands:")
print("-" * 70)
print("1. Quick test (8 requests, 2 iterations):")
print("   cd satellite_scheduling && ./run_cosp_quick_test.sh\n")

print("2. Full test (1000+ requests, 4 iterations):")
print("   cd satellite_scheduling && ./run_cosp_full_test.sh\n")

print("3. Custom command:")
print("   cd satellite_scheduling")
print("   python3 main.py --scenario test_minimal.json \\")
print("     --algorithms_json cosp_algorithm_configs.json \\")
print("     --output_json output/test.json \\")
print("     --framework iterative_pricing --max_iterations 2\n")

print("See COSPSOLVER_CONFIG_GUIDE.md for detailed documentation\n")
