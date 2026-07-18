# COSPSolver - Quick Start Guide

 All Configuration Files Ready## 

### What Was Created

1. **cosp_algorithm_configs.json** (1.1K)
   - 7 algorithm configurations (MGM, DSA variants, MaxSum variants)
   - Ready to use with COSPSolver

2. **test_minimal.json** (1.5K)
   - Minimal scenario: 8 requests, 3 satellites
   - Perfect for quick testing (~1-5 seconds)

3. **3 Execution Scripts**
   - run_cosp_quick_test.sh - Quick validation (~1-5 sec)
   - run_cosp_full_test.sh - Full test (~30-120 sec)
   - run_cosp_perf_test.sh - Performance test (~1-5 min)

4. **Documentation**
   - COSPSOLVER_CONFIG_GUIDE.md - Complete usage guide
   - CONFIG_FILES_SUMMARY.md - Detailed overview

## 
### Fastest: Quick Test
```bash
cd satellite_scheduling
./run_cosp_quick_test.sh
```
Time: ~1-5 seconds | Output: output/cosp_quick_test.json

### Recommended: Full Test
```bash
cd satellite_scheduling
./run_cosp_full_test.sh
```
Time: ~30-120 seconds | Output: output/cosp_full_test.json

### Custom Command
```bash
cd satellite_scheduling
python3 main.py \
  --scenario scenarios/test_minimal.json \
  --algorithms_json algo_configs/cosp_algorithm_configs.json \
  --output_json output/my_test.json \
  --framework iterative_pricing \
  --max_iterations 2
```

## 
| Algorithm | Variants | Time | Best For |
|-----------|----------|------|----------|
| MGM | 2 | Fast | Small problems |
| DSA | 3 (A,B,C) | Medium | Large problems |
| MaxSum | 2 | Fast | Dense constraints |

## 
```
satellite_scheduling/
 cosp_algorithm_configs.json ............. Algorithms
 test_minimal.json ....................... Scenario (8 req)
 test.json .............................. Scenario (1000+ req)
 test_large.json ........................ Scenario (5000+ req)
 run_cosp_quick_test.sh ................. Script
 run_cosp_full_test.sh .................. Script
 run_cosp_perf_test.sh .................. Script
 COSPSOLVER_CONFIG_GUIDE.md ............. Full guide
 main.py ............................... Main entry point
 cosp_solver.py ......................... Algorithm implementation
 constraint_parser.py ................... Expression evaluator
 utils.py .............................. Integration adapters
```

 Key Features## 

 Works with both iterative_pricing and constraint_generation frameworks
 Custom constraint utilities fully supported (penalties like "-10 * var")
 3 DCOP algorithms with multiple variants
 Scenarios from 8 to 5000+ requests
 In-process execution (no external dependencies)
 Safe expression evaluation (no code injection risk)

## 
See **COSPSOLVER_CONFIG_GUIDE.md** for:
- Detailed parameter reference
- Custom scenario creation
- Output format description
- Troubleshooting guide
- Framework integration details

## 
1. **Run Quick Test**
   ```bash
   ./run_cosp_quick_test.sh
   ```

2. **Check Results**
   ```bash
   cat output/cosp_quick_test.json | head -50
   ```

3. **Run Full Test**
   ```bash
   ./run_cosp_full_test.sh
   ```

4. **Customize** parameters as needed

---

** All files created and validatedStatus**: 
**Ready to run**: Yes
**Documentation**: Complete
