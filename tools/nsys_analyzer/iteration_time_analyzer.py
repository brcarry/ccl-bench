#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze iteration time from nsys traces
Calculates: average iteration time, P99, CDF

METRIC 0 Implementation:
  - iteration_time_mean: Mean wall-clock duration of one training iteration
  - iteration_time_p99: 99th percentile of iteration time
"""

import subprocess
import re
import os
import numpy as np

# 1. Define valid iteration marker patterns (regex)
#    Match "Train Step X" or "training step X", exclude DeepSpeed internal markers
ITERATION_PATTERNS = [
    r'train\s*step\s*\d+',           # "Train Step 0", "train step 1"
    r'training\s*step\s*\d+',        # "training step 0"
    r'iteration\s*\d+',              # "iteration 0", "Iteration 1"
    r'step\s*\d+\s*$',               # "step 0" (end of line, avoid matching optimizer.step)
    r'forward\s*backward',           # "forward_backward" (some frameworks use this)
]

# 2. Define patterns to exclude (DeepSpeed internal markers, etc.)
EXCLUDE_PATTERNS = [
    r'optimizer',                    # DeepSpeedZeroOptimizer_Stage3.step
    r'deepspeed',                    # DeepSpeed internal markers
    r'_step$',                       # xxx._step
    r'_post_step',                   # _post_step
    r'backward_step',                # part of backward pass
    r'reduce_step',                  # reduce operations
]


def is_valid_iteration_marker(line):
    """
    Check if a line is a valid iteration marker
    
    Args:
        line: A line from NVTX output
    
    Returns:
        bool: True if this is a valid iteration marker
    """
    line_lower = line.lower()
    
    # 1. First check if it matches any exclude pattern
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, line_lower):
            return False
    
    # 2. Then check if it matches any iteration pattern
    for pattern in ITERATION_PATTERNS:
        if re.search(pattern, line_lower):
            return True
    
    return False


def analyze_iteration_time(nsys_rep_file):
    """
    Extract iteration timing information from nsys-rep file
    
    Method:
      1. Use nsys stats --report nvtx_sum to read NVTX markers
      2. Match valid iteration markers (Train Step X)
      3. Calculate statistics
    
    Returns:
        dict: Iteration time statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing iteration times in {nsys_rep_file}...")
    
    # Use nsys stats to get NVTX ranges
    cmd = ["nsys", "stats", "--report", "nvtx_sum", nsys_rep_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        output = result.stdout.decode('utf-8')
        
        iteration_times = []
        matched_markers = []  # For debugging: record matched markers
        
        # Parse NVTX ranges
        # Format: Time(%)  TotalTime(ns)  Instances  Avg(ns)  Med(ns)  Min(ns)  Max(ns)  StdDev(ns)  Style  Range
        lines = output.split('\n')
        in_data_section = False
        
        for line in lines:
            # Skip header lines until we see the separator
            if '--------' in line:
                in_data_section = True
                continue
            
            if not in_data_section:
                continue
            
            # Use precise matching logic
            if not is_valid_iteration_marker(line):
                continue
            
            # Parse data columns
            parts = line.split()
            if len(parts) < 4:
                continue
            
            try:
                # Column indices: 0=Time%, 1=TotalTime, 2=Instances, 3=Avg, 4=Med, 5=Min, 6=Max, 7=StdDev
                avg_time_ns = float(parts[3].replace(',', ''))
                
                # Validate time range is reasonable (> 100ms and < 1 hour)
                # Iteration time is typically between 1-60 seconds
                if 1e8 < avg_time_ns < 3600e9:  # 100ms - 1hour
                    iteration_times.append(avg_time_ns)
                    marker_name = ' '.join(parts[-3:]) if len(parts) >= 3 else line.strip()
                    matched_markers.append(marker_name)
                    print(f"  ✓ Found iteration marker: {marker_name} -> {avg_time_ns/1e6:.2f} ms")
            except (ValueError, IndexError):
                continue
        
        if not iteration_times:
            print("  ✗ No valid iteration markers found")
            print("  Hint: Make sure training script uses NVTX markers, e.g. nvtx.range_push('Train Step X')")
            return None
        
        # Calculate statistics
        iteration_times_ms = [t / 1e6 for t in iteration_times]
        
        stats = {
            # METRIC 0 core metrics
            "iteration_time_mean": np.mean(iteration_times_ms),      # Mean iteration time
            "iteration_time_p99": np.percentile(iteration_times_ms, 99),  # P99 iteration time
            
            # Additional statistics
            "avg_iteration_time_ms": np.mean(iteration_times_ms),    # For backward compatibility
            "p50_iteration_time_ms": np.percentile(iteration_times_ms, 50),
            "p99_iteration_time_ms": np.percentile(iteration_times_ms, 99),
            "min_iteration_time_ms": np.min(iteration_times_ms),
            "max_iteration_time_ms": np.max(iteration_times_ms),
            "std_iteration_time_ms": np.std(iteration_times_ms),
            "num_iterations": len(iteration_times_ms),
            "iteration_times_ms": iteration_times_ms,
            "matched_markers": matched_markers
        }
        
        # Print summary
        print(f"\n  === METRIC 0 Summary ===")
        print(f"  Number of iterations: {stats['num_iterations']}")
        print(f"  iteration_time_mean: {stats['iteration_time_mean']:.2f} ms")
        print(f"  iteration_time_p99: {stats['iteration_time_p99']:.2f} ms")
        
        return stats
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {nsys_rep_file}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def metric_cal(directory):
    """
    Calculate iteration time metrics
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        float: Average iteration time in milliseconds
    """
    # Find nsys-rep files
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return 0.0
    
    # Analyze first nsys file found
    for filename in os.listdir(nsys_dir):
        if filename.endswith(".nsys-rep"):
            nsys_file = os.path.join(nsys_dir, filename)
            stats = analyze_iteration_time(nsys_file)
            
            if stats:
                print("\n" + "="*60)
                print("Iteration Time Analysis Results")
                print("="*60)
                print(f"Number of iterations: {stats['num_iterations']}")
                print(f"Average iteration time: {stats['avg_iteration_time_ms']:.2f} ms")
                print(f"Median (P50) iteration time: {stats['p50_iteration_time_ms']:.2f} ms")
                print(f"P99 iteration time: {stats['p99_iteration_time_ms']:.2f} ms")
                print(f"Min iteration time: {stats['min_iteration_time_ms']:.2f} ms")
                print(f"Max iteration time: {stats['max_iteration_time_ms']:.2f} ms")
                print(f"Std deviation: {stats['std_iteration_time_ms']:.2f} ms")
                print("="*60)
                
                return stats['avg_iteration_time_ms']
            break
    
    return 0.0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        print(f"\nAverage iteration time: {result:.2f} ms")
    else:
        print("Usage: python iteration_time_analyzer.py <nsys_directory>")

