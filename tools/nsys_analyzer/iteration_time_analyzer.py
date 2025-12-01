#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze iteration time from nsys traces
Calculates: average iteration time, P99, CDF
"""

import subprocess
import re
import os
import numpy as np

def analyze_iteration_time(nsys_rep_file):
    """
    Extract iteration timing information from nsys-rep file
    
    Returns:
        dict: Iteration time statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing iteration times in {nsys_rep_file}...")
    
    # Use nsys stats to get NVTX ranges (if training script used NVTX markers)
    cmd = ["nsys", "stats", "--report", "nvtx_sum", nsys_rep_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        output = result.stdout.decode('utf-8')
        
        iteration_times = []
        
        # Parse NVTX ranges for iteration markers
        # Expected format: Time(%)  TotalTime(ns)  Instances  Avg(ns)  Med(ns)  Min(ns)  Max(ns)  StdDev(ns)  Style  Range
        lines = output.split('\n')
        in_data_section = False
        
        for line in lines:
            # Skip header lines until we see the data separator
            if '--------' in line:
                in_data_section = True
                continue
            
            if not in_data_section:
                continue
            
            # Look for iteration/step markers (case insensitive)
            line_lower = line.lower()
            if not ('iteration' in line_lower or 'step' in line_lower or 'train' in line_lower):
                continue
            
            # Split by whitespace and extract columns
            parts = line.split()
            if len(parts) < 4:
                continue
            
            try:
                # Column indices: 0=Time%, 1=TotalTime, 2=Instances, 3=Avg, 4=Med, 5=Min, 6=Max, 7=StdDev
                # We want the Avg (column 3) for iteration time
                avg_time_ns = float(parts[3].replace(',', ''))
                
                # Verify this looks like a valid iteration time (> 1ms and < 1 hour)
                if 1e6 < avg_time_ns < 3600e9:
                    iteration_times.append(avg_time_ns)
                    print(f"  Found iteration marker: {' '.join(parts[-3:])} -> {avg_time_ns/1e6:.2f} ms")
            except (ValueError, IndexError):
                continue
        
        if not iteration_times:
            print("  No explicit iteration markers found")
            return None
        
        # Calculate statistics
        iteration_times_ms = [t / 1e6 for t in iteration_times]  # Convert to milliseconds
        
        stats = {
            "avg_iteration_time_ms": np.mean(iteration_times_ms),
            "p50_iteration_time_ms": np.percentile(iteration_times_ms, 50),
            "p99_iteration_time_ms": np.percentile(iteration_times_ms, 99),
            "min_iteration_time_ms": np.min(iteration_times_ms),
            "max_iteration_time_ms": np.max(iteration_times_ms),
            "std_iteration_time_ms": np.std(iteration_times_ms),
            "num_iterations": len(iteration_times_ms),
            "iteration_times_ms": iteration_times_ms
        }
        
        return stats
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {nsys_rep_file}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
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

