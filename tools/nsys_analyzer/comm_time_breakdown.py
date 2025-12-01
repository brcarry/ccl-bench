#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Communication Time Breakdown Analyzer
Analyzes time spent in different types of communication operations
For DeepSpeed: DP All-Reduce, TP All-Reduce/All-Gather, PP Send/Recv

Uses accurate timeline analysis to compute true wall-clock communication time
"""

import subprocess
import os
import re
import sqlite3
from collections import defaultdict

def analyze_timeline_accurate(nsys_rep_file):
    """
    Analyze GPU timeline to compute accurate wall-clock communication time
    
    Returns:
        dict: Accurate timing statistics
    """
    # Check for existing SQLite export
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    
    if not os.path.exists(sqlite_file):
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query all CUDA kernels with timing information
        query = """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """
        
        cursor.execute(query)
        kernels = cursor.fetchall()
        conn.close()
        
        if not kernels:
            return None
        
        # Build timeline events
        timeline_events = []
        
        for start, end, name in kernels:
            name_str = str(name) if name else ""
            
            if 'nccl' in name_str.lower():
                kernel_type = 'comm'
            else:
                kernel_type = 'compute'
            
            timeline_events.append({
                'time': start,
                'type': 'start',
                'category': kernel_type
            })
            timeline_events.append({
                'time': end,
                'type': 'end',
                'category': kernel_type
            })
        
        # Sort events by time
        timeline_events.sort(key=lambda x: x['time'])
        
        # Sweep through timeline
        comm_active_count = 0
        compute_active_count = 0
        last_time = timeline_events[0]['time']
        
        # Accumulators
        total_time = 0
        comm_only_time = 0
        compute_only_time = 0
        overlap_time = 0
        idle_time = 0
        
        for event in timeline_events:
            current_time = event['time']
            duration = current_time - last_time
            
            if duration > 0:
                total_time += duration
                
                # Categorize the time period
                if comm_active_count > 0 and compute_active_count > 0:
                    overlap_time += duration
                elif comm_active_count > 0:
                    comm_only_time += duration
                elif compute_active_count > 0:
                    compute_only_time += duration
                else:
                    idle_time += duration
            
            # Update active counts
            if event['type'] == 'start':
                if event['category'] == 'comm':
                    comm_active_count += 1
                else:
                    compute_active_count += 1
            else:  # end
                if event['category'] == 'comm':
                    comm_active_count -= 1
                else:
                    compute_active_count -= 1
            
            last_time = current_time
        
        # Convert from nanoseconds to milliseconds
        total_time_ms = total_time / 1e6
        comm_only_time_ms = comm_only_time / 1e6
        compute_only_time_ms = compute_only_time / 1e6
        overlap_time_ms = overlap_time / 1e6
        idle_time_ms = idle_time / 1e6
        
        # Calculate effective times
        effective_comm_time_ms = comm_only_time_ms + overlap_time_ms
        effective_compute_time_ms = compute_only_time_ms + overlap_time_ms
        
        return {
            'total_time_ms': total_time_ms,
            'comm_only_time_ms': comm_only_time_ms,
            'compute_only_time_ms': compute_only_time_ms,
            'overlap_time_ms': overlap_time_ms,
            'idle_time_ms': idle_time_ms,
            'effective_comm_time_ms': effective_comm_time_ms,
            'effective_compute_time_ms': effective_compute_time_ms,
            'comm_percentage': (effective_comm_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0,
            'compute_percentage': (effective_compute_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0,
            'overlap_percentage': (overlap_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0,
            'idle_percentage': (idle_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        }
        
    except Exception as e:
        return None


def parse_nsys_cuda_kernel_stats(nsys_rep_file):
    """
    Parse CUDA kernel statistics from nsys-rep file
    
    Returns:
        dict: Kernel timing statistics (for breakdown by comm type)
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing communication time breakdown in {nsys_rep_file}...")
    
    # Get CUDA kernel summary
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", nsys_rep_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        output = result.stdout.decode('utf-8')
        
        # Get accurate timeline analysis
        timeline_stats = analyze_timeline_accurate(nsys_rep_file)
        
        stats = {
            "nccl_kernels": defaultdict(lambda: {"time_ns": 0, "calls": 0}),
            "compute_kernels": defaultdict(lambda: {"time_ns": 0, "calls": 0}),
            "total_nccl_time_ns": 0,
            "total_compute_time_ns": 0,
            "timeline_analysis": timeline_stats  # Add accurate timeline analysis
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # Skip header lines
            if not line.strip() or 'Time(%)' in line or '---' in line:
                continue
            
            # Parse kernel statistics
            # Format: Time(%) Total Time(ns) Instances Avg(ns) Med(ns) Min(ns) Max(ns) StdDev(ns) Name
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                # Extract fields
                time_pct = parts[0]
                total_time_str = parts[1].replace(',', '')
                instances_str = parts[2].replace(',', '')
                kernel_name = ' '.join(parts[8:])  # Name might have spaces
                
                if not total_time_str.replace('.', '').isdigit():
                    continue
                if not instances_str.isdigit():
                    continue
                
                total_time_ns = float(total_time_str)
                instances = int(instances_str)
                
                # Categorize kernel
                if 'nccl' in kernel_name.lower():
                    # NCCL communication kernel
                    stats["total_nccl_time_ns"] += total_time_ns
                    
                    # Categorize by communication type
                    if "AllReduce" in kernel_name:
                        cat = "AllReduce"
                    elif "ReduceScatter" in kernel_name:
                        cat = "ReduceScatter"
                    elif "AllGather" in kernel_name:
                        cat = "AllGather"
                    elif "Send" in kernel_name or "Recv" in kernel_name:
                        cat = "SendRecv"
                    elif "Broadcast" in kernel_name:
                        cat = "Broadcast"
                    else:
                        cat = "Other_NCCL"
                    
                    stats["nccl_kernels"][cat]["time_ns"] += total_time_ns
                    stats["nccl_kernels"][cat]["calls"] += instances
                else:
                    # Compute kernel
                    stats["total_compute_time_ns"] += total_time_ns
                    stats["compute_kernels"]["compute"]["time_ns"] += total_time_ns
                    stats["compute_kernels"]["compute"]["calls"] += instances
                    
            except (ValueError, IndexError) as e:
                continue
        
        return stats
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {nsys_rep_file}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def analyze_comm_breakdown(nsys_path):
    """
    Analyze communication time breakdown for all traces in directory, or single trace file
    
    Args:
        nsys_path: Directory containing nsys-rep files, or single nsys-rep file path
    
    Returns:
        dict: Aggregated communication breakdown statistics
    """
    results = {
        "total_nccl_time_ms": 0,
        "total_compute_time_ms": 0,
        "nccl_breakdown": defaultdict(lambda: {"time_ms": 0, "calls": 0}),
        "files_analyzed": []
    }
    
    # 1. Check if input is single file or directory
    nsys_files = []
    if os.path.isfile(nsys_path):
        # Single file mode
        if nsys_path.endswith(".nsys-rep"):
            nsys_files.append(nsys_path)
        else:
            print(f"Error: File is not a .nsys-rep file: {nsys_path}")
            return results
    elif os.path.isdir(nsys_path):
        # Directory mode: find all nsys-rep files
        for filename in os.listdir(nsys_path):
            if filename.endswith(".nsys-rep"):
                nsys_files.append(os.path.join(nsys_path, filename))
    else:
        print(f"Error: Invalid path: {nsys_path}")
        return results
    
    if not nsys_files:
        print(f"No .nsys-rep files found in {nsys_path}")
        return results
    
    print(f"Found {len(nsys_files)} nsys-rep file(s)\n")
    
    # 2. Analyze each file
    for nsys_file in nsys_files:
        stats = parse_nsys_cuda_kernel_stats(nsys_file)
        if stats:
            results["total_nccl_time_ms"] += stats["total_nccl_time_ns"] / 1e6
            results["total_compute_time_ms"] += stats["total_compute_time_ns"] / 1e6
            
            for cat, data in stats["nccl_kernels"].items():
                results["nccl_breakdown"][cat]["time_ms"] += data["time_ns"] / 1e6
                results["nccl_breakdown"][cat]["calls"] += data["calls"]
            
            results["files_analyzed"].append(os.path.basename(nsys_file))
            print(f"  NCCL kernel time (cumulative): {stats['total_nccl_time_ns']/1e6:.2f} ms")
            print(f"  Compute kernel time (cumulative): {stats['total_compute_time_ns']/1e6:.2f} ms")
            
            # Print accurate timeline analysis if available
            if stats.get('timeline_analysis'):
                tl = stats['timeline_analysis']
                print(f"\n  Accurate Timeline Analysis:")
                print(f"    Total GPU time: {tl['total_time_ms']:.2f} ms")
                print(f"    Communication only: {tl['comm_only_time_ms']:.2f} ms ({tl['comm_only_time_ms']/tl['total_time_ms']*100:.1f}%)")
                print(f"    Compute only: {tl['compute_only_time_ms']:.2f} ms ({tl['compute_only_time_ms']/tl['total_time_ms']*100:.1f}%)")
                print(f"    Overlap: {tl['overlap_time_ms']:.2f} ms ({tl['overlap_percentage']:.1f}%)")
                print(f"    Idle: {tl['idle_time_ms']:.2f} ms ({tl['idle_percentage']:.1f}%)")
                print(f"    ---")
                print(f"    Effective comm: {tl['effective_comm_time_ms']:.2f} ms ({tl['comm_percentage']:.1f}%)")
                print(f"    Effective compute: {tl['effective_compute_time_ms']:.2f} ms ({tl['compute_percentage']:.1f}%)")
                
                # Store timeline analysis in results
                results['timeline_analysis'] = tl
            
            print()
    
    return results

def metric_cal(directory):
    """
    Calculate communication time breakdown metrics
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        float: Percentage of time spent in communication
    """
    # Find nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return 0.0
    
    results = analyze_comm_breakdown(nsys_dir)
    
    print("\n" + "="*60)
    print("Communication Time Breakdown Analysis")
    print("="*60)
    print(f"Files analyzed: {', '.join(results['files_analyzed'])}")
    print(f"\nTotal time: {results['total_time_ms']:.2f} ms")
    print(f"Total NCCL communication time: {results['total_nccl_time_ms']:.2f} ms")
    print(f"Total compute time: {results['total_compute_time_ms']:.2f} ms")
    
    if results['total_time_ms'] > 0:
        comm_pct = (results['total_nccl_time_ms'] / results['total_time_ms']) * 100
        compute_pct = (results['total_compute_time_ms'] / results['total_time_ms']) * 100
        print(f"\nCommunication percentage: {comm_pct:.2f}%")
        print(f"Compute percentage: {compute_pct:.2f}%")
    
    print("\nNCCL Communication Breakdown:")
    for cat, data in sorted(results['nccl_breakdown'].items()):
        time_ms = data['time_ms']
        calls = data['calls']
        if results['total_nccl_time_ms'] > 0:
            pct = (time_ms / results['total_nccl_time_ms']) * 100
            print(f"  {cat}: {time_ms:.2f} ms ({pct:.1f}%), {calls} calls")
        else:
            print(f"  {cat}: {time_ms:.2f} ms, {calls} calls")
    
    print("="*60)
    
    # Return communication percentage
    if results['total_time_ms'] > 0:
        return (results['total_nccl_time_ms'] / results['total_time_ms']) * 100
    return 0.0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        print(f"\nCommunication time percentage: {result:.2f}%")
    else:
        print("Usage: python comm_time_breakdown.py <nsys_directory>")

