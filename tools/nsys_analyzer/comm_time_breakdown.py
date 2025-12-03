#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 1: Communication Time Breakdown Analyzer

Analyzes time spent in different parallelism types:
  - compute_time: Time spent in compute kernels
  - comm_DP_time: Data Parallel communication (ReduceScatter, AllGather in ZeRO, AllReduce for gradients)
  - comm_TP_time: Tensor Parallel communication (AllReduce/AllGather within TP group)
  - comm_PP_time: Pipeline Parallel communication (Send/Recv between stages)
  - comm_EP_time: Expert Parallel communication (AllToAll for MoE)
  - idle_time: GPU idle time

Uses accurate timeline analysis to compute true wall-clock communication time.

Note: Parallelism type inference is heuristic-based. For accurate classification,
provide TP/PP/DP/EP sizes in config or use NCCL communicator IDs.
"""

import subprocess
import os
import re
import sqlite3
from collections import defaultdict
import numpy as np


def categorize_nccl_kernel(kernel_name, tp_size=1, pp_size=1, dp_size=1, ep_size=1):
    """
    Categorize NCCL kernel by parallelism type (DP/TP/PP/EP)
    
    Heuristic rules for DeepSpeed ZeRO-3:
      - ReduceScatter: DP (ZeRO gradient partitioning)
      - AllGather: DP (ZeRO parameter gathering) - NOT TP!
      - AllReduce: DP (gradient aggregation) or TP (tensor parallel reduce)
      - Send/Recv: PP (pipeline parallel)
      - AllToAll: EP (expert parallel for MoE)
      - Broadcast: OTHER (initialization, etc.)
    
    Args:
        kernel_name: NCCL kernel name string
        tp_size: Tensor parallel size (for future use with config)
        pp_size: Pipeline parallel size
        dp_size: Data parallel size
        ep_size: Expert parallel size
    
    Returns:
        str: Parallelism type ('DP', 'TP', 'PP', 'EP', 'OTHER')
    """
    name_lower = kernel_name.lower()
    
    # 1. Send/Recv -> PP (Pipeline Parallel)
    if 'send' in name_lower or 'recv' in name_lower:
        return 'PP'
    
    # 2. AllToAll -> EP (Expert Parallel for MoE)
    if 'alltoall' in name_lower:
        return 'EP'
    
    # 3. ReduceScatter -> DP (ZeRO gradient partitioning)
    if 'reducescatter' in name_lower:
        return 'DP'
    
    # 4. AllGather -> DP (ZeRO parameter gathering)
    #    Note: In ZeRO-3, AllGather is used to gather partitioned parameters
    #    This is DP communication, not TP!
    if 'allgather' in name_lower:
        return 'DP'
    
    # 5. AllReduce -> Could be DP or TP
    #    Heuristic: If TP is enabled (tp_size > 1), some AllReduce are for TP
    #    Without config, assume AllReduce is DP (gradient aggregation)
    if 'allreduce' in name_lower:
        # TODO: With config, can check message size to distinguish DP vs TP
        # TP AllReduce typically has smaller message size (hidden_dim / tp_size)
        # DP AllReduce has full gradient size
        if tp_size > 1:
            # If TP is enabled, classify based on additional heuristics
            # For now, still assume DP as default
            return 'DP'
        return 'DP'
    
    # 6. Broadcast -> OTHER (usually initialization)
    if 'broadcast' in name_lower:
        return 'OTHER'
    
    # 7. Default -> OTHER
    return 'OTHER'


def analyze_timeline_by_parallelism(nsys_rep_file, tp_size=1, pp_size=1, dp_size=1, ep_size=1):
    """
    Analyze GPU timeline to compute wall-clock time by parallelism type
    
    This is the key function for METRIC 1. It uses timeline sweep to compute
    accurate wall-clock time for each category, handling overlapping kernels correctly.
    
    Returns:
        dict: Time breakdown by category (DP, TP, PP, EP, compute, idle)
    """
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    
    if not os.path.exists(sqlite_file):
        print(f"  SQLite file not found: {sqlite_file}")
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
        
        # Build timeline events with parallelism category
        timeline_events = []
        category_counts = defaultdict(int)  # For debugging
        
        for start, end, name in kernels:
            name_str = str(name) if name else ""
            
            if 'nccl' in name_str.lower():
                # Categorize by parallelism type
                category = categorize_nccl_kernel(name_str, tp_size, pp_size, dp_size, ep_size)
                category_counts[category] += 1
            else:
                category = 'compute'
                category_counts['compute'] += 1
            
            timeline_events.append({
                'time': start,
                'type': 'start',
                'category': category
            })
            timeline_events.append({
                'time': end,
                'type': 'end',
                'category': category
            })
        
        # Sort events by time
        timeline_events.sort(key=lambda x: x['time'])
        
        # Track active counts for each category
        active_counts = defaultdict(int)
        last_time = timeline_events[0]['time']
        
        # Accumulators for each category
        time_accum = defaultdict(float)  # DP, TP, PP, EP, compute
        total_time = 0
        idle_time = 0
        overlap_time = 0
        
        for event in timeline_events:
            current_time = event['time']
            duration = current_time - last_time
            
            if duration > 0:
                total_time += duration
                
                # Count active categories
                active_categories = [cat for cat, cnt in active_counts.items() if cnt > 0]
                num_active = len(active_categories)
                
                if num_active == 0:
                    # Idle time
                    idle_time += duration
                elif num_active == 1:
                    # Single category active
                    time_accum[active_categories[0]] += duration
                else:
                    # Multiple categories active (overlap)
                    overlap_time += duration
                    # Distribute time equally among active categories
                    for cat in active_categories:
                        time_accum[cat] += duration / num_active
            
            # Update active counts
            if event['type'] == 'start':
                active_counts[event['category']] += 1
            else:
                active_counts[event['category']] -= 1
            
            last_time = current_time
        
        # Convert to milliseconds
        total_time_ms = total_time / 1e6
        idle_time_ms = idle_time / 1e6
        overlap_time_ms = overlap_time / 1e6
        
        # Calculate time for each category
        result = {
            'total_time_ms': total_time_ms,
            'compute_time_ms': time_accum['compute'] / 1e6,
            'comm_DP_time_ms': time_accum['DP'] / 1e6,
            'comm_TP_time_ms': time_accum['TP'] / 1e6,
            'comm_PP_time_ms': time_accum['PP'] / 1e6,
            'comm_EP_time_ms': time_accum['EP'] / 1e6,
            'comm_OTHER_time_ms': time_accum['OTHER'] / 1e6,
            'idle_time_ms': idle_time_ms,
            'overlap_time_ms': overlap_time_ms,
            'category_counts': dict(category_counts)
        }
        
        # Calculate total communication time
        result['total_comm_time_ms'] = (
            result['comm_DP_time_ms'] + 
            result['comm_TP_time_ms'] + 
            result['comm_PP_time_ms'] + 
            result['comm_EP_time_ms'] +
            result['comm_OTHER_time_ms']
        )
        
        # Calculate percentages
        if total_time_ms > 0:
            result['compute_ratio'] = result['compute_time_ms'] / total_time_ms
            result['comm_DP_ratio'] = result['comm_DP_time_ms'] / total_time_ms
            result['comm_TP_ratio'] = result['comm_TP_time_ms'] / total_time_ms
            result['comm_PP_ratio'] = result['comm_PP_time_ms'] / total_time_ms
            result['comm_EP_ratio'] = result['comm_EP_time_ms'] / total_time_ms
            result['comm_OTHER_ratio'] = result['comm_OTHER_time_ms'] / total_time_ms
            result['idle_ratio'] = result['idle_time_ms'] / total_time_ms
        
        return result
        
    except Exception as e:
        print(f"  Error in timeline analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_timeline_accurate(nsys_rep_file):
    """
    Analyze GPU timeline to compute accurate wall-clock communication time
    (Backward compatible function - uses simple comm/compute classification)
    
    Returns:
        dict: Accurate timing statistics
    """
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    
    if not os.path.exists(sqlite_file):
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
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
                
                if comm_active_count > 0 and compute_active_count > 0:
                    overlap_time += duration
                elif comm_active_count > 0:
                    comm_only_time += duration
                elif compute_active_count > 0:
                    compute_only_time += duration
                else:
                    idle_time += duration
            
            if event['type'] == 'start':
                if event['category'] == 'comm':
                    comm_active_count += 1
                else:
                    compute_active_count += 1
            else:
                if event['category'] == 'comm':
                    comm_active_count -= 1
                else:
                    compute_active_count -= 1
            
            last_time = current_time
        
        # Convert to milliseconds
        total_time_ms = total_time / 1e6
        comm_only_time_ms = comm_only_time / 1e6
        compute_only_time_ms = compute_only_time / 1e6
        overlap_time_ms = overlap_time / 1e6
        idle_time_ms = idle_time / 1e6
        
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
        dict: Kernel timing statistics
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
        
        # Get accurate timeline analysis (simple comm/compute)
        timeline_stats = analyze_timeline_accurate(nsys_rep_file)
        
        # Get parallelism-aware timeline analysis (METRIC 1)
        parallelism_stats = analyze_timeline_by_parallelism(nsys_rep_file)
        
        stats = {
            "nccl_kernels": defaultdict(lambda: {"time_ns": 0, "calls": 0}),
            "compute_kernels": defaultdict(lambda: {"time_ns": 0, "calls": 0}),
            "total_nccl_time_ns": 0,
            "total_compute_time_ns": 0,
            "timeline_analysis": timeline_stats,
            "parallelism_breakdown": parallelism_stats  # METRIC 1 data
        }
        
        lines = output.split('\n')
        
        for line in lines:
            if not line.strip() or 'Time(%)' in line or '---' in line:
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                total_time_str = parts[1].replace(',', '')
                instances_str = parts[2].replace(',', '')
                kernel_name = ' '.join(parts[8:])
                
                if not total_time_str.replace('.', '').isdigit():
                    continue
                if not instances_str.isdigit():
                    continue
                
                total_time_ns = float(total_time_str)
                instances = int(instances_str)
                
                if 'nccl' in kernel_name.lower():
                    stats["total_nccl_time_ns"] += total_time_ns
                    
                    # Categorize by parallelism type (for cumulative stats)
                    cat = categorize_nccl_kernel(kernel_name)
                    stats["nccl_kernels"][cat]["time_ns"] += total_time_ns
                    stats["nccl_kernels"][cat]["calls"] += instances
                else:
                    stats["total_compute_time_ns"] += total_time_ns
                    stats["compute_kernels"]["compute"]["time_ns"] += total_time_ns
                    stats["compute_kernels"]["compute"]["calls"] += instances
                    
            except (ValueError, IndexError):
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
    
    METRIC 1 Output Format:
      - compute_time_ms / compute_ratio
      - comm_DP_time_ms / comm_DP_ratio
      - comm_TP_time_ms / comm_TP_ratio
      - comm_PP_time_ms / comm_PP_ratio
      - comm_EP_time_ms / comm_EP_ratio
      - idle_time_ms / idle_ratio
    
    Args:
        nsys_path: Directory containing nsys-rep files, or single nsys-rep file path
    
    Returns:
        dict: Aggregated communication breakdown statistics
    """
    results = {
        "total_nccl_time_ms": 0,
        "total_compute_time_ms": 0,
        "nccl_breakdown": defaultdict(lambda: {"time_ms": 0, "calls": 0}),
        "files_analyzed": [],
        # METRIC 1 fields
        "parallelism_breakdown": None
    }
    
    # 1. Check if input is single file or directory
    nsys_files = []
    if os.path.isfile(nsys_path):
        if nsys_path.endswith(".nsys-rep"):
            nsys_files.append(nsys_path)
        else:
            print(f"Error: File is not a .nsys-rep file: {nsys_path}")
            return results
    elif os.path.isdir(nsys_path):
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
            
            # Print accurate timeline analysis (simple comm/compute)
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
                
                results['timeline_analysis'] = tl
            
            # Print METRIC 1 parallelism breakdown
            if stats.get('parallelism_breakdown'):
                pb = stats['parallelism_breakdown']
                print(f"\n  === METRIC 1: Time Breakdown by Parallelism ===")
                print(f"    Total GPU time: {pb['total_time_ms']:.2f} ms")
                print(f"    compute_time:   {pb['compute_time_ms']:.2f} ms ({pb.get('compute_ratio', 0)*100:.1f}%)")
                print(f"    comm_DP_time:   {pb['comm_DP_time_ms']:.2f} ms ({pb.get('comm_DP_ratio', 0)*100:.1f}%)")
                print(f"    comm_TP_time:   {pb['comm_TP_time_ms']:.2f} ms ({pb.get('comm_TP_ratio', 0)*100:.1f}%)")
                print(f"    comm_PP_time:   {pb['comm_PP_time_ms']:.2f} ms ({pb.get('comm_PP_ratio', 0)*100:.1f}%)")
                print(f"    comm_EP_time:   {pb['comm_EP_time_ms']:.2f} ms ({pb.get('comm_EP_ratio', 0)*100:.1f}%)")
                print(f"    comm_OTHER:     {pb['comm_OTHER_time_ms']:.2f} ms ({pb.get('comm_OTHER_ratio', 0)*100:.1f}%)")
                print(f"    idle_time:      {pb['idle_time_ms']:.2f} ms ({pb.get('idle_ratio', 0)*100:.1f}%)")
                print(f"    ---")
                print(f"    Total comm:     {pb['total_comm_time_ms']:.2f} ms")
                print(f"    Overlap:        {pb['overlap_time_ms']:.2f} ms")
                
                if pb.get('category_counts'):
                    print(f"\n    Kernel counts by category:")
                    for cat, count in sorted(pb['category_counts'].items()):
                        print(f"      {cat}: {count} kernels")
                
                results['parallelism_breakdown'] = pb
            
            print()
    
    return results

def metric_cal(directory):
    """
    Calculate communication time breakdown metrics (METRIC 1)
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        dict: METRIC 1 results with time breakdown by parallelism type
    """
    # Find nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return {}
    
    results = analyze_comm_breakdown(nsys_dir)
    
    print("\n" + "="*70)
    print("METRIC 1: Communication Time Breakdown Analysis")
    print("="*70)
    print(f"Files analyzed: {', '.join(results['files_analyzed'])}")
    
    # Print METRIC 1 summary
    if results.get('parallelism_breakdown'):
        pb = results['parallelism_breakdown']
        print(f"\n{'Category':<20} {'Time (ms)':>12} {'Ratio':>10}")
        print("-" * 44)
        print(f"{'compute_time':<20} {pb['compute_time_ms']:>12.2f} {pb.get('compute_ratio', 0)*100:>9.1f}%")
        print(f"{'comm_DP_time':<20} {pb['comm_DP_time_ms']:>12.2f} {pb.get('comm_DP_ratio', 0)*100:>9.1f}%")
        print(f"{'comm_TP_time':<20} {pb['comm_TP_time_ms']:>12.2f} {pb.get('comm_TP_ratio', 0)*100:>9.1f}%")
        print(f"{'comm_PP_time':<20} {pb['comm_PP_time_ms']:>12.2f} {pb.get('comm_PP_ratio', 0)*100:>9.1f}%")
        print(f"{'comm_EP_time':<20} {pb['comm_EP_time_ms']:>12.2f} {pb.get('comm_EP_ratio', 0)*100:>9.1f}%")
        print(f"{'idle_time':<20} {pb['idle_time_ms']:>12.2f} {pb.get('idle_ratio', 0)*100:>9.1f}%")
        print("-" * 44)
        print(f"{'total_time':<20} {pb['total_time_ms']:>12.2f} {'100.0':>9}%")
    
    # Print cumulative breakdown by parallelism type
    print(f"\nCumulative NCCL breakdown by parallelism type:")
    for cat, data in sorted(results['nccl_breakdown'].items()):
        time_ms = data['time_ms']
        calls = data['calls']
        if results['total_nccl_time_ms'] > 0:
            pct = (time_ms / results['total_nccl_time_ms']) * 100
            print(f"  {cat}: {time_ms:.2f} ms ({pct:.1f}%), {calls} calls")
        else:
            print(f"  {cat}: {time_ms:.2f} ms, {calls} calls")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        
        # Print METRIC 1 summary
        if result.get('parallelism_breakdown'):
            pb = result['parallelism_breakdown']
            print(f"\n=== METRIC 1 Summary ===")
            print(f"compute_ratio: {pb.get('compute_ratio', 0)*100:.1f}%")
            print(f"comm_DP_ratio: {pb.get('comm_DP_ratio', 0)*100:.1f}%")
            print(f"comm_TP_ratio: {pb.get('comm_TP_ratio', 0)*100:.1f}%")
            print(f"comm_PP_ratio: {pb.get('comm_PP_ratio', 0)*100:.1f}%")
            print(f"comm_EP_ratio: {pb.get('comm_EP_ratio', 0)*100:.1f}%")
            print(f"idle_ratio: {pb.get('idle_ratio', 0)*100:.1f}%")
    else:
        print("Usage: python comm_time_breakdown.py <nsys_directory>")

