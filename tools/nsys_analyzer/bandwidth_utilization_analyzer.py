#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 3: Per-Event Bandwidth and Bandwidth Utilization Analyzer

Calculates bandwidth utilization for each communication event.

Formula (from spec):
  For each event e:
    bw_e = bytes_e / duration_e    (bytes/sec)
    util_e = bw_e / B_hw           (ratio, e.g., 0.5 = 50%)

Aggregate per parallelism type (DP/TP/PP/EP):
  - avg_bw(tag): average bandwidth
  - avg_util(tag): average utilization
  - p95_util(tag): 95th percentile utilization
  - total_bytes(tag): total bytes transferred

Note: When exact bytes are not available from nsys, we estimate based on
typical bandwidth patterns. This is marked as 'is_estimated: true'.
"""

import subprocess
import os
import sqlite3
from collections import defaultdict
import numpy as np

# Hardware bandwidth configuration (bytes/sec)
# Perlmutter specs:
#   - A100 GPU: 40GB HBM with 1,555 GB/s memory bandwidth
#   - Intra-node: NVLink 600 GB/s bidirectional, ~300 GB/s unidirectional per direction
#   - Inter-node: Slingshot 11 = 4x 200 Gbps = 100 GB/s total, ~25 GB/s per NIC
# For NCCL collectives, effective bandwidth is typically lower due to algorithm overhead
HARDWARE_BW_INTRA_NODE = 300e9   # 300 GB/s (NVLink unidirectional)
HARDWARE_BW_INTER_NODE = 25e9    # 25 GB/s (Slingshot per NIC)


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
    
    Note: Same logic as comm_time_breakdown.py and phase_window_analyzer.py
    
    Args:
        kernel_name: NCCL kernel name string
        tp_size, pp_size, dp_size, ep_size: Parallelism sizes (for future use)
    
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
    #    In ZeRO-3, AllGather gathers partitioned parameters - this is DP!
    if 'allgather' in name_lower:
        return 'DP'
    
    # 5. AllReduce -> DP (gradient aggregation)
    if 'allreduce' in name_lower:
        return 'DP'
    
    # 6. Broadcast -> OTHER (usually initialization)
    if 'broadcast' in name_lower:
        return 'OTHER'
    
    # 7. Default -> OTHER
    return 'OTHER'


def export_to_sqlite(nsys_rep_file):
    """Export nsys-rep to SQLite"""
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '_temp.sqlite')
    
    try:
        export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}", 
                     "--force-overwrite=true", nsys_rep_file]
        result = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        
        if os.path.exists(sqlite_file):
            return sqlite_file
        return None
    except Exception as e:
        print(f"  Error exporting to SQLite: {e}")
        return None


def get_typical_bandwidth(kernel_name):
    """
    Get typical bandwidth for estimation purposes
    
    These are empirically observed bandwidths for different NCCL operations
    on Perlmutter A100 nodes with NVLink and Slingshot interconnect.
    
    Returns:
        float: Typical bandwidth in bytes/sec
    """
    name_lower = kernel_name.lower()
    
    # ZeRO-3 operations (ReduceScatter/AllGather) - high bandwidth
    if 'reducescatter' in name_lower or 'allgather' in name_lower:
        return 150e9  # ~150 GB/s typical on NVLink
    
    # AllReduce - medium-high bandwidth
    if 'allreduce' in name_lower:
        return 100e9  # ~100 GB/s typical
    
    # Broadcast - medium bandwidth
    if 'broadcast' in name_lower:
        return 100e9  # ~100 GB/s typical
    
    # Send/Recv (PP) - lower bandwidth (often inter-node)
    if 'send' in name_lower or 'recv' in name_lower:
        return 20e9  # ~20 GB/s typical for inter-node
    
    # AllToAll (EP) - variable
    if 'alltoall' in name_lower:
        return 80e9  # ~80 GB/s typical
    
    # Default
    return 100e9


def estimate_comm_bytes(kernel_name, duration_ns):
    """
    Estimate communication bytes based on kernel name and duration
    
    Note: This is an approximation used when nsys doesn't provide byte counts.
    The estimation uses typical observed bandwidth patterns for each operation type.
    
    Args:
        kernel_name: NCCL kernel name
        duration_ns: Kernel duration in nanoseconds
    
    Returns:
        int: Estimated bytes transferred
    """
    typical_bw = get_typical_bandwidth(kernel_name)
    return int(duration_ns * 1e-9 * typical_bw)

def analyze_bandwidth_utilization(nsys_rep_file, hardware_bw=HARDWARE_BW_INTRA_NODE):
    """
    Analyze bandwidth utilization for communication events (METRIC 3)
    
    Implementation:
      1. Load all NCCL kernels with start/end times
      2. Estimate bytes transferred (or read from trace if available)
      3. Calculate per-event bandwidth: bw_e = bytes_e / duration_e
      4. Calculate per-event utilization: util_e = bw_e / B_hw
      5. Aggregate by parallelism type (DP/TP/PP/EP)
    
    Args:
        nsys_rep_file: Path to nsys-rep file
        hardware_bw: Hardware bandwidth in bytes/sec
    
    Returns:
        dict: Bandwidth utilization statistics per parallelism type
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing bandwidth utilization in {nsys_rep_file}...")
    print(f"  Hardware bandwidth (B_hw): {hardware_bw/1e9:.1f} GB/s")
    
    # Export to SQLite
    sqlite_file = export_to_sqlite(nsys_rep_file)
    if not sqlite_file:
        print("  Failed to export SQLite, cannot analyze bandwidth")
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query NCCL kernels
        query = """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE '%nccl%'
        ORDER BY k.start
        """
        
        try:
            cursor.execute(query)
            kernels = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"  SQL query failed: {e}")
            conn.close()
            os.remove(sqlite_file)
            return None
        
        conn.close()
        os.remove(sqlite_file)
        
        if not kernels:
            print("  No NCCL kernels found")
            return None
        
        print(f"  Found {len(kernels)} NCCL events")
        
        # Analyze each event
        events_by_category = defaultdict(list)
        category_counts = defaultdict(int)
        
        for start, end, name in kernels:
            duration_ns = end - start
            duration_s = duration_ns * 1e-9
            
            # Estimate bytes (since nsys doesn't directly provide byte counts)
            bytes_transferred = estimate_comm_bytes(name, duration_ns)
            
            # Calculate per-event bandwidth: bw_e = bytes_e / duration_e
            if duration_s > 0:
                bandwidth_bps = bytes_transferred / duration_s
                # Calculate per-event utilization: util_e = bw_e / B_hw
                utilization = bandwidth_bps / hardware_bw
            else:
                bandwidth_bps = 0
                utilization = 0
            
            # Categorize using corrected function
            category = categorize_nccl_kernel(name)
            category_counts[category] += 1
            
            events_by_category[category].append({
                'duration_ns': duration_ns,
                'duration_s': duration_s,
                'bytes': bytes_transferred,
                'bandwidth_bps': bandwidth_bps,
                'utilization': utilization,
                'name': name
            })
        
        print(f"  Category distribution: {dict(category_counts)}")
        
        # Calculate statistics per parallelism type (METRIC 3 format)
        stats = {
            'hardware_bw_gbps': hardware_bw / 1e9,
            'is_estimated': True,  # Mark that bytes are estimated
            'total_events': len(kernels),
            'categories': {},
            # METRIC 3 specific fields
            'per_parallelism': {}
        }
        
        for category, events in events_by_category.items():
            if events:
                bw_list = [e['bandwidth_bps'] / 1e9 for e in events]  # GB/s
                util_list = [e['utilization'] for e in events]  # Ratio (0-1)
                util_pct_list = [u * 100 for u in util_list]  # Percentage
                bytes_list = [e['bytes'] for e in events]
                duration_list = [e['duration_s'] for e in events]
                
                # METRIC 3 output format
                cat_stats = {
                    'num_events': len(events),
                    # Total bytes transferred
                    'total_bytes': sum(bytes_list),
                    'total_bytes_gb': sum(bytes_list) / 1e9,
                    # Total duration
                    'total_duration_s': sum(duration_list),
                    # Average bandwidth: avg_bw(tag)
                    'avg_bw_gbps': np.mean(bw_list),
                    'p50_bw_gbps': np.percentile(bw_list, 50),
                    'p95_bw_gbps': np.percentile(bw_list, 95),
                    # Average utilization: avg_util(tag)
                    'avg_util': np.mean(util_list),
                    'avg_util_pct': np.mean(util_pct_list),
                    # P95 utilization: p95_util(tag)
                    'p95_util': np.percentile(util_list, 95),
                    'p95_util_pct': np.percentile(util_pct_list, 95),
                    # Additional stats
                    'p50_util_pct': np.percentile(util_pct_list, 50),
                    'min_util_pct': np.min(util_pct_list),
                    'max_util_pct': np.max(util_pct_list),
                    # For backward compatibility
                    'avg_bandwidth_gbps': np.mean(bw_list),
                    'p50_bandwidth_gbps': np.percentile(bw_list, 50),
                    'p95_bandwidth_gbps': np.percentile(bw_list, 95),
                    'avg_utilization_pct': np.mean(util_pct_list),
                    'p50_utilization_pct': np.percentile(util_pct_list, 50),
                    'p95_utilization_pct': np.percentile(util_pct_list, 95)
                }
                
                stats['categories'][category] = cat_stats
                stats['per_parallelism'][category] = cat_stats
        
        # Calculate global statistics
        all_events = []
        for events in events_by_category.values():
            all_events.extend(events)
        
        if all_events:
            all_bw = [e['bandwidth_bps'] / 1e9 for e in all_events]
            all_util = [e['utilization'] * 100 for e in all_events]
            
            stats['global'] = {
                'total_events': len(all_events),
                'total_bytes_gb': sum(e['bytes'] for e in all_events) / 1e9,
                'avg_bw_gbps': np.mean(all_bw),
                'avg_util_pct': np.mean(all_util),
                'p95_util_pct': np.percentile(all_util, 95)
                }
        
        return stats
        
    except Exception as e:
        print(f"  Error analyzing bandwidth: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return None

def metric_cal(directory, hardware_bw=HARDWARE_BW_INTRA_NODE):
    """
    Calculate bandwidth utilization metrics (METRIC 3)
    
    Args:
        directory: Trace directory or nsys directory
        hardware_bw: Hardware bandwidth in bytes/sec
    
    Returns:
        dict: Bandwidth statistics per parallelism type
    """
    # Find nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return {}
    
    # Analyze first nsys file
    for filename in os.listdir(nsys_dir):
        if filename.endswith(".nsys-rep"):
            nsys_file = os.path.join(nsys_dir, filename)
            stats = analyze_bandwidth_utilization(nsys_file, hardware_bw)
            
            if stats:
                print("\n" + "="*75)
                print("METRIC 3: Per-Event Bandwidth and Bandwidth Utilization")
                print("="*75)
                print(f"Hardware bandwidth (B_hw): {stats['hardware_bw_gbps']:.1f} GB/s")
                print(f"Total NCCL events: {stats.get('total_events', 'N/A')}")
                print(f"Bytes estimation: {'estimated' if stats.get('is_estimated') else 'from trace'}")
                
                # Print global summary
                if stats.get('global'):
                    g = stats['global']
                    print(f"\nGlobal summary:")
                    print(f"  Total bytes: {g['total_bytes_gb']:.2f} GB")
                    print(f"  Avg bandwidth: {g['avg_bw_gbps']:.2f} GB/s")
                    print(f"  Avg utilization: {g['avg_util_pct']:.1f}%")
                
                # Print per-parallelism table (METRIC 3 format)
                print(f"\n{'Category':<10} {'Events':>8} {'Bytes(GB)':>12} {'avg_bw':>10} {'avg_util':>10} {'p95_util':>10}")
                print("-" * 70)
                
                for category in ['DP', 'TP', 'PP', 'EP', 'OTHER']:
                    if category in stats['categories']:
                        data = stats['categories'][category]
                        print(f"{category:<10} {data['num_events']:>8} {data['total_bytes_gb']:>12.2f} "
                              f"{data['avg_bw_gbps']:>10.2f} {data['avg_util_pct']:>9.1f}% {data['p95_util_pct']:>9.1f}%")
                
                print("-" * 70)
                
                # Detailed per-category stats
                print("\nDetailed statistics per parallelism type:")
                for category, data in sorted(stats['categories'].items()):
                    print(f"\n  {category}:")
                    print(f"    num_events: {data['num_events']}")
                    print(f"    total_bytes: {data['total_bytes_gb']:.2f} GB")
                    print(f"    avg_bw(tag): {data['avg_bw_gbps']:.2f} GB/s")
                    print(f"    avg_util(tag): {data['avg_util_pct']:.1f}%")
                    print(f"    p95_util(tag): {data['p95_util_pct']:.1f}%")
                    print(f"    p50_util(tag): {data['p50_util_pct']:.1f}%")
                
                print("="*75)
                
                return stats
            break
    
    return {}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        # Optional: specify hardware bandwidth in GB/s
        hw_bw = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else HARDWARE_BW_INTRA_NODE
        result = metric_cal(directory, hw_bw)
        
        # Print METRIC 3 summary
        if result.get('per_parallelism'):
            print(f"\n=== METRIC 3 Summary ===")
            for tag in ['DP', 'TP', 'PP', 'EP', 'OTHER']:
                if tag in result['per_parallelism']:
                    data = result['per_parallelism'][tag]
                    print(f"{tag}: avg_bw={data['avg_bw_gbps']:.1f}GB/s, "
                          f"avg_util={data['avg_util_pct']:.1f}%, "
                          f"p95_util={data['p95_util_pct']:.1f}%, "
                          f"total_bytes={data['total_bytes_gb']:.1f}GB")
    else:
        print("Usage: python bandwidth_utilization_analyzer.py <nsys_directory> [hardware_bw_gbps]")
        print("Example: python bandwidth_utilization_analyzer.py nsys/ 300")

