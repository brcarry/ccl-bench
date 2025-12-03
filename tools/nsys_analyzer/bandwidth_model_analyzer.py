#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 6: Bandwidth Utilization with Ring Model

Based on the theoretical Ring AllReduce time model:
    T_theory = α(n-1) + s(n-1)/(nB)

Where:
    n: Number of processes in collective (DP/TP group size)
    s: Message size (bytes)
    B: Link bandwidth (bytes/second)
    α: Per-hop latency (seconds)
    T: Collective operation time (seconds)

Three bandwidth calculation methods:
    1. B_eff_simple = s / T_measured
    2. B_eff_ring = s(n-1) / [n · (T_measured - α(n-1))]
    3. B_eff_ring_approx = s(n-1) / (n · T_measured)  # ignore α

Utilization:
    utilization = B_eff / B_hardware
"""

import subprocess
import os
import sqlite3
from collections import defaultdict
import numpy as np


# ============================================================================
# Hardware Configuration (Perlmutter A100 nodes)
# ============================================================================
# Measured from: nvidia-smi nvlink -s, nvidia-smi topo -m, ip link show
#
# NVLink (Intra-node, GPU-to-GPU):
#   - Each GPU has 12 NVLink ports, each 25 GB/s
#   - GPU pairs connected via NV4 (4 NVLinks bonded)
#   - Per GPU-pair bandwidth: 4 × 25 GB/s = 100 GB/s unidirectional
#   - Full-duplex: 200 GB/s bidirectional per GPU pair
#
# Slingshot (Inter-node):
#   - 4 HSN NICs per node: hsn0, hsn1, hsn2, hsn3
#   - Each NIC: 200 Gbps = 25 GB/s
#   - Total node bandwidth: 4 × 25 GB/s = 100 GB/s
#   - NCCL typically uses 1-4 NICs depending on configuration
#
HARDWARE_BW_NVLINK = 100e9      # 100 GB/s (NV4 = 4 NVLinks × 25 GB/s, unidirectional)
HARDWARE_BW_SLINGSHOT = 25e9    # 25 GB/s per NIC (Slingshot 200 Gbps)

# Per-hop latency estimation (microseconds -> seconds)
# These are typical values; actual latency depends on message size and congestion
ALPHA_INTRA_NODE = 1e-6         # ~1 μs for NVLink
ALPHA_INTER_NODE = 5e-6         # ~5 μs for Slingshot


# ============================================================================
# Categorization (consistent with other analyzers)
# ============================================================================
def categorize_nccl_kernel(kernel_name):
    """
    Categorize NCCL kernel by parallelism type (DP/TP/PP/EP)
    
    For ZeRO-3:
      - ReduceScatter: DP
      - AllGather: DP
      - AllReduce: DP (or TP if tp_size > 1)
      - Send/Recv: PP
      - AllToAll: EP
      - Broadcast: OTHER
    """
    name_lower = kernel_name.lower()
    
    if 'send' in name_lower or 'recv' in name_lower:
        return 'PP'
    if 'alltoall' in name_lower:
        return 'EP'
    if 'reducescatter' in name_lower:
        return 'DP'
    if 'allgather' in name_lower:
        return 'DP'
    if 'allreduce' in name_lower:
        return 'DP'
    if 'broadcast' in name_lower:
        return 'OTHER'
    return 'OTHER'


def get_collective_type(kernel_name):
    """Get the collective operation type from kernel name"""
    name_lower = kernel_name.lower()
    
    if 'allreduce' in name_lower:
        return 'AllReduce'
    elif 'reducescatter' in name_lower:
        return 'ReduceScatter'
    elif 'allgather' in name_lower:
        return 'AllGather'
    elif 'broadcast' in name_lower:
        return 'Broadcast'
    elif 'alltoall' in name_lower:
        return 'AllToAll'
    elif 'send' in name_lower:
        return 'Send'
    elif 'recv' in name_lower:
        return 'Recv'
    else:
        return 'Other'


# ============================================================================
# Message Size Estimation
# ============================================================================
def estimate_message_size(kernel_name, duration_ns, hardware_bw=100e9):
    """
    Estimate message size from kernel duration
    
    When nsys doesn't provide byte counts, we use:
        s_estimated = T_measured * B_typical
    
    IMPORTANT: The typical bandwidth used for estimation should be based on
    observed utilization ratios, NOT exceed hardware bandwidth.
    
    Typical utilization on Perlmutter A100 (NVLink NV4 = 100 GB/s):
        - AllGather/ReduceScatter: ~70-80% utilization → ~70-80 GB/s
        - AllReduce: ~60-70% utilization → ~60-70 GB/s
        - Broadcast: ~50-60% utilization → ~50-60 GB/s
        - P2P (inter-node): ~80% of Slingshot → ~20 GB/s
    
    Args:
        kernel_name: NCCL kernel name
        duration_ns: Kernel duration in nanoseconds
        hardware_bw: Hardware bandwidth for reference (bytes/sec)
    
    Returns:
        int: Estimated message size in bytes
    """
    name_lower = kernel_name.lower()
    
    # Estimate typical utilization ratio for each operation type
    # These are conservative estimates based on typical NCCL performance
    if 'reducescatter' in name_lower or 'allgather' in name_lower:
        # ZeRO-3 operations typically achieve good bandwidth
        utilization = 0.75  # ~75% of hardware BW
    elif 'allreduce' in name_lower:
        utilization = 0.65  # ~65% of hardware BW
    elif 'broadcast' in name_lower:
        utilization = 0.55  # ~55% of hardware BW
    elif 'send' in name_lower or 'recv' in name_lower:
        # P2P typically goes through Slingshot (inter-node)
        # Use Slingshot bandwidth instead
        utilization = 0.80
        hardware_bw = 25e9  # Slingshot per NIC
    elif 'alltoall' in name_lower:
        utilization = 0.60  # ~60% of hardware BW
    else:
        utilization = 0.50  # Conservative default
    
    # Estimate bandwidth = hardware_bw × utilization
    estimated_bw = hardware_bw * utilization
    
    duration_s = duration_ns * 1e-9
    return int(duration_s * estimated_bw)


# ============================================================================
# Bandwidth Calculation Functions
# ============================================================================
def calc_bandwidth_simple(s, T):
    """
    Simple bandwidth: B_eff = s / T
    
    Args:
        s: Message size (bytes)
        T: Measured time (seconds)
    
    Returns:
        float: Effective bandwidth (bytes/second)
    """
    if T > 0:
        return s / T
    return 0.0


def calc_bandwidth_ring(s, T, n, alpha=0.0):
    """
    Ring model bandwidth: B_eff = s(n-1) / [n · (T - α(n-1))]
    
    Args:
        s: Message size (bytes)
        T: Measured time (seconds)
        n: Number of processes in collective
        alpha: Per-hop latency (seconds)
    
    Returns:
        float: Effective bandwidth based on ring model (bytes/second)
    """
    if n <= 1:
        return calc_bandwidth_simple(s, T)
    
    latency_term = alpha * (n - 1)
    effective_time = T - latency_term
    
    if effective_time > 0:
        return s * (n - 1) / (n * effective_time)
    return 0.0


def calc_bandwidth_ring_approx(s, T, n):
    """
    Ring model bandwidth (ignore α): B_eff ≈ s(n-1) / (n · T)
    
    Args:
        s: Message size (bytes)
        T: Measured time (seconds)
        n: Number of processes in collective
    
    Returns:
        float: Effective bandwidth (bytes/second)
    """
    if n <= 1 or T <= 0:
        return calc_bandwidth_simple(s, T)
    
    return s * (n - 1) / (n * T)


def calc_utilization(B_eff, B_hardware):
    """
    Calculate bandwidth utilization
    
    Args:
        B_eff: Effective bandwidth (bytes/second)
        B_hardware: Hardware bandwidth (bytes/second)
    
    Returns:
        float: Utilization ratio (0.0 - 1.0+)
    """
    if B_hardware > 0:
        return B_eff / B_hardware
    return 0.0


# ============================================================================
# Main Analysis Function
# ============================================================================
def analyze_bandwidth_with_model(nsys_rep_file, 
                                  dp_size=1, tp_size=1, pp_size=1, ep_size=1,
                                  hardware_bw=HARDWARE_BW_NVLINK,
                                  alpha=ALPHA_INTRA_NODE):
    """
    Analyze bandwidth utilization using ring model
    
    Args:
        nsys_rep_file: Path to nsys-rep file
        dp_size: Data parallel size (for DP collectives)
        tp_size: Tensor parallel size (for TP collectives)
        pp_size: Pipeline parallel size (for PP communication)
        ep_size: Expert parallel size (for EP collectives)
        hardware_bw: Hardware bandwidth (bytes/second)
        alpha: Per-hop latency (seconds)
    
    Returns:
        dict: Bandwidth utilization statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing bandwidth with ring model in {nsys_rep_file}...")
    print(f"  Hardware bandwidth (B_hw): {hardware_bw/1e9:.1f} GB/s")
    print(f"  Per-hop latency (α): {alpha*1e6:.2f} μs")
    print(f"  Parallelism config: DP={dp_size}, TP={tp_size}, PP={pp_size}, EP={ep_size}")
    
    # Check for SQLite file
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '.sqlite')
    temp_sqlite = False
    
    if not os.path.exists(sqlite_file):
        sqlite_file = nsys_rep_file.replace('.nsys-rep', '_temp.sqlite')
        temp_sqlite = True
        
        try:
            export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}", 
                         "--force-overwrite=true", nsys_rep_file]
            subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        except:
            print("  Failed to export SQLite")
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
        
        cursor.execute(query)
        kernels = cursor.fetchall()
        conn.close()
        
        if temp_sqlite and os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        
        if not kernels:
            print("  No NCCL kernels found")
            return None
        
        print(f"  Found {len(kernels)} NCCL events")
        
        # Analyze each event
        events_by_category = defaultdict(list)
        events_by_collective = defaultdict(list)
        
        for start, end, name in kernels:
            duration_ns = end - start
            duration_s = duration_ns * 1e-9
            
            # Get category and collective type
            category = categorize_nccl_kernel(name)
            collective_type = get_collective_type(name)
            
            # Determine n based on category
            if category == 'DP':
                n = dp_size
            elif category == 'TP':
                n = tp_size
            elif category == 'PP':
                n = 2  # P2P is between 2 processes
            elif category == 'EP':
                n = ep_size
            else:
                n = max(dp_size, tp_size, 1)  # Default
            
            # Estimate message size (using hardware bandwidth as reference)
            s = estimate_message_size(name, duration_ns, hardware_bw)
            
            # Calculate bandwidths using three methods
            bw_simple = calc_bandwidth_simple(s, duration_s)
            bw_ring = calc_bandwidth_ring(s, duration_s, n, alpha)
            bw_ring_approx = calc_bandwidth_ring_approx(s, duration_s, n)
            
            # Calculate utilizations
            util_simple = calc_utilization(bw_simple, hardware_bw)
            util_ring = calc_utilization(bw_ring, hardware_bw)
            util_ring_approx = calc_utilization(bw_ring_approx, hardware_bw)
            
            event_data = {
                'duration_ns': duration_ns,
                'duration_s': duration_s,
                'message_size': s,
                'n': n,
                'collective_type': collective_type,
                # Simple method
                'bw_simple': bw_simple,
                'util_simple': util_simple,
                # Ring model
                'bw_ring': bw_ring,
                'util_ring': util_ring,
                # Ring approx
                'bw_ring_approx': bw_ring_approx,
                'util_ring_approx': util_ring_approx
            }
            
            events_by_category[category].append(event_data)
            events_by_collective[collective_type].append(event_data)
        
        # Aggregate statistics
        stats = {
            'hardware_bw_gbps': hardware_bw / 1e9,
            'alpha_us': alpha * 1e6,
            'config': {
                'dp_size': dp_size,
                'tp_size': tp_size,
                'pp_size': pp_size,
                'ep_size': ep_size
            },
            'per_parallelism': {},
            'per_collective': {},
            'global': {}
        }
        
        # Per-parallelism stats
        for category, events in events_by_category.items():
            if events:
                stats['per_parallelism'][category] = calc_category_stats(events, category)
        
        # Per-collective stats
        for coll_type, events in events_by_collective.items():
            if events:
                stats['per_collective'][coll_type] = calc_category_stats(events, coll_type)
        
        # Global stats
        all_events = []
        for events in events_by_category.values():
            all_events.extend(events)
        
        if all_events:
            stats['global'] = calc_category_stats(all_events, 'global')
            stats['global']['total_events'] = len(all_events)
        
        return stats
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        if temp_sqlite and os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return None


def calc_category_stats(events, category_name):
    """Calculate aggregated statistics for a category of events"""
    
    # Extract lists
    s_list = [e['message_size'] for e in events]
    bw_simple_list = [e['bw_simple'] / 1e9 for e in events]  # GB/s
    bw_ring_list = [e['bw_ring'] / 1e9 for e in events]
    bw_ring_approx_list = [e['bw_ring_approx'] / 1e9 for e in events]
    util_simple_list = [e['util_simple'] * 100 for e in events]  # %
    util_ring_list = [e['util_ring'] * 100 for e in events]
    util_ring_approx_list = [e['util_ring_approx'] * 100 for e in events]
    
    return {
        'num_events': len(events),
        'total_bytes_gb': sum(s_list) / 1e9,
        # Simple method (B_eff = s/T)
        'simple': {
            'avg_bw_gbps': np.mean(bw_simple_list),
            'p50_bw_gbps': np.percentile(bw_simple_list, 50),
            'p95_bw_gbps': np.percentile(bw_simple_list, 95),
            'avg_util_pct': np.mean(util_simple_list),
            'p50_util_pct': np.percentile(util_simple_list, 50),
            'p95_util_pct': np.percentile(util_simple_list, 95)
        },
        # Ring model (B_eff = s(n-1)/[n(T-α(n-1))])
        'ring_model': {
            'avg_bw_gbps': np.mean(bw_ring_list),
            'p50_bw_gbps': np.percentile(bw_ring_list, 50),
            'p95_bw_gbps': np.percentile(bw_ring_list, 95),
            'avg_util_pct': np.mean(util_ring_list),
            'p50_util_pct': np.percentile(util_ring_list, 50),
            'p95_util_pct': np.percentile(util_ring_list, 95)
        },
        # Ring approx (B_eff ≈ s(n-1)/(n·T))
        'ring_approx': {
            'avg_bw_gbps': np.mean(bw_ring_approx_list),
            'p50_bw_gbps': np.percentile(bw_ring_approx_list, 50),
            'p95_bw_gbps': np.percentile(bw_ring_approx_list, 95),
            'avg_util_pct': np.mean(util_ring_approx_list),
            'p50_util_pct': np.percentile(util_ring_approx_list, 50),
            'p95_util_pct': np.percentile(util_ring_approx_list, 95)
        }
    }


# ============================================================================
# CLI Entry Point
# ============================================================================
def metric_cal(directory, dp_size=8, tp_size=1, pp_size=1, ep_size=1,
               hardware_bw=HARDWARE_BW_NVLINK, alpha=ALPHA_INTRA_NODE):
    """
    Calculate bandwidth utilization with ring model (METRIC 6)
    
    Args:
        directory: Trace directory or nsys directory
        dp_size, tp_size, pp_size, ep_size: Parallelism configuration
        hardware_bw: Hardware bandwidth (bytes/second)
        alpha: Per-hop latency (seconds)
    
    Returns:
        dict: Bandwidth statistics
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
            stats = analyze_bandwidth_with_model(
                nsys_file, dp_size, tp_size, pp_size, ep_size,
                hardware_bw, alpha
            )
            
            if stats:
                print_results(stats)
                return stats
            break
    
    return {}


def print_results(stats):
    """Print METRIC 6 results"""
    print("\n" + "="*80)
    print("METRIC 6: Bandwidth Utilization with Ring Model")
    print("="*80)
    print(f"Hardware bandwidth (B_hw): {stats['hardware_bw_gbps']:.1f} GB/s")
    print(f"Per-hop latency (α): {stats['alpha_us']:.2f} μs")
    print(f"Parallelism config: DP={stats['config']['dp_size']}, TP={stats['config']['tp_size']}, "
          f"PP={stats['config']['pp_size']}, EP={stats['config']['ep_size']}")
    
    print(f"\n{'='*80}")
    print("Bandwidth Calculation Methods:")
    print("  1. Simple:     B_eff = s / T")
    print("  2. Ring:       B_eff = s(n-1) / [n · (T - α(n-1))]")
    print("  3. Ring Approx: B_eff ≈ s(n-1) / (n · T)")
    print(f"{'='*80}")
    
    # Per-parallelism results
    print("\nPer-Parallelism Type:")
    print(f"{'Category':<10} {'Events':>8} {'Bytes(GB)':>10} "
          f"{'Simple':>12} {'Ring':>12} {'RingApprox':>12}")
    print(f"{'':10} {'':>8} {'':>10} "
          f"{'avg_util%':>12} {'avg_util%':>12} {'avg_util%':>12}")
    print("-" * 75)
    
    for cat in ['DP', 'TP', 'PP', 'EP', 'OTHER']:
        if cat in stats['per_parallelism']:
            data = stats['per_parallelism'][cat]
            print(f"{cat:<10} {data['num_events']:>8} {data['total_bytes_gb']:>10.2f} "
                  f"{data['simple']['avg_util_pct']:>11.1f}% "
                  f"{data['ring_model']['avg_util_pct']:>11.1f}% "
                  f"{data['ring_approx']['avg_util_pct']:>11.1f}%")
    
    # Global summary
    if stats.get('global'):
        g = stats['global']
        print("-" * 75)
        print(f"{'TOTAL':<10} {g['total_events']:>8} {g['total_bytes_gb']:>10.2f} "
              f"{g['simple']['avg_util_pct']:>11.1f}% "
              f"{g['ring_model']['avg_util_pct']:>11.1f}% "
              f"{g['ring_approx']['avg_util_pct']:>11.1f}%")
    
    # Detailed per-collective results
    print(f"\n{'='*80}")
    print("Per-Collective Type (using Ring Approx):")
    print(f"{'Collective':<15} {'Events':>8} {'Bytes(GB)':>10} {'avg_bw':>10} {'avg_util':>10}")
    print("-" * 55)
    
    for coll in ['AllReduce', 'ReduceScatter', 'AllGather', 'Broadcast', 'AllToAll', 'Send', 'Recv', 'Other']:
        if coll in stats['per_collective']:
            data = stats['per_collective'][coll]
            ra = data['ring_approx']
            print(f"{coll:<15} {data['num_events']:>8} {data['total_bytes_gb']:>10.2f} "
                  f"{ra['avg_bw_gbps']:>9.1f} {ra['avg_util_pct']:>9.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        
        # Parse optional arguments
        dp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        tp_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        pp_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        ep_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        
        result = metric_cal(directory, dp_size, tp_size, pp_size, ep_size)
        
        if result and result.get('global'):
            print(f"\n=== METRIC 6 Summary ===")
            g = result['global']
            print(f"Simple method:      avg_util = {g['simple']['avg_util_pct']:.1f}%")
            print(f"Ring model:         avg_util = {g['ring_model']['avg_util_pct']:.1f}%")
            print(f"Ring approx:        avg_util = {g['ring_approx']['avg_util_pct']:.1f}%")
    else:
        print("Usage: python bandwidth_model_analyzer.py <nsys_directory> [dp_size] [tp_size] [pp_size] [ep_size]")
        print("Example: python bandwidth_model_analyzer.py nsys/ 8 1 1 1")
        print("\nFormulas:")
        print("  Simple:      B_eff = s / T")
        print("  Ring:        B_eff = s(n-1) / [n · (T - α(n-1))]")
        print("  Ring Approx: B_eff ≈ s(n-1) / (n · T)")
        print("  Utilization: util = B_eff / B_hardware")

