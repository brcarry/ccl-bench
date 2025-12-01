#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 3: Per-Event Bandwidth Utilization Analyzer
Calculates bandwidth utilization for each communication event
"""

import subprocess
import os
import sqlite3
from collections import defaultdict
import numpy as np

# Hardware bandwidth configuration (bytes/sec)
# Based on GPU vs CPU Memory Bandwidth comparison:
# - A100 GPU: 40GB HBM with 1,555 GB/s memory bandwidth
# - For NCCL collective operations, effective bandwidth depends on:
#   * Intra-node: NVLink ~600 GB/s bidirectional, ~300 GB/s unidirectional
#   * Inter-node: Slingshot 11 = 200 Gbps = 25 GB/s
# Using NVLink bandwidth for intra-node communication
HARDWARE_BW_INTRA_NODE = 300e9   # 300 GB/s (NVLink unidirectional)
HARDWARE_BW_INTER_NODE = 25e9    # 25 GB/s (Slingshot 200Gbps)

def categorize_comm_event(kernel_name):
    """Categorize communication event by parallelism type"""
    name_lower = kernel_name.lower()
    
    if 'allreduce' in name_lower:
        return 'DP'
    elif 'allgather' in name_lower:
        return 'TP'
    elif 'reducescatter' in name_lower:
        return 'DP'
    elif 'send' in name_lower or 'recv' in name_lower:
        return 'PP'
    elif 'alltoall' in name_lower:
        return 'EP'
    else:
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

def estimate_comm_bytes(kernel_name, duration_ns):
    """
    Estimate communication bytes based on kernel name and duration
    This is a rough estimation when exact bytes are not available
    """
    # Typical message sizes for different operations (rough estimates)
    # For Llama-3.1-8B with bf16: ~8B params * 2 bytes = 16GB total
    
    name_lower = kernel_name.lower()
    
    # AllReduce: typically gradient size
    if 'allreduce' in name_lower:
        # Estimate based on typical gradient chunk size
        estimated_bw = 100e9  # 100 GB/s typical
        return int(duration_ns * 1e-9 * estimated_bw)
    
    # ReduceScatter/AllGather: ZeRO-3 operations
    elif 'reducescatter' in name_lower or 'allgather' in name_lower:
        estimated_bw = 150e9  # 150 GB/s typical
        return int(duration_ns * 1e-9 * estimated_bw)
    
    # Send/Recv: pipeline chunks
    elif 'send' in name_lower or 'recv' in name_lower:
        estimated_bw = 50e9  # 50 GB/s typical
        return int(duration_ns * 1e-9 * estimated_bw)
    
    else:
        # Default estimation
        estimated_bw = 100e9
        return int(duration_ns * 1e-9 * estimated_bw)

def analyze_bandwidth_utilization(nsys_rep_file, hardware_bw=HARDWARE_BW_INTRA_NODE):
    """
    Analyze bandwidth utilization for communication events
    
    Args:
        nsys_rep_file: Path to nsys-rep file
        hardware_bw: Hardware bandwidth in bytes/sec
    
    Returns:
        dict: Bandwidth utilization statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing bandwidth utilization in {nsys_rep_file}...")
    print(f"  Hardware bandwidth: {hardware_bw/1e9:.1f} GB/s")
    
    # Export to SQLite
    sqlite_file = export_to_sqlite(nsys_rep_file)
    if not sqlite_file:
        print("  Failed to export SQLite, cannot analyze bandwidth")
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query NCCL kernels
        # shortName is an INTEGER ID referencing StringIds table
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
        
        # Analyze each event
        events_by_category = defaultdict(list)
        
        for start, end, name in kernels:
            duration_ns = end - start
            duration_s = duration_ns * 1e-9
            
            # Estimate bytes (since nsys doesn't directly provide this)
            bytes_transferred = estimate_comm_bytes(name, duration_ns)
            
            # Calculate bandwidth
            if duration_s > 0:
                bandwidth_bps = bytes_transferred / duration_s
                utilization = bandwidth_bps / hardware_bw
            else:
                bandwidth_bps = 0
                utilization = 0
            
            # Categorize
            category = categorize_comm_event(name)
            
            events_by_category[category].append({
                'duration_ns': duration_ns,
                'bytes': bytes_transferred,
                'bandwidth_bps': bandwidth_bps,
                'utilization': utilization
            })
        
        # Calculate statistics per category
        stats = {
            'hardware_bw_gbps': hardware_bw / 1e9,
            'categories': {}
        }
        
        for category, events in events_by_category.items():
            if events:
                bw_list = [e['bandwidth_bps'] / 1e9 for e in events]  # GB/s
                util_list = [e['utilization'] * 100 for e in events]  # Percentage
                bytes_list = [e['bytes'] for e in events]
                
                stats['categories'][category] = {
                    'num_events': len(events),
                    'total_bytes': sum(bytes_list),
                    'avg_bandwidth_gbps': np.mean(bw_list),
                    'p50_bandwidth_gbps': np.percentile(bw_list, 50),
                    'p95_bandwidth_gbps': np.percentile(bw_list, 95),
                    'avg_utilization_pct': np.mean(util_list),
                    'p50_utilization_pct': np.percentile(util_list, 50),
                    'p95_utilization_pct': np.percentile(util_list, 95)
                }
        
        return stats
        
    except Exception as e:
        print(f"  Error analyzing bandwidth: {e}")
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return None

def metric_cal(directory, hardware_bw=HARDWARE_BW_INTRA_NODE):
    """
    Calculate bandwidth utilization metrics
    
    Args:
        directory: Trace directory or nsys directory
        hardware_bw: Hardware bandwidth in bytes/sec
    
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
            stats = analyze_bandwidth_utilization(nsys_file, hardware_bw)
            
            if stats:
                print("\n" + "="*60)
                print("Bandwidth Utilization Analysis Results")
                print("="*60)
                print(f"Hardware bandwidth: {stats['hardware_bw_gbps']:.1f} GB/s")
                print("\nPer-category statistics:")
                
                for category, data in sorted(stats['categories'].items()):
                    print(f"\n  {category}:")
                    print(f"    Events: {data['num_events']}")
                    print(f"    Total bytes: {data['total_bytes']/1e9:.2f} GB")
                    print(f"    Avg bandwidth: {data['avg_bandwidth_gbps']:.2f} GB/s")
                    print(f"    P50 bandwidth: {data['p50_bandwidth_gbps']:.2f} GB/s")
                    print(f"    P95 bandwidth: {data['p95_bandwidth_gbps']:.2f} GB/s")
                    print(f"    Avg utilization: {data['avg_utilization_pct']:.1f}%")
                    print(f"    P50 utilization: {data['p50_utilization_pct']:.1f}%")
                    print(f"    P95 utilization: {data['p95_utilization_pct']:.1f}%")
                
                print("="*60)
                
                return stats
            break
    
    return {}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        # Optional: specify hardware bandwidth
        hw_bw = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else HARDWARE_BW_INTRA_NODE
        result = metric_cal(directory, hw_bw)
    else:
        print("Usage: python bandwidth_utilization_analyzer.py <nsys_directory> [hardware_bw_gbps]")
        print("Example: python bandwidth_utilization_analyzer.py nsys/ 300")

