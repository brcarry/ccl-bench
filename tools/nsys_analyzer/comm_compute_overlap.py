#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Communication-Computation Overlap Analyzer
Measures the degree of overlap between communication and computation operations
"""

import subprocess
import os
import re

def analyze_overlap(nsys_rep_file):
    """
    Analyze communication-computation overlap from nsys-rep file
    
    Uses CUDA API and kernel timeline to detect overlap
    
    Returns:
        dict: Overlap statistics
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing comm-compute overlap in {nsys_rep_file}...")
    
    # Export to SQLite for detailed timeline analysis
    sqlite_file = nsys_rep_file.replace('.nsys-rep', '_temp.sqlite')
    
    try:
        # Export to SQLite
        export_cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_file}", 
                     "--force-overwrite=true", nsys_rep_file]
        result = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        
        if not os.path.exists(sqlite_file):
            print("  Failed to export SQLite, using stats-based estimation")
            return estimate_overlap_from_stats(nsys_rep_file)
        
        # Query SQLite for kernel timeline
        import sqlite3
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Get all CUDA kernels with their start and end times
        # shortName is an INTEGER ID referencing StringIds table
        query = """
        SELECT k.start, k.end, s.value as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """
        
        try:
            cursor.execute(query)
            kernels = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # Table might not exist, fall back to estimation
            print(f"  SQL query failed: {e}, using estimation")
            conn.close()
            os.remove(sqlite_file)
            return estimate_overlap_from_stats(nsys_rep_file)
        
        conn.close()
        os.remove(sqlite_file)
        
        if not kernels:
            print("  No kernel data found")
            return None
        
        # Categorize kernels and find overlaps
        nccl_intervals = []
        compute_intervals = []
        
        for start, end, name in kernels:
            # Handle case where name might be None or not a string
            name_str = str(name) if name else ""
            if 'nccl' in name_str.lower():
                nccl_intervals.append((start, end))
            else:
                compute_intervals.append((start, end))
        
        # Calculate overlap
        total_nccl_time = sum(end - start for start, end in nccl_intervals)
        total_compute_time = sum(end - start for start, end in compute_intervals)
        
        # Find overlapping time
        overlap_time = 0
        for nccl_start, nccl_end in nccl_intervals:
            for comp_start, comp_end in compute_intervals:
                # Calculate intersection
                overlap_start = max(nccl_start, comp_start)
                overlap_end = min(nccl_end, comp_end)
                if overlap_start < overlap_end:
                    overlap_time += (overlap_end - overlap_start)
        
        stats = {
            "total_nccl_time_ns": total_nccl_time,
            "total_compute_time_ns": total_compute_time,
            "overlap_time_ns": overlap_time,
            "nccl_intervals": len(nccl_intervals),
            "compute_intervals": len(compute_intervals)
        }
        
        return stats
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {nsys_rep_file}")
        return None
    except Exception as e:
        print(f"  Error during overlap analysis: {e}")
        # Clean up temp file if it exists
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return estimate_overlap_from_stats(nsys_rep_file)

def estimate_overlap_from_stats(nsys_rep_file):
    """
    Estimate overlap using statistical approach when timeline data unavailable
    
    This is a conservative estimation based on kernel statistics
    """
    print("  Using statistical estimation for overlap")
    
    # Get kernel statistics
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", nsys_rep_file]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        output = result.stdout.decode('utf-8')
        
        total_nccl_time = 0
        total_compute_time = 0
        
        lines = output.split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                total_time_str = parts[1].replace(',', '')
                if not total_time_str.replace('.', '').isdigit():
                    continue
                
                total_time_ns = float(total_time_str)
                kernel_name = ' '.join(parts[8:])
                
                if 'nccl' in kernel_name.lower():
                    total_nccl_time += total_time_ns
                else:
                    total_compute_time += total_time_ns
            except (ValueError, IndexError):
                continue
        
        # Conservative estimation: assume minimal overlap
        # In practice, DeepSpeed ZeRO-3 has some overlap due to async operations
        # Estimate ~20-30% overlap based on typical patterns
        estimated_overlap = min(total_nccl_time, total_compute_time) * 0.25
        
        stats = {
            "total_nccl_time_ns": total_nccl_time,
            "total_compute_time_ns": total_compute_time,
            "overlap_time_ns": estimated_overlap,
            "is_estimated": True
        }
        
        return stats
        
    except Exception as e:
        print(f"  Error in estimation: {e}")
        return None

def metric_cal(directory):
    """
    Calculate communication-computation overlap percentage
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        float: Overlap percentage
    """
    # Find nsys directory
    if os.path.exists(os.path.join(directory, "..", "..", "nsys")):
        nsys_dir = os.path.join(directory, "..", "..", "nsys")
    elif any(f.endswith(".nsys-rep") for f in os.listdir(directory)):
        nsys_dir = directory
    else:
        print("No nsys-rep files found")
        return 0.0
    
    # Analyze first nsys file (overlap is per-GPU metric)
    for filename in os.listdir(nsys_dir):
        if filename.endswith(".nsys-rep"):
            nsys_file = os.path.join(nsys_dir, filename)
            stats = analyze_overlap(nsys_file)
            
            if stats:
                print("\n" + "="*60)
                print("Communication-Computation Overlap Analysis")
                print("="*60)
                
                nccl_time_ms = stats["total_nccl_time_ns"] / 1e6
                compute_time_ms = stats["total_compute_time_ns"] / 1e6
                overlap_time_ms = stats["overlap_time_ns"] / 1e6
                
                print(f"Total NCCL time: {nccl_time_ms:.2f} ms")
                print(f"Total compute time: {compute_time_ms:.2f} ms")
                print(f"Overlap time: {overlap_time_ms:.2f} ms")
                
                if stats["total_nccl_time_ns"] > 0:
                    overlap_pct = (stats["overlap_time_ns"] / stats["total_nccl_time_ns"]) * 100
                    print(f"\nOverlap percentage (of comm time): {overlap_pct:.2f}%")
                else:
                    overlap_pct = 0.0
                    print("\nNo communication time detected")
                
                if stats.get("is_estimated"):
                    print("\nNote: Overlap is estimated (conservative)")
                
                print("="*60)
                
                return overlap_pct
            break
    
    return 0.0

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
        print(f"\nComm-Compute overlap: {result:.2f}%")
    else:
        print("Usage: python comm_compute_overlap.py <nsys_directory>")

