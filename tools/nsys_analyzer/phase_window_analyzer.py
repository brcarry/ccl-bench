#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
METRIC 2: Parallelism-Phase Window Time Analyzer
Measures time gaps between different parallelism phases (TP->DP, DP->PP, etc.)
"""

import subprocess
import os
import sqlite3
from collections import defaultdict
import numpy as np

def export_to_sqlite(nsys_rep_file):
    """Export nsys-rep to SQLite for detailed timeline analysis"""
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

def categorize_comm_event(kernel_name):
    """
    Categorize communication event by parallelism type
    Based on NCCL kernel name patterns
    """
    name_lower = kernel_name.lower()
    
    # DP: typically AllReduce for gradient synchronization
    if 'allreduce' in name_lower:
        return 'DP'
    
    # TP: AllReduce/AllGather for tensor parallelism
    elif 'allgather' in name_lower:
        return 'TP'
    
    # ZeRO-3: ReduceScatter for sharded gradients
    elif 'reducescatter' in name_lower:
        return 'DP'  # ZeRO-3 is data parallel
    
    # PP: Send/Recv for pipeline parallelism
    elif 'send' in name_lower or 'recv' in name_lower:
        return 'PP'
    
    # EP: AllToAll for expert parallelism (MoE)
    elif 'alltoall' in name_lower:
        return 'EP'
    
    else:
        return 'OTHER'

def analyze_phase_windows(nsys_rep_file):
    """
    Analyze time windows between different parallelism phases
    
    Returns:
        dict: Window statistics by phase transition
    """
    if not os.path.exists(nsys_rep_file):
        print(f"File not found: {nsys_rep_file}")
        return None
    
    print(f"Analyzing phase windows in {nsys_rep_file}...")
    
    # Export to SQLite
    sqlite_file = export_to_sqlite(nsys_rep_file)
    if not sqlite_file:
        print("  Failed to export SQLite, cannot analyze phase windows")
        return None
    
    try:
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Query NCCL kernels with timeline
        # shortName/demangledName are INTEGER IDs referencing StringIds table
        query = """
        SELECT k.start, k.end, s.value as name, k.streamId
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
        
        if not kernels:
            print("  No NCCL kernels found in trace")
            conn.close()
            os.remove(sqlite_file)
            return None
        
        conn.close()
        os.remove(sqlite_file)
        
        if not kernels:
            print("  No NCCL kernels found")
            return None
        
        # Categorize events and build phases
        events = []
        for start, end, name, stream in kernels:
            category = categorize_comm_event(name)
            events.append({
                'start': start,
                'end': end,
                'category': category,
                'stream': stream
            })
        
        # Group consecutive events of same category into phases
        phases = []
        if events:
            current_phase = {
                'category': events[0]['category'],
                'start': events[0]['start'],
                'end': events[0]['end']
            }
            
            for event in events[1:]:
                if event['category'] == current_phase['category']:
                    # Extend current phase
                    current_phase['end'] = max(current_phase['end'], event['end'])
                else:
                    # Save current phase and start new one
                    phases.append(current_phase)
                    current_phase = {
                        'category': event['category'],
                        'start': event['start'],
                        'end': event['end']
                    }
            
            # Add last phase
            phases.append(current_phase)
        
        # Calculate windows between phases
        windows = defaultdict(list)
        
        for i in range(len(phases) - 1):
            phase1 = phases[i]
            phase2 = phases[i + 1]
            
            # Window time = start of next phase - end of current phase
            window_time_ns = phase2['start'] - phase1['end']
            
            # Categorize by transition type
            transition = f"{phase1['category']}->{phase2['category']}"
            windows[transition].append(window_time_ns)
        
        # Calculate statistics for each transition type
        stats = {
            'transitions': {},
            'total_windows': sum(len(w) for w in windows.values())
        }
        
        for transition, window_list in windows.items():
            if window_list:
                window_list_ms = [w / 1e6 for w in window_list]  # Convert to ms
                stats['transitions'][transition] = {
                    'count': len(window_list),
                    'mean_ms': np.mean(window_list_ms),
                    'p50_ms': np.percentile(window_list_ms, 50),
                    'p95_ms': np.percentile(window_list_ms, 95),
                    'min_ms': np.min(window_list_ms),
                    'max_ms': np.max(window_list_ms)
                }
        
        return stats
        
    except Exception as e:
        print(f"  Error analyzing phase windows: {e}")
        if os.path.exists(sqlite_file):
            os.remove(sqlite_file)
        return None

def metric_cal(directory):
    """
    Calculate phase window metrics
    
    Args:
        directory: Trace directory or nsys directory
    
    Returns:
        dict: Phase window statistics
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
            stats = analyze_phase_windows(nsys_file)
            
            if stats:
                print("\n" + "="*60)
                print("Phase Window Analysis Results")
                print("="*60)
                print(f"Total phase transitions: {stats['total_windows']}")
                print("\nWindow times by transition type:")
                
                for transition, data in sorted(stats['transitions'].items()):
                    print(f"\n  {transition}:")
                    print(f"    Count: {data['count']}")
                    print(f"    Mean: {data['mean_ms']:.2f} ms")
                    print(f"    P50:  {data['p50_ms']:.2f} ms")
                    print(f"    P95:  {data['p95_ms']:.2f} ms")
                    print(f"    Range: [{data['min_ms']:.2f}, {data['max_ms']:.2f}] ms")
                
                print("="*60)
                
                return stats
            break
    
    return {}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        result = metric_cal(directory)
    else:
        print("Usage: python phase_window_analyzer.py <nsys_directory>")

