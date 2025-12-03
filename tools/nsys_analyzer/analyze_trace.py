#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Entry Point for Trace Analysis
Analyzes nsys traces and computes all metrics

Usage:
    python analyze_trace.py <nsys_directory_or_file> [options]

Examples:
    # Analyze directory with multiple trace files
    python analyze_trace.py ../../nsys
    
    # Analyze single trace file
    python analyze_trace.py ../../nsys/trace_2nodes_rank_0.nsys-rep
    
    # Run specific metrics
    python analyze_trace.py ../../nsys --metrics nccl_calls,iteration_time
    
    # Export results
    python analyze_trace.py ../../nsys --output results.json --csv results.csv
    
    # Specify workload name and parallelism config
    python analyze_trace.py ../../nsys --name llama-3.1-8b-1n4g --dp 8 --tp 1 --pp 1 --ep 1
"""

import os
import sys
import json
import argparse
from collections import OrderedDict

# Import all metric analyzers
import direct_nsys_analyzer
import iteration_time_analyzer
import comm_time_breakdown
import comm_compute_overlap
import phase_window_analyzer
import bandwidth_model_analyzer  # Use new ring model analyzer

# Define all available metrics
AVAILABLE_METRICS = {
    'nccl_calls': {
        'name': 'NCCL Communication Calls',
        'module': direct_nsys_analyzer,
        'description': 'Count and categorize NCCL communication calls'
    },
    'iteration_time': {
        'name': 'Iteration Time Statistics',
        'module': iteration_time_analyzer,
        'description': 'Calculate mean, P50, P99 iteration times'
    },
    'comm_breakdown': {
        'name': 'Communication Time Breakdown',
        'module': comm_time_breakdown,
        'description': 'Break down time by DP/TP/PP/EP communication'
    },
    'overlap': {
        'name': 'Communication-Computation Overlap',
        'module': comm_compute_overlap,
        'description': 'Measure overlap between comm and compute'
    },
    'phase_windows': {
        'name': 'Parallelism Phase Windows',
        'module': phase_window_analyzer,
        'description': 'Analyze time gaps between parallelism phases'
    },
    'bandwidth': {
        'name': 'Bandwidth Utilization (Ring Model)',
        'module': bandwidth_model_analyzer,
        'description': 'Calculate bandwidth utilization using ring collective model'
    }
}

# Global parallelism config (set from command line)
PARALLELISM_CONFIG = {
    'dp_size': 1,
    'tp_size': 1,
    'pp_size': 1,
    'ep_size': 1
}

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_subheader(text):
    """Print formatted subsection header"""
    print("\n" + "-"*70)
    print(text)
    print("-"*70)

def validate_nsys_input(nsys_path):
    """
    Validate input path (can be directory or single file)
    
    Returns:
        tuple: (is_valid, is_file, trace_files)
            - is_valid: whether the input is valid
            - is_file: whether input is a single file
            - trace_files: list of trace files
    """
    if not os.path.exists(nsys_path):
        print(f"Error: Path not found: {nsys_path}")
        return False, False, []
    
    # 1. Check if it's a single .nsys-rep file
    if os.path.isfile(nsys_path):
        if nsys_path.endswith('.nsys-rep'):
            return True, True, [nsys_path]
        else:
            print(f"Error: File is not a .nsys-rep file: {nsys_path}")
            return False, False, []
    
    # 2. Check if it's a directory containing .nsys-rep files
    if os.path.isdir(nsys_path):
        nsys_files = [f for f in os.listdir(nsys_path) if f.endswith('.nsys-rep')]
        if not nsys_files:
            print(f"Error: No .nsys-rep files found in directory: {nsys_path}")
            return False, False, []
        return True, False, nsys_files
    
    print(f"Error: Invalid path: {nsys_path}")
    return False, False, []

def run_metric(metric_key, nsys_path, is_single_file):
    """
    Run a single metric analyzer
    
    Args:
        metric_key: metric identifier
        nsys_path: nsys file path (can be directory or single file)
        is_single_file: whether input is a single file
    
    Returns:
        dict: metric analysis results
    """
    if metric_key not in AVAILABLE_METRICS:
        print(f"Warning: Unknown metric '{metric_key}'")
        return None
    
    metric_info = AVAILABLE_METRICS[metric_key]
    print_subheader(f"Running: {metric_info['name']}")
    print(f"Description: {metric_info['description']}")
    
    try:
        # Get trace file path based on input type
        if is_single_file:
            trace_file = nsys_path
        else:
            # Directory mode: get first trace file
            trace_files = [f for f in os.listdir(nsys_path) if f.endswith('.nsys-rep')]
            if trace_files:
                trace_file = os.path.join(nsys_path, trace_files[0])
            else:
                trace_file = None
        
        # Run different metric analyses
        if metric_key == 'nccl_calls':
            # nccl_calls supports both directory and single file
            results = metric_info['module'].analyze_trace_directory(nsys_path)
        elif metric_key == 'iteration_time':
            if trace_file:
                results = metric_info['module'].analyze_iteration_time(trace_file)
            else:
                results = None
        elif metric_key == 'comm_breakdown':
            breakdown = metric_info['module'].analyze_comm_breakdown(nsys_path)
            # Note: communication_percentage will be calculated later based on iteration time
            results = breakdown
        elif metric_key == 'overlap':
            if trace_file:
                overlap_stats = metric_info['module'].analyze_overlap(trace_file)
                if overlap_stats:
                    # Calculate overlap percentage
                    if overlap_stats.get("total_nccl_time_ns", 0) > 0:
                        overlap_pct = (overlap_stats["overlap_time_ns"] / overlap_stats["total_nccl_time_ns"]) * 100
                    else:
                        overlap_pct = 0.0
                    results = {
                        'overlap_percentage': overlap_pct,
                        'total_nccl_time_ns': overlap_stats.get("total_nccl_time_ns", 0),
                        'total_compute_time_ns': overlap_stats.get("total_compute_time_ns", 0),
                        'overlap_time_ns': overlap_stats.get("overlap_time_ns", 0),
                        'is_estimated': overlap_stats.get("is_estimated", False)
                    }
                else:
                    results = None
            else:
                results = None
        elif metric_key == 'phase_windows':
            if trace_file:
                results = metric_info['module'].analyze_phase_windows(trace_file)
            else:
                results = None
        elif metric_key == 'bandwidth':
            if trace_file:
                # Use ring model analyzer with parallelism config and hardware bandwidth
                results = metric_info['module'].analyze_bandwidth_with_model(
                    trace_file,
                    dp_size=PARALLELISM_CONFIG['dp_size'],
                    tp_size=PARALLELISM_CONFIG['tp_size'],
                    pp_size=PARALLELISM_CONFIG['pp_size'],
                    ep_size=PARALLELISM_CONFIG['ep_size'],
                    hardware_bw=PARALLELISM_CONFIG.get('hardware_bw', 100e9)
                )
            else:
                results = None
        else:
            results = None
        
        if results:
            print(f"✓ {metric_info['name']} completed")
        else:
            print(f"✗ {metric_info['name']} failed or returned no data")
        
        return results
        
    except Exception as e:
        print(f"✗ Error running {metric_info['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_results(results, output_file):
    """Export results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Results exported to: {output_file}")
    except Exception as e:
        print(f"\n✗ Failed to export JSON: {e}")

def export_csv(results, csv_file):
    """Export summary results to CSV"""
    import csv
    
    try:
        # Safely extract key metrics for CSV
        nccl_calls = results.get('nccl_calls') or {}
        iteration_time = results.get('iteration_time') or {}
        comm_breakdown = results.get('comm_breakdown') or {}
        overlap = results.get('overlap') or {}
        
        # Get communication percentage from timeline analysis if available
        comm_pct = 'N/A'
        if comm_breakdown.get('timeline_analysis'):
            comm_pct = comm_breakdown['timeline_analysis'].get('comm_percentage', 'N/A')
        elif 'communication_percentage' in comm_breakdown:
            comm_pct = comm_breakdown.get('communication_percentage', 'N/A')
        
        row = {
            'workload_name': results.get('workload_name', 'unknown'),
            'num_traces': results.get('num_traces', 0),
            'nccl_total_calls': nccl_calls.get('total_nccl_calls', 'N/A'),
            'avg_iteration_time_ms': iteration_time.get('avg_iteration_time_ms', 'N/A'),
            'p99_iteration_time_ms': iteration_time.get('p99_iteration_time_ms', 'N/A'),
            'communication_pct': comm_pct,
            'overlap_pct': overlap.get('overlap_percentage', 'N/A') if isinstance(overlap, dict) else 'N/A'
        }
        
        # Check if file exists
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        print(f"✓ Results appended to: {csv_file}")
        
    except Exception as e:
        print(f"✗ Failed to export CSV: {e}")

def main():
    global PARALLELISM_CONFIG
    
    parser = argparse.ArgumentParser(
        description='Unified trace analyzer for nsys profiling data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all trace files in directory
  python analyze_trace.py ../../nsys
  
  # Analyze single trace file
  python analyze_trace.py ../../nsys/trace_2nodes_rank_0.nsys-rep
  
  # Run specific metrics only
  python analyze_trace.py ../../nsys --metrics nccl_calls,iteration_time
  
  # Export analysis results
  python analyze_trace.py ../../nsys --output results.json --csv summary.csv
  
  # Specify workload name and parallelism config
  python analyze_trace.py ../../nsys --name llama-3.1-8b --dp 8 --tp 1 --pp 1 --ep 1

Available metrics:
  nccl_calls      - NCCL communication call statistics
  iteration_time  - Iteration timing statistics (mean, P99)
  comm_breakdown  - Communication time breakdown by type
  overlap         - Communication-computation overlap analysis
  phase_windows   - Parallelism phase window analysis
  bandwidth       - Bandwidth utilization with ring model
        """
    )
    
    parser.add_argument('nsys_path', help='Directory containing nsys-rep files, or single nsys-rep file path')
    parser.add_argument('--name', default='unknown', help='Workload name for identification')
    parser.add_argument('--metrics', help='Comma-separated list of metrics (default: all)')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--csv', help='Output CSV file path (append mode)')
    parser.add_argument('--list', action='store_true', help='List available metrics and exit')
    
    # Parallelism configuration
    parser.add_argument('--dp', type=int, default=1, help='Data parallel size (default: 1)')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size (default: 1)')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline parallel size (default: 1)')
    parser.add_argument('--ep', type=int, default=1, help='Expert parallel size (default: 1)')
    
    # Hardware configuration
    parser.add_argument('--bw', type=float, default=100.0, 
                        help='Hardware bandwidth in GB/s (default: 100 for NVLink NV4)')
    
    args = parser.parse_args()
    
    # Set global parallelism config
    PARALLELISM_CONFIG['dp_size'] = args.dp
    PARALLELISM_CONFIG['tp_size'] = args.tp
    PARALLELISM_CONFIG['pp_size'] = args.pp
    PARALLELISM_CONFIG['ep_size'] = args.ep
    PARALLELISM_CONFIG['hardware_bw'] = args.bw * 1e9  # Convert GB/s to bytes/s
    
    # List metrics and exit
    if args.list:
        print("Available Metrics:")
        print("-" * 70)
        for key, info in AVAILABLE_METRICS.items():
            print(f"  {key:15s} - {info['description']}")
        return 0
    
    # Validate input path
    is_valid, is_single_file, trace_files = validate_nsys_input(args.nsys_path)
    if not is_valid:
        return 1
    
    # Determine which metrics to run
    if args.metrics:
        metrics_to_run = [m.strip() for m in args.metrics.split(',')]
        # Validate metric names
        invalid = [m for m in metrics_to_run if m not in AVAILABLE_METRICS]
        if invalid:
            print(f"Error: Unknown metrics: {', '.join(invalid)}")
            print(f"Use --list to see available metrics")
            return 1
    else:
        metrics_to_run = list(AVAILABLE_METRICS.keys())
    
    # Print analysis header
    print_header(f"Trace Analysis: {args.name}")
    if is_single_file:
        print(f"Mode: Single file")
        print(f"Trace file: {os.path.basename(args.nsys_path)}")
    else:
        print(f"Mode: Directory")
        print(f"Directory: {args.nsys_path}")
        print(f"Trace files: {', '.join(trace_files)}")
    print(f"Parallelism config: DP={args.dp}, TP={args.tp}, PP={args.pp}, EP={args.ep}")
    print(f"Hardware bandwidth: {args.bw} GB/s")
    print(f"Metrics to run: {', '.join(metrics_to_run)}")
    
    # Run all selected metrics
    results = OrderedDict()
    results['workload_name'] = args.name
    results['nsys_path'] = args.nsys_path
    results['is_single_file'] = is_single_file
    results['num_traces'] = 1 if is_single_file else len(trace_files)
    results['trace_files'] = [os.path.basename(args.nsys_path)] if is_single_file else trace_files
    results['parallelism_config'] = {
        'dp_size': args.dp,
        'tp_size': args.tp,
        'pp_size': args.pp,
        'ep_size': args.ep,
        'hardware_bw_gbps': args.bw
    }
    
    for metric_key in metrics_to_run:
        metric_results = run_metric(metric_key, args.nsys_path, is_single_file)
        results[metric_key] = metric_results
    
    # Post-process: Use accurate timeline analysis for communication percentage
    # The timeline_analysis provides accurate wall-clock time breakdown
    if results.get('comm_breakdown'):
        comm_data = results['comm_breakdown']
        
        # Check if we have accurate timeline analysis
        if comm_data.get('timeline_analysis'):
            # Timeline analysis is already computed and included in comm_breakdown
            # Add a note about the accuracy
            comm_data['note'] = 'Communication percentage based on accurate GPU timeline analysis'
        elif results.get('iteration_time'):
            # Fallback: estimate based on iteration time if timeline analysis not available
            iteration_data = results['iteration_time']
            complete_iterations = [t for t in iteration_data.get('iteration_times_ms', []) if t > 1000]
            
            if complete_iterations:
                actual_time_ms = sum(complete_iterations)
                nccl_time_ms = comm_data.get('total_nccl_time_ms', 0)
                compute_time_ms = comm_data.get('total_compute_time_ms', 0)
                
                # NCCL time may exceed actual time due to parallel execution on multiple streams
                if nccl_time_ms > actual_time_ms:
                    effective_comm_time = actual_time_ms - compute_time_ms
                    comm_data['effective_comm_time_ms'] = effective_comm_time
                    comm_data['comm_parallelism'] = nccl_time_ms / actual_time_ms
                    comm_data['communication_percentage'] = (effective_comm_time / actual_time_ms) * 100
                    comm_data['note'] = 'Estimated: NCCL kernels execute in parallel on multiple streams'
                else:
                    comm_data['effective_comm_time_ms'] = nccl_time_ms
                    comm_data['comm_parallelism'] = 1.0
                    comm_data['communication_percentage'] = (nccl_time_ms / actual_time_ms) * 100
                    comm_data['note'] = 'Estimated from iteration time'
                
                comm_data['actual_iteration_time_ms'] = actual_time_ms
                comm_data['num_complete_iterations'] = len(complete_iterations)
    
    # Print summary
    print_header("Analysis Summary")
    print(f"Workload: {args.name}")
    print(f"Traces analyzed: {1 if is_single_file else len(trace_files)}")
    print(f"Metrics computed: {len([r for r in results.values() if r is not None])}/{len(metrics_to_run)}")
    
    # Export results if requested
    if args.output:
        export_results(results, args.output)
    
    if args.csv:
        export_csv(results, args.csv)
    
    print("\n✓ Analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

