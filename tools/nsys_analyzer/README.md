# Communication-Computation Metrics - Group X

Analyzes communication and computation time from nsys traces.

## Metrics

- Total Time
- Comm. Ratio
- Overlap %
- Kernel Duration
- Idle

## Structure

```
|-- nsys_analyzer/
    |-- analyze_trace.py                # Entry point
    |-- comm_time_breakdown.py          # Total Time, Comm. Ratio, Idle
    |-- comm_compute_overlap.py         # Overlap %
    |-- traffic_interval_analyzer.py    # Kernel Duration, Call Interval
    |-- iteration_time_analyzer.py      # Iteration Time Mean, P99
    |-- phase_window_analyzer.py        # Window time between phases
    |-- direct_nsys_analyzer.py         # NCCL call count
    |-- accurate_comm_time_analyzer.py  # Accurate comm time (helper)
    |-- visualize_kernel_breakdown.py   # Visualization charts
```

## Input

- nsys-rep files

## Output

- JSON with timing statistics
