#!/bin/bash
# Configuration file for trace analysis
# Edit the TRACE_DIR variable to switch between different traces

# =================================================================
# Configuration - 修改这里切换不同的trace
# =================================================================

# 选择要分析的trace目录（取消注释其中一个）
TRACE_DIR="../../nsys"                                    # 默认：当前所有traces
# TRACE_DIR="../../nsys_1node_4gpu"                       # 1节点4卡
# TRACE_DIR="../../nsys_2node_8gpu"                       # 2节点8卡
# TRACE_DIR="/path/to/your/custom/nsys/directory"        # 自定义路径

# Workload名称（用于识别）
WORKLOAD_NAME="llama-3.1-8b-deepspeed"

# 选择要运行的metrics（留空表示运行所有）
# 可选: nccl_calls,iteration_time,comm_breakdown,overlap,phase_windows,bandwidth
METRICS=""  # 留空=运行所有
# METRICS="nccl_calls,iteration_time,comm_breakdown"     # 只运行这几个

# 输出文件配置
OUTPUT_JSON="results_${WORKLOAD_NAME}.json"
OUTPUT_CSV="all_results.csv"

# =================================================================
# 运行分析（不需要修改下面的代码）
# =================================================================

echo "======================================================================="
echo "Trace Analysis Configuration"
echo "======================================================================="
echo "Trace directory: $TRACE_DIR"
echo "Workload name:   $WORKLOAD_NAME"
echo "Metrics:         ${METRICS:-all}"
echo "Output JSON:     $OUTPUT_JSON"
echo "Output CSV:      $OUTPUT_CSV"
echo ""

# 检查trace目录是否存在
if [ ! -d "$TRACE_DIR" ]; then
    echo "Error: Trace directory not found: $TRACE_DIR"
    echo "Please edit analyze_config.sh and set the correct TRACE_DIR"
    exit 1
fi

# 检查是否有nsys-rep文件
if ! ls "$TRACE_DIR"/*.nsys-rep 1> /dev/null 2>&1; then
    echo "Error: No .nsys-rep files found in $TRACE_DIR"
    exit 1
fi

echo "Found nsys-rep files:"
ls -lh "$TRACE_DIR"/*.nsys-rep
echo ""

# 构建命令
CMD="python3 analyze_trace.py \"$TRACE_DIR\" --name \"$WORKLOAD_NAME\""

if [ -n "$METRICS" ]; then
    CMD="$CMD --metrics \"$METRICS\""
fi

if [ -n "$OUTPUT_JSON" ]; then
    CMD="$CMD --output \"$OUTPUT_JSON\""
fi

if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --csv \"$OUTPUT_CSV\""
fi

# 运行分析
echo "Running command:"
echo "$CMD"
echo ""
echo "======================================================================="
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✓ Analysis completed successfully!"
    echo "======================================================================="
    echo "Results saved to:"
    [ -f "$OUTPUT_JSON" ] && echo "  - $OUTPUT_JSON"
    [ -f "$OUTPUT_CSV" ] && echo "  - $OUTPUT_CSV"
else
    echo ""
    echo "======================================================================="
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "======================================================================="
fi

exit $EXIT_CODE

