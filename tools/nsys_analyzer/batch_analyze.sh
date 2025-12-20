#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# 批量分析脚本 - 对所有实验数据进行完整的 nsys trace 分析
# 
# 使用方法:
#   chmod +x batch_analyze.sh
#   ./batch_analyze.sh
#
# 输出:
#   每个文件夹下生成 analysis_result.json
#   汇总结果输出到 all_results_summary.json
# ============================================================================

set -e  # 遇到错误时停止

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 切换到分析器目录
cd "$SCRIPT_DIR"

echo "=============================================="
echo "CCL-Bench 批量分析脚本"
echo "=============================================="
echo "分析器目录: $SCRIPT_DIR"
echo "数据目录: $BASE_DIR"
echo ""

# 创建 result 目录
RESULT_DIR="$BASE_DIR/result"
mkdir -p "$RESULT_DIR"
echo "结果目录: $RESULT_DIR"
echo ""

# 定义要分析的实验配置
# 格式: "文件夹名:工作负载名称:DP大小:TP大小:PP大小:EP大小"
EXPERIMENTS=(
    "nsys_1node:llama-8b-1n4g-zero3:4:1:1:1"
    "nsys_2node:llama-8b-2n8g-zero3:8:1:1:1"
    "llama_1node_4GPU_zero2:llama-8b-1n4g-zero2:4:1:1:1"
    "llama_3zero_offload:llama-8b-1n4g-zero3-offload:4:1:1:1"
    "DeepSeek-V2-Lite_2n8g_zero3:deepseek-v2-lite-2n8g-zero3:8:1:1:1"
    "deepseek_r1_distill_llama_8B:deepseek-r1-8b-1n4g-zero3:4:1:1:1"
)

# 创建汇总结果文件
SUMMARY_FILE="$BASE_DIR/all_results_summary.json"
echo "[" > "$SUMMARY_FILE"

FIRST=true
TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # 解析配置
    IFS=':' read -r FOLDER NAME DP TP PP EP <<< "$exp"
    
    DATA_DIR="$BASE_DIR/$FOLDER"
    OUTPUT_FILE="$DATA_DIR/analysis_result.json"
    
    echo ""
    echo "=============================================="
    echo "[$CURRENT/$TOTAL] 分析: $NAME"
    echo "=============================================="
    echo "  文件夹: $FOLDER"
    echo "  并行配置: DP=$DP, TP=$TP, PP=$PP, EP=$EP"
    echo "  输出文件: $OUTPUT_FILE"
    echo ""
    
    # 检查数据目录是否存在
    if [ ! -d "$DATA_DIR" ]; then
        echo "  ⚠️  警告: 目录不存在，跳过: $DATA_DIR"
        continue
    fi
    
    # 检查是否有 nsys-rep 文件
    if ! ls "$DATA_DIR"/*.nsys-rep 1> /dev/null 2>&1; then
        echo "  ⚠️  警告: 没有找到 .nsys-rep 文件，跳过: $DATA_DIR"
        continue
    fi
    
    # 运行分析 (使用 python3 确保 f-string 支持)
    if python3 analyze_trace.py "$DATA_DIR" \
        --name "$NAME" \
        --dp "$DP" --tp "$TP" --pp "$PP" --ep "$EP" \
        -o "$OUTPUT_FILE"; then
        
        echo ""
        echo "  ✓ 分析完成: $OUTPUT_FILE"
        
        # 拷贝结果到 result 目录
        RESULT_FILENAME="${NAME}.json"
        cp "$OUTPUT_FILE" "$RESULT_DIR/$RESULT_FILENAME"
        echo "  ✓ 已拷贝到: $RESULT_DIR/$RESULT_FILENAME"
        
        # 追加到汇总文件
        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> "$SUMMARY_FILE"
        fi
        
        # 提取关键指标到汇总
        echo "  {" >> "$SUMMARY_FILE"
        echo "    \"name\": \"$NAME\"," >> "$SUMMARY_FILE"
        echo "    \"folder\": \"$FOLDER\"," >> "$SUMMARY_FILE"
        echo "    \"config\": {\"dp\": $DP, \"tp\": $TP, \"pp\": $PP, \"ep\": $EP}," >> "$SUMMARY_FILE"
        echo "    \"result_file\": \"$OUTPUT_FILE\"" >> "$SUMMARY_FILE"
        echo "  }" >> "$SUMMARY_FILE"
        
    else
        echo ""
        echo "  ✗ 分析失败: $FOLDER"
    fi
    
done

# 关闭 JSON 数组
echo "" >> "$SUMMARY_FILE"
echo "]" >> "$SUMMARY_FILE"

echo ""
echo "=============================================="
echo "批量分析完成!"
echo "=============================================="
echo "汇总文件: $SUMMARY_FILE"
echo ""

# 打印各实验结果文件位置
echo "各实验结果文件:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r FOLDER NAME DP TP PP EP <<< "$exp"
    OUTPUT_FILE="$BASE_DIR/$FOLDER/analysis_result.json"
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "  ✓ $NAME: $OUTPUT_FILE ($SIZE)"
    else
        echo "  ✗ $NAME: 未生成"
    fi
done

echo ""
echo "完成时间: $(date)"

