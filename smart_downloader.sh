#!/usr/bin/env bash
# 智能数据集下载器 - 自动检测、跳过、重试
set -euo pipefail

BASE_DIR="$HOME/projects/moe-q"
DATA_DIR="$BASE_DIR/data"
CONDA_ENV="moe_ir"

# 标准数据集（确认可用）
STANDARD_DATASETS=(
    "trec-covid"
    "nfcorpus"
    "nq"
    "hotpotqa"
    "fiqa"
    "arguana"
    "webis-touche2020"
    "cqadupstack"
    "quora"
    "dbpedia-entity"
    "scidocs"
    "fever"
    "climate-fever"
    "scifact"
)

# 问题数据集的名称变体（按优先级）
declare -A DATASET_VARIANTS=(
    ["signal"]="signal1m signal-1m signal signalM signal-1M"
    ["news"]="trec-news newsir news trec-news-2019"
    ["robust"]="robust04 robust robust-2004 robust-04"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_downloaded() {
    local dataset=$1
    local doc_file="$DATA_DIR/$dataset/${dataset}_documents.jsonl"
    
    if [ -f "$doc_file" ] && [ -s "$doc_file" ]; then
        local count=$(wc -l < "$doc_file")
        if [ "$count" -gt 100 ]; then
            return 0  # 已下载且完整
        fi
    fi
    return 1  # 未下载或不完整
}

test_url() {
    local dataset=$1
    local url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${dataset}.zip"
    
    # 快速检查 HTTP 状态码
    local status=$(curl -s -o /dev/null -w "%{http_code}" -I "$url" --connect-timeout 10 --max-time 15)
    
    if [ "$status" = "200" ]; then
        return 0  # URL 可用
    fi
    return 1  # URL 不可用
}

download_dataset() {
    local dataset=$1
    
    log "下载: $dataset"
    
    if python "$BASE_DIR/robust_dataset_downloader.py" "$dataset" "$DATA_DIR"; then
        log "✓ 成功: $dataset"
        return 0
    else
        log "✗ 失败: $dataset"
        return 1
    fi
}

try_variants() {
    local base_name=$1
    local variants="${DATASET_VARIANTS[$base_name]}"
    
    log "尝试 $base_name 的多个名称变体..."
    
    for variant in $variants; do
        log "  测试: $variant"
        
        # 先检查是否已下载
        if check_downloaded "$variant"; then
            log "  ✓ 已下载: $variant"
            echo "$variant"
            return 0
        fi
        
        # 测试 URL 是否可用
        if test_url "$variant"; then
            log "  ✓ URL 可用: $variant"
            
            # 尝试下载
            if download_dataset "$variant"; then
                echo "$variant"
                return 0
            fi
        else
            log "  ✗ URL 404: $variant"
        fi
    done
    
    log "  ✗ 所有变体都失败: $base_name"
    return 1
}

main() {
    log "=========================================="
    log "智能数据集下载器"
    log "=========================================="
    
    # 检查网络
    if ! curl -s --connect-timeout 5 https://www.google.com > /dev/null 2>&1; then
        log "错误: 无法访问外网"
        log "请先运行: border.sh open"
        exit 1
    fi
    
    # 激活环境
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || {
        log "错误: Conda 环境不存在"
        exit 1
    }
    
    # 存储实际可用的数据集
    local available_file="$BASE_DIR/available_datasets.txt"
    > "$available_file"  # 清空文件
    
    local total=0
    local downloaded=0
    local skipped=0
    local failed=0
    
    # 1. 处理标准数据集
    log ""
    log "=== 处理标准数据集 ==="
    for dataset in "${STANDARD_DATASETS[@]}"; do
        total=$((total + 1))
        
        # 检查是否已下载
        if check_downloaded "$dataset"; then
            log "✓ 已存在: $dataset"
            echo "$dataset" >> "$available_file"
            skipped=$((skipped + 1))
            continue
        fi
        
        # 下载
        if download_dataset "$dataset"; then
            echo "$dataset" >> "$available_file"
            downloaded=$((downloaded + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    # 2. 处理问题数据集（尝试变体）
    log ""
    log "=== 处理问题数据集（多名称尝试）==="
    for base_name in signal news robust; do
        total=$((total + 1))
        
        if result=$(try_variants "$base_name"); then
            echo "$result" >> "$available_file"
            downloaded=$((downloaded + 1))
        else
            failed=$((failed + 1))
        fi
    done
    
    # 3. 总结
    log ""
    log "=========================================="
    log "下载完成"
    log "=========================================="
    log "总计: $total"
    log "  - 新下载: $downloaded"
    log "  - 已存在: $skipped"
    log "  - 失败: $failed"
    log ""
    
    # 显示实际可用的数据集
    local actual_count=$(wc -l < "$available_file")
    log "实际可用的数据集 ($actual_count 个):"
    cat "$available_file" | while read ds; do
        log "  - $ds"
    done
    
    # 保存数据集列表供后续使用
    log ""
    log "数据集列表已保存到: $available_file"
    log ""
    
    if [ $actual_count -lt 10 ]; then
        log "警告: 可用数据集少于 10 个，建议检查网络和下载日志"
    fi
    
    log "下一步: 运行向量化"
    log "  bash $BASE_DIR/2_submit_embeddings_smart.sh"
}

main "$@"
