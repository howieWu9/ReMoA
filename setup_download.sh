#!/usr/bin/env bash
# Script 1: 环境设置、克隆仓库、下载数据集并向量化
set -euo pipefail

# ==================== 配置 ====================
BASE_DIR="$HOME/projects/moe-q"
DUQGEN_DIR="$BASE_DIR/DUQGen"
DATA_DIR="$BASE_DIR/data"
CACHE_DIR="$BASE_DIR/cache"
LOGS_DIR="$BASE_DIR/logs"
MODELS_DIR="$BASE_DIR/models"
QUERIES_DIR="$BASE_DIR/queries"

# BEIR 数据集列表（完整的 18 个数据集）
DATASETS=(
    "fiqa"
    "scifact"
    "nfcorpus"
    "trec-covid"
    "bioasq"
    "nq"
    "hotpotqa"
    "signal1m"
    "trec-news"
    "robust04"
    "arguana"
    "touche-2020"
    "cqadupstack"
    "quora"
    "dbpedia-entity"
    "scidocs"
    "fever"
    "climate-fever"
)

# Conda 环境名称
CONDA_ENV="moe_ir"

# ==================== 辅助函数 ====================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# ==================== 主流程 ====================
main() {
    log "开始设置 MOE-Q 项目环境..."
    
    # 1. 创建目录结构
    log "创建目录结构..."
    mkdir -p "$BASE_DIR" "$DATA_DIR" "$CACHE_DIR" "$LOGS_DIR" "$MODELS_DIR" "$QUERIES_DIR"
    cd "$BASE_DIR" || error_exit "无法进入 $BASE_DIR"
    
    # 2. 克隆 DUQGen 仓库（如果还没有）
    if [ ! -d "$DUQGEN_DIR" ]; then
        log "克隆 DUQGen 仓库..."
        git clone https://github.com/emory-irlab/DUQGen.git "$DUQGEN_DIR" || error_exit "克隆仓库失败"
    else
        log "DUQGen 仓库已存在，跳过克隆"
    fi
    
    # 3. 创建/激活 Conda 环境
    log "设置 Conda 环境..."
    eval "$(conda shell.bash hook)"
    
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        log "创建新的 Conda 环境: $CONDA_ENV"
        conda create -n "$CONDA_ENV" python=3.9 -y || error_exit "创建环境失败"
    fi
    
    conda activate "$CONDA_ENV" || error_exit "激活环境失败"
    
    # 4. 安装依赖
    log "安装 Python 依赖..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate datasets beir scikit-learn pandas tqdm
    pip install sentencepiece protobuf bitsandbytes
    
    # 安装 DUQGen 依赖
    if [ -f "$DUQGEN_DIR/requirements.txt" ]; then
        pip install -r "$DUQGEN_DIR/requirements.txt"
    fi
    
    # 5. 创建数据下载和处理脚本
    log "创建数据处理脚本..."
    cat > "$BASE_DIR/process_dataset.py" << 'PYEOF'
import os
import sys
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def process_dataset(dataset_name, data_dir):
    """下载并处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    # 创建数据集目录
    dataset_path = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 下载数据集
    print(f"下载 {dataset_name} 数据集...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    try:
        downloaded_path = util.download_and_unzip(url, dataset_path)
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    
    # 加载数据
    print(f"加载数据...")
    try:
        corpus, queries, qrels = GenericDataLoader(downloaded_path).load(split="test")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return False
    
    # 转换为 DUQGen 格式
    output_file = os.path.join(dataset_path, f"{dataset_name}_documents.jsonl")
    print(f"转换为 JSONL 格式: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, doc_data in corpus.items():
            title = doc_data.get('title', '').strip()
            text = doc_data.get('text', '').strip()
            doctext = (title + ' ' + text).strip() if title else text
            
            doc_entry = {
                'docid': doc_id,
                'doctext': doctext
            }
            f.write(json.dumps(doc_entry, ensure_ascii=False) + '\n')
    
    print(f"✓ 完成 {dataset_name}")
    print(f"  - 文档数量: {len(corpus)}")
    print(f"  - 查询数量: {len(queries)}")
    print(f"  - 输出文件: {output_file}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_dataset.py <dataset_name> <data_dir>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    data_dir = sys.argv[2]
    
    success = process_dataset(dataset_name, data_dir)
    sys.exit(0 if success else 1)
PYEOF
    
    # 6. 为每个数据集提交下载作业
    log "提交数据集下载作业..."
    for dataset in "${DATASETS[@]}"; do
        dataset_dir="$DATA_DIR/$dataset"
        doc_file="$dataset_dir/${dataset}_documents.jsonl"
        
        # 如果文档文件已存在，跳过
        if [ -f "$doc_file" ]; then
            log "数据集 $dataset 已存在，跳过下载"
            continue
        fi
        
        log "提交下载作业: $dataset"
        submit.sh -j "dl-$dataset" \
                  -o "$LOGS_DIR/download_${dataset}.out" \
                  -e "$LOGS_DIR/download_${dataset}.err" \
                  -env "$CONDA_ENV" \
                  python "$BASE_DIR/process_dataset.py" "$dataset" "$DATA_DIR"
    done
    
    log "等待所有下载作业完成..."
    log "请使用 'squeue' 检查作业状态"
    log "下载完成后，运行脚本进行向量化"
    
    # 7. 创建向量化提交脚本
    cat > "$BASE_DIR/submit_embeddings.sh" << 'EMBEOF'
#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$HOME/projects/moe-q"
DUQGEN_DIR="$BASE_DIR/DUQGen"
DATA_DIR="$BASE_DIR/data"
CACHE_DIR="$BASE_DIR/cache"
LOGS_DIR="$BASE_DIR/logs"
CONDA_ENV="moe_ir"

DATASETS=(
    "fiqa" "scifact" "nfcorpus" "trec-covid" "quora" "dbpedia-entity"
    "scidocs" "fever" "climate-fever" "nq" "hotpotqa" "bioasq"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "提交文档向量化作业..."

for dataset in "${DATASETS[@]}"; do
    dataset_dir="$DATA_DIR/$dataset"
    doc_file="$dataset_dir/${dataset}_documents.jsonl"
    emb_file="$dataset_dir/${dataset}_embeddings.pt"
    
    # 检查文档文件是否存在
    if [ ! -f "$doc_file" ]; then
        log "警告: $dataset 文档文件不存在，跳过"
        continue
    fi
    
    # 如果向量文件已存在，跳过
    if [ -f "$emb_file" ]; then
        log "$dataset 向量文件已存在，跳过"
        continue
    fi
    
    log "提交向量化作业: $dataset"
    submit.sh -j "emb-$dataset" \
              -o "$LOGS_DIR/embedding_${dataset}.out" \
              -e "$LOGS_DIR/embedding_${dataset}.err" \
              -env "$CONDA_ENV" \
              python "$DUQGEN_DIR/data_preparation/target_representation/generate_document_embedding.py" \
              --collection_data_filepath "$doc_file" \
              --save_collection_embedding_filepath "$emb_file" \
              --cache_dir "$CACHE_DIR"
done

log "所有向量化作业已提交"
log "使用 'squeue' 检查作业状态"
EMBEOF
    
    chmod +x "$BASE_DIR/submit_embeddings.sh"
    
    # 8. 创建状态检查脚本
    cat > "$BASE_DIR/check_progress.sh" << 'CHKEOF'
#!/usr/bin/env bash

BASE_DIR="$HOME/projects/moe-q"
DATA_DIR="$BASE_DIR/data"

DATASETS=(
    "fiqa" "scifact" "nfcorpus" "trec-covid" "quora" "dbpedia-entity"
    "scidocs" "fever" "climate-fever" "nq" "hotpotqa" "bioasq"
)

echo "=========================================="
echo "数据集准备进度检查"
echo "=========================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    dataset_dir="$DATA_DIR/$dataset"
    doc_file="$dataset_dir/${dataset}_documents.jsonl"
    emb_file="$dataset_dir/${dataset}_embeddings.pt"
    
    printf "%-20s" "$dataset:"
    
    if [ -f "$doc_file" ]; then
        printf " ✓ 文档"
    else
        printf " ✗ 文档"
    fi
    
    if [ -f "$emb_file" ]; then
        printf " ✓ 向量"
    else
        printf " ✗ 向量"
    fi
    
    printf "\n"
done

echo ""
echo "运行中的作业:"
squeue -u $(whoami)
CHKEOF
    
    chmod +x "$BASE_DIR/check_progress.sh"
    
    log "=========================================="
    log "设置完成！"
    log "=========================================="
    log ""
    log "下一步操作："
    log "1. 等待下载作业完成（约 10-30 分钟）"
    log "   检查状态: $BASE_DIR/check_progress.sh"
    log ""
    log "2. 下载完成后运行向量化："
    log "   bash $BASE_DIR/submit_embeddings.sh"
    log ""
    log "3. 向量化完成后（约 1-3 小时），运行查询生成："
    log "   bash $BASE_DIR/2_generate_queries.sh"
    log ""
    log "目录结构："
    log "  - 项目根目录: $BASE_DIR"
    log "  - 数据目录: $DATA_DIR"
    log "  - 日志目录: $LOGS_DIR"
    log "  - 缓存目录: $CACHE_DIR"
}

main "$@"
