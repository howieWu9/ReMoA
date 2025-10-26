#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$HOME/projects/moe-q"
DATA_DIR="$BASE_DIR/data"
CACHE_DIR="$BASE_DIR/cache"
LOGS_DIR="$BASE_DIR/logs"
DUQGEN_DIR="$BASE_DIR/DUQGen"

HF_HOME="$CACHE_DIR/hf"
TRANSFORMERS_CACHE="$HF_HOME"
HF_DATASETS_CACHE="$HF_HOME"

mkdir -p "$LOGS_DIR"

readarray -t DATASETS < "$BASE_DIR/available_datasets.txt"

for ds in "${DATASETS[@]}"; do
  DOC="$DATA_DIR/$ds/${ds}_documents.jsonl"
  EMB="$DATA_DIR/$ds/${ds}_embeddings.pt"
  [[ -s "$DOC" ]] || { echo "✗ $ds: 无文档"; continue; }
  [[ -s "$EMB" ]] && { echo "✓ $ds: 已有向量，跳过"; continue; }

  submit.sh -j "emb-$ds" \
    -o "$LOGS_DIR/embedding_${ds}.out" \
    -e "$LOGS_DIR/embedding_${ds}.err" \
    env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python "$DUQGEN_DIR/data_preparation/target_representation/generate_document_embedding.py" \
      --collection_data_filepath "$DOC" \
      --save_collection_embedding_filepath "$EMB" \
      --cache_dir "$CACHE_DIR"
done
