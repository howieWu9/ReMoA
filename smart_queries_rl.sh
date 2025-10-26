#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/projects/moe-q"
DATA_ROOT="$BASE/data"
CACHE="$BASE/cache"
LOGS="$BASE/logs"
SRC="$BASE/moe_duqgen_klock_rl.py"
TMP="$BASE/.qgen_run.py"

mkdir -p "$LOGS"

# 1) 生成“临时副本”，只把 from_pretrained(...) 里的 dtype= 改成 torch_dtype=（不改原文件）
cp -f "$SRC" "$TMP"
perl -0777 -pe 's/(from_pretrained\([^)]*?)\bdtype\s*=/\1torch_dtype=/sg' -i "$TMP"

# 2) 纯离线 & 线程/UTF-8
export HF_HOME="$CACHE/hf" HF_DATASETS_CACHE="$CACHE/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MALLOC_ARENA_MAX=2
export PYTHONUTF8=1

# 3) 本地 Llama-2 路径（优先 snapshots/<hash>）
LLAMA="$HF_HOME/models--meta-llama--Llama-2-7b-chat-hf"
if [[ -d "$LLAMA/snapshots" ]]; then
  SNAP=$(find "$LLAMA/snapshots" -maxdepth 1 -type d ! -path "$LLAMA/snapshots" | head -n1 || true)
  [[ -n "$SNAP" && -f "$SNAP/config.json" ]] && LLAMA="$SNAP"
fi
[[ -f "$LLAMA/config.json" ]] || { echo "[FATAL] 未找到本地 Llama-2：$LLAMA/config.json"; exit 86; }

# 4) 数据集列表：优先 DATASETS 环境变量；否则读 available_datasets.txt；再否则用 data 目录下的子目录名
declare -a DS_ARR=()
if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
elif [[ -f "$BASE/available_datasets.txt" ]]; then
  readarray -t DS_ARR < "$BASE/available_datasets.txt"
else
  while IFS= read -r d; do DS_ARR+=("$(basename "$d")"); done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d)
fi

# 5) 逐数据集提交作业（固定 1000 条）
for ds in "${DS_ARR[@]}"; do
  ds="${ds%%[[:space:]]*}"
  [[ -n "$ds" ]] || continue

  DATA="$DATA_ROOT/$ds"
  DOC="$DATA/${ds}_documents.jsonl"
  EMB="$DATA/${ds}_embeddings.pt"
  OUT_DIR="$DATA/queries"
  mkdir -p "$OUT_DIR"

  if [[ ! -s "$DOC" || ! -s "$EMB" ]]; then
    echo "✗ 跳过 $ds：缺 $DOC 或 $EMB（先完成向量化）"
    continue
  fi

  submit.sh -j "qRL-$ds" \
    -o "$LOGS/query_rl_${ds}.out" \
    -e "$LOGS/query_rl_${ds}.err" \
    env HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MALLOC_ARENA_MAX=2 PYTHONUTF8=1 \
    python "$TMP" \
      --collection_jsonl "$DOC" \
      --collection_emb_pt "$EMB" \
      --out_dir "$OUT_DIR" \
      --cache_dir "$CACHE" \
      --local_only \
      --llama_model "$LLAMA" \
      --target_num_q 1000 \
      --use_candidate_best --n_cands 10 \
      --use_dedup --dedup_threshold 0.90 \
      --use_zscore --z_temp 1.3 \
      --eta 0.30 --use_floor --w_floor 0.05 --use_entropy --entropy 0.10 \
      --hard_match --lambda_hard 0.80 --hard_warmup_ratio 0.30 \
      --topk 500 --sim_threshold 0.50 --r1_tau 42 \
      --temp 0.7 --top_p 0.95 \
      --use_advantage --baseline_beta 0.05

  echo "→ 已提交 qRL-$ds ，输出：$OUT_DIR/generated_queries.jsonl"
done

echo
echo "查看日志：tail -f $LOGS/query_rl_<ds>.out"
echo "看进度（示例 scifact）：watch -n 2 'wc -l $DATA_ROOT/scifact/queries/generated_queries.jsonl; wc -l $DATA_ROOT/scifact/queries/weights_history.jsonl'"
