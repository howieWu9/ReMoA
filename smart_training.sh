#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$HOME/projects/moe-q"
DATA_DIR="$BASE_DIR/data"
CACHE_DIR="$BASE_DIR/cache"
LOGS_DIR="$BASE_DIR/logs"
RESULTS_DIR="$BASE_DIR/results"
MODELS_DIR="$BASE_DIR/models"
DUQGEN_DIR="$BASE_DIR/DUQGen"
HF_HOME="$CACHE_DIR/hf"; TRANSFORMERS_CACHE="$HF_HOME"; HF_DATASETS_CACHE="$HF_HOME"
MONOT5_BASE="castorini/monot5-3b-msmarco-10k"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$MODELS_DIR"
readarray -t DATASETS < "$BASE_DIR/available_datasets.txt"

for ds in "${DATASETS[@]}"; do
  QFILE="$DATA_DIR/$ds/${ds}_queries_rl.jsonl"
  [[ -s "$QFILE" ]] || { echo "✗ $ds: 缺少 queries：$QFILE"; continue; }

  # 5) 生成训练数据（两个文件：reranker + colbert）
  RERANK_TRAIN="$DATA_DIR/$ds/${ds}_reranker_train.jsonl"
  COLBERT_TRAIN="$DATA_DIR/$ds/${ds}_colbert_train.tsv"
  if [[ ! -s "$RERANK_TRAIN" || ! -s "$COLBERT_TRAIN" ]]; then
    submit.sh -j "train-data-$ds" \
      -o "$LOGS_DIR/traindata_${ds}.out" -e "$LOGS_DIR/traindata_${ds}.err" \
      env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python "$DUQGEN_DIR/data_preparation/query_generation/train_data_generation.py" \
        --dataset_name "$ds" \
        --generated_queries_filepath "$QFILE" \
        --save_reranker_traindata_filepath "$RERANK_TRAIN" \
        --save_colbert_traindata_filepath "$COLBERT_TRAIN" \
        --num_pos_to_neg 4
  fi

  # 6) 微调 T5（3B）
  OUT_MODEL="$MODELS_DIR/monot5_${ds}_rl"
  if [[ ! -f "$OUT_MODEL/config.json" ]]; then
    submit.sh -j "t5-train-$ds" \
      -o "$LOGS_DIR/t5train_${ds}.out" -e "$LOGS_DIR/t5train_${ds}.err" \
      env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python "$DUQGEN_DIR/monot5_finetuning/train_monot5.py" \
        --base_modename_or_path "$MONOT5_BASE" \
        --train_data_filepath "$RERANK_TRAIN" \
        --save_model_path "$OUT_MODEL" \
        --cache_dir "$CACHE_DIR"
  fi

  # 7) 准备测试数据
  BM25="$DATA_DIR/$ds/${ds}_bm25_top100.txt"
  TEST_JSON="$DATA_DIR/$ds/${ds}_testdata.json"
  QRELS_TXT="$DATA_DIR/$ds/${ds}_qrels.txt"
  if [[ ! -s "$TEST_JSON" ]]; then
    submit.sh -j "test-prep-$ds" \
      -o "$LOGS_DIR/testprep_${ds}.out" -e "$LOGS_DIR/testprep_${ds}.err" \
      env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python "$DUQGEN_DIR/data_preparation/testdata_preparation/generate_bm25_reranking_data.py" \
        --dataset_name "$ds" \
        --save_bm25_results_filepath "$BM25" \
        --save_testdata_filepath "$TEST_JSON" \
        --save_qrels_filepath "$QRELS_TXT"
  fi

  # 8) 评估 T5
  PRED="$DATA_DIR/$ds/${ds}_monot5_rl_predictions.json"
  if [[ ! -s "$PRED" ]]; then
    submit.sh -j "t5-eval-$ds" \
      -o "$LOGS_DIR/t5eval_${ds}.out" -e "$LOGS_DIR/t5eval_${ds}.err" \
      env HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python "$DUQGEN_DIR/monot5_finetuning/test_monot5.py" \
        --model_name_or_checkpoint_path "$OUT_MODEL" \
        --save_predictions_fn "$PRED" \
        --test_filename_path "$TEST_JSON" \
        --qrle_filename "$QRELS_TXT" \
        --cache_dir "$CACHE_DIR"
  fi
done

# 汇总 MonoT5 结果
cat > "$BASE_DIR/collect_monot5_results.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$HOME/projects/moe-q"; RESULTS_DIR="$BASE_DIR/results"; LOGS_DIR="$BASE_DIR/logs"
readarray -t DATASETS < "$BASE_DIR/available_datasets.txt"
OUT="$RESULTS_DIR/monot5_rl_results.csv"; mkdir -p "$RESULTS_DIR"
echo "dataset,ndcg@10,map@100,recall@100,mrr@10" > "$OUT"
for ds in "${DATASETS[@]}"; do
  log="$LOGS_DIR/t5eval_${ds}.out"
  if [[ -s "$log" ]]; then
    ndcg=$(grep -oP "NDCG@10[:\s]+\K[\d\.]+" "$log" || echo ""); \
    map=$(grep -oP "MAP@100[:\s]+\K[\d\.]+" "$log" || echo ""); \
    recall=$(grep -oP "Recall@100[:\s]+\K[\d\.]+" "$log" || echo ""); \
    mrr=$(grep -oP "MRR@10[:\s]+\K[\d\.]+" "$log" || echo ""); \
    echo "$ds,$ndcg,$map,$recall,$mrr" >> "$OUT"
  fi
done
echo "[OK] $OUT"; column -t -s, "$OUT" || cat "$OUT"
EOF
chmod +x "$BASE_DIR/collect_monot5_results.sh"
