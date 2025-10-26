#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/projects/moe-q"
DUQ="$BASE/DUQGen"
DATA_ROOT="$BASE/data"
CACHE="$BASE/cache"
LOGS="$BASE/logs"
mkdir -p "$LOGS" "$BASE/models"

TRAIN_PY="$DUQ/monot5_finetuning/train_monot5.py"
TEST_PY="$DUQ/monot5_finetuning/test_monot5.py"
REMOTE_MODEL="castorini/monot5-3b-msmarco-10k"

HF_HOME="${CACHE}/hf"   # 你之前就这样设的
export HF_HOME HF_DATASETS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# —— 解析 HuggingFace 本地 snapshot —— #
resolve_local_model() {
  local repo="$1"   # 形如 org/name
  local name="${repo/\//--}"                 # org--name
  local root="${HF_HOME}/models--${name}"
  # 先看根目录（有些缓存结构把 config.json 放根）
  if [[ -f "${root}/config.json" ]]; then
    echo "${root}"; return 0
  fi
  # 再找 snapshots 下的任意一个包含 config.json 的目录
  if [[ -d "${root}/snapshots" ]]; then
    local snap
    snap=$(find "${root}/snapshots" -mindepth 1 -maxdepth 1 -type d \
            -exec test -f "{}/config.json" ';' -print | head -n1)
    if [[ -n "${snap:-}" ]]; then
      echo "${snap}"; return 0
    fi
  fi
  # 兜底：在整个 HF_HOME 下搜（防止结构非标准）
  local any
  any=$(find "${HF_HOME}" -path "*/models--${name}/snapshots/*" -maxdepth 0 -type d \
          -exec test -f "{}/config.json" ';' -print | head -n1)
  if [[ -n "${any:-}" ]]; then
    echo "${any}"; return 0
  fi
  echo ""
}

LOCAL_MODEL="$(resolve_local_model "${REMOTE_MODEL}")"
if [[ -z "${LOCAL_MODEL}" ]]; then
  echo "[FATAL] 本地未找到 ${REMOTE_MODEL} 的 config.json"
  echo "        期望在：${HF_HOME}/models--castorini--monot5-3b-msmarco-10k[/snapshots/<hash>]/config.json"
  echo "        解决：在有网节点预下载到该目录（或把已有缓存 rsync/拷贝过来），再重试。"
  exit 7
fi
echo "[OK] 使用离线基座模型：${LOCAL_MODEL}"

# —— 遍历所有有训练&测试文件的数据集 —— #
mapfile -t DIRS < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

for d in "${DIRS[@]}"; do
  ds="$(basename "$d")"
  TRAIN="$d/${ds}_reranker_train.jsonl"
  TESTD="$d/${ds}_testdata.json"
  QRELS="$d/${ds}_qrels.txt"
  [[ -s "$TRAIN" && -s "$TESTD" && -s "$QRELS" ]] || { echo "– 跳过：$ds（缺训练或测试文件）"; continue; }

  SAVE="$BASE/models/monot5_${ds}_rl_3b"
  PRED="$d/${ds}_monot5_rl_predictions.json"

  # 训练：如果已经有权重/配置就跳过
  if [[ -s "$SAVE/pytorch_model.bin" || -s "$SAVE/config.json" ]]; then
    echo "✓ 跳过训练：$ds（$SAVE 已存在）"
  else
    submit.sh -j "t5-train-$ds" \
      -o "$LOGS/t5_train_${ds}.out" -e "$LOGS/t5_train_${ds}.err" \
      env HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python "$TRAIN_PY" \
        --base_modename_or_path "$LOCAL_MODEL" \
        --train_data_filepath "$TRAIN" \
        --save_model_path "$SAVE" \
        --cache_dir "$CACHE"
  fi

  # 评测：已有预测就跳过；否则如果刚提交了训练就加依赖
  if [[ -s "$PRED" ]]; then
    echo "✓ 跳过评测：$ds（已存在 $PRED）"
  else
    if squeue -h -n "t5-train-$ds" | grep -q .; then
      submit.sh -j "t5-eval-$ds" -d afterok:"t5-train-$ds" \
        -o "$LOGS/t5_eval_${ds}.out" -e "$LOGS/t5_eval_${ds}.err" \
        env HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python "$TEST_PY" \
          --model_name_or_checkpoint_path "$SAVE" \
          --save_predictions_fn "$PRED" \
          --test_filename_path "$TESTD" \
          --qrle_filename "$QRELS" \
          --cache_dir "$CACHE"
    else
      submit.sh -j "t5-eval-$ds" \
        -o "$LOGS/t5_eval_${ds}.out" -e "$LOGS/t5_eval_${ds}.err" \
        env HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python "$TEST_PY" \
          --model_name_or_checkpoint_path "$SAVE" \
          --save_predictions_fn "$PRED" \
          --test_filename_path "$TESTD" \
          --qrle_filename "$QRELS" \
          --cache_dir "$CACHE"
    fi
  fi
done

echo "日志：tail -f $LOGS/t5_train_<ds>.out    或    tail -f $LOGS/t5_eval_<ds>.out"
