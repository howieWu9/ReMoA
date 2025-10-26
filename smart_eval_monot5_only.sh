#!/usr/bin/env bash
set -euo pipefail

# ========== 目录 ==========
BASE="$HOME/projects/moe-q"
DUQ="$BASE/DUQGen"
DATA_ROOT="$BASE/data"
CACHE="$BASE/cache"
LOGS="$BASE/logs"
mkdir -p "$CACHE/hf" "$CACHE/pyterrier" "$LOGS"

TEST_PY="$DUQ/monot5_finetuning/test_monot5.py"

# 模式：prep | local | submit(默认)
MODE="${1:-submit}"

# 统一 HF 缓存（评估期默认离线；local 模式下会临时允许联网以兜底）
export HF_HOME="$CACHE/hf" HF_DATASETS_CACHE="$CACHE/hf"
export TOKENIZERS_PARALLELISM=false

# PyTerrier 放这里（而非 ~/.pyterrier）
export PYTERRIER_HOME="$CACHE/pyterrier"

# ========== 函数：有网节点预拉 PyTerrier ==========
prefetch_pyterrier_online() {
  echo "[PREP] 下载 PyTerrier 组件到: $PYTERRIER_HOME"
  python - <<PY
import os
os.environ["PYTERRIER_HOME"] = os.path.expanduser("$PYTERRIER_HOME")
import pyterrier as pt
pt.java.init()  # 会把 terrier-assemblies 与 helper 下载到 PYTERRIER_HOME
print("[OK] PyTerrier ready at:", pt.TERRIER_HOME)
PY
  echo "[PREP] 完成。现在计算节点可在离线情况下用 test_monot5.py。"
}

# ========== Java 检测 ==========
ensure_java() {
  if ! command -v javac >/dev/null 2>&1; then
    echo "[FATAL] 未找到 javac。请在该节点加载/安装 JDK（建议 17/21）。" >&2
    exit 3
  fi
  if [[ -z "${JAVA_HOME:-}" ]]; then
    export JAVA_HOME="$(readlink -f "$(command -v javac)" | sed -E 's#/bin/javac$##')"
  fi
  export PATH="$JAVA_HOME/bin:$PATH"
  echo "[INFO] JAVA_HOME=$JAVA_HOME"
}

# ========== 评估：本机直接跑 ==========
eval_one_local() {
  local ds="$1"
  local d="$DATA_ROOT/$ds"
  local MODEL="$BASE/models/monot5_${ds}_rl_3b"
  local TESTD="$d/${ds}_testdata.json"
  local QRELS="$d/${ds}_qrels.txt"
  local PRED="$d/${ds}_monot5_rl_predictions.json"

  [[ -s "$TESTD" && -s "$QRELS" ]] || { echo "– 跳过 $ds：缺测试文件"; return; }
  [[ -d "$MODEL" ]] || { echo "– 跳过 $ds：缺模型 $MODEL"; return; }
  [[ -s "$PRED" ]] && { echo "✓ 跳过 $ds：已存在 $PRED"; return; }

  ensure_java

  # 本机评估：允许联网（仅用于缺少 tokenizer 等极端兜底；主要用本地缓存）
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

  echo ">>> 评估 $ds（local）..."
  python "$TEST_PY" \
    --model_name_or_checkpoint_path "$MODEL" \
    --save_predictions_fn "$PRED" \
    --test_filename_path "$TESTD" \
    --qrle_filename "$QRELS" \
    --cache_dir "$CACHE" | tee "$LOGS/t5_eval_${ds}.out"
  echo "→ 完成：$PRED  | 日志：$LOGS/t5_eval_${ds}.out"
}

# ========== 评估：提交 SLURM（离线跑）==========
eval_one_submit() {
  local ds="$1"
  local d="$DATA_ROOT/$ds"
  local MODEL="$BASE/models/monot5_${ds}_rl_3b"
  local TESTD="$d/${ds}_testdata.json"
  local QRELS="$d/${ds}_qrels.txt"
  local PRED="$d/${ds}_monot5_rl_predictions.json"

  [[ -s "$TESTD" && -s "$QRELS" ]] || { echo "– 跳过 $ds：缺测试文件"; return; }
  [[ -d "$MODEL" ]] || { echo "– 跳过 $ds：缺模型 $MODEL"; return; }
  [[ -s "$PRED" ]] && { echo "✓ 跳过 $ds：已存在 $PRED"; return; }

  # 彻底离线：HF 与 PyTerrier 都用缓存
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  submit.sh -j "t5-eval-$ds" \
    -o "$LOGS/t5_eval_${ds}.out" -e "$LOGS/t5_eval_${ds}.err" \
    env HF_HOME="$HF_HOME" HF_DATASETS_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTERRIER_HOME="$PYTERRIER_HOME" \
    python "$TEST_PY" \
      --model_name_or_checkpoint_path "$MODEL" \
      --save_predictions_fn "$PRED" \
      --test_filename_path "$TESTD" \
      --qrle_filename "$QRELS" \
      --cache_dir "$CACHE"

  echo "→ 已提交：t5-eval-$ds    日志：$LOGS/t5_eval_${ds}.out"
}

# ========== 主流程 ==========
case "$MODE" in
  prep)
    prefetch_pyterrier_online
    ;;
  local)
    ensure_java
    for d in "$DATA_ROOT"/*; do
      [[ -d "$d" ]] || continue
      eval_one_local "$(basename "$d")"
    done
    ;;
  submit)
    # 这里不强制 ensure_java；计算节点通常有 module/jdk，失败会在作业日志体现
    for d in "$DATA_ROOT"/*; do
      [[ -d "$d" ]] || continue
      eval_one_submit "$(basename "$d")"
    done
    echo "查看队列：squeue -u $USER -n t5-eval-*"
    echo "看日志：tail -f $LOGS/t5_eval_*.out"
    ;;
  *)
    echo "Usage: $0 {prep|local|submit}"
    exit 2
    ;;
esac

# 友好提示
echo "预测文件：$DATA_ROOT/*/*_monot5_rl_predictions.json"
