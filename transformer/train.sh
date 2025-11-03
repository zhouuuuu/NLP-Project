#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00         
#SBATCH --mem=32G
#SBATCH --job-name=train_seq_5
#SBATCH --output=logs/train_seq_%j.out
#SBATCH --error=logs/train_seq_%j.err

set -eo pipefail

# ===== 环境 =====
module load anaconda
eval "$(conda shell.bash hook)"
conda activate zhr
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HOME/.cache/huggingface"
# 如首次必须联网拉模型，先别开离线；缓存齐后可打开下一行
# export TRANSFORMERS_OFFLINE=1

cd ~/NLP_project/transformer
echo "[INFO] Node: $(hostname)  |  Start: $(date)"

# ===== 数据路径（你重划分后的文件）=====
export DATA_PATH="author_style_dataset_balanced.csv"

# ===== 统一超参（如需覆盖可在每个模型内再改）=====
export EPOCHS=8
export LR=2e-5
export SUPCON_LAMBDA=0.1
export MAX_LEN=512
export USE_AMP=True
export USE_MEAN_POOL=True
export WEIGHT_DECAY=0.01
export WARMUP_RATIO=0.1
export SEED=42

# ===== 要顺序运行的模型列表 =====
#MODELS="microsoft/deberta-v3-large"
#MODELS="microsoft/deberta-v3-base"
#MODELS="roberta-base"
#MODELS="bert-base-cased"
MODELS="bert-base-uncased"

# 汇总文件
SUMMARY="checkpoints/run_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "model_name,save_dir,val_macro_f1,epochs,lr,supcon_lambda,finished_at,exit_code" > "$SUMMARY"

for MODEL in "${MODELS[@]}"; do
  export MODEL_NAME="$MODEL"
  export SAVE_DIR="./checkpoints/${MODEL//\//-}"
  mkdir -p "$SAVE_DIR"

  # 按模型调批量与累积：large 显存大一点
  export BATCH_SIZE=16
  export GRAD_ACCUM_STEPS=1
  if [[ "$MODEL_NAME" == "microsoft/deberta-v3-large" ]]; then
    export BATCH_SIZE=8
    export GRAD_ACCUM_STEPS=2
  fi

  echo "=================================================="
  echo "[RUN] $(date) | MODEL=$MODEL_NAME | SAVE_DIR=$SAVE_DIR"
  echo "=================================================="

  # 可选：预下载（若需要联网缓存）
  # python - <<'PY'
  # from transformers import AutoTokenizer, AutoModel; import os
  # name = os.environ["MODEL_NAME"]
  # print("[Prefetch] Tokenizer:", name); AutoTokenizer.from_pretrained(name, use_fast=False)
  # print("[Prefetch] Model    :", name); AutoModel.from_pretrained(name)
  # PY

  # 训练并同时写各自目录下的 run.log
  python -u train.py 2>&1 | tee "$SAVE_DIR/run.log"
  RC=${PIPESTATUS[0]}

  # 从 results.jsonl 抓取最后一条 val_macro_f1（若存在）
  LAST_F1="NA"
  if [[ -f "$SAVE_DIR/results.jsonl" ]]; then
    LAST_F1=$(tail -n 1 "$SAVE_DIR/results.jsonl" | python - <<'PY'
import sys, json
line = sys.stdin.read().strip()
try:
    d = json.loads(line)
    print(d.get("val_macro_f1","NA"))
except Exception:
    print("NA")
PY
)
  fi

  echo "${MODEL_NAME},${SAVE_DIR},${LAST_F1},${EPOCHS},${LR},${SUPCON_LAMBDA},$(date),${RC}" >> "$SUMMARY"

  if [[ $RC -ne 0 ]]; then
    echo "[WARN] ${MODEL_NAME} 训练失败，退出码 $RC —— 将继续下一个模型。"
  else
    echo "[OK] ${MODEL_NAME} 完成。"
  fi
done

echo "=================================================="
echo "[DONE] All models finished at $(date)"
echo "[SUMMARY] $SUMMARY"
