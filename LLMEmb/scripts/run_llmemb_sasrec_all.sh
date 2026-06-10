#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=0
LOG_DIR="logs/llmemb_sasrec_all"
mkdir -p "${LOG_DIR}"

DATASETS=("beauty" "ml-1m" "toys")
SEEDS=(0 1 42)

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    LOG_FILE="${LOG_DIR}/${DATASET}_llmemb_sasrec_seed${SEED}.log"

    echo "============================================================"
    echo "[START] DATASET=${DATASET} SEED=${SEED} MODEL=llmemb_sasrec"
    echo "[LOG]   ${LOG_FILE}"
    echo "============================================================"

    # 这里使用你之前适配后的 LLMEmb 固定负采样评估版本。
    # 如果你的 LLMEmb 参数名不是 --model_name，而是 --model，请把下面 --model_name 改成 --model。
    python main.py \
      --model_name llmemb_sasrec \
      --dataset "${DATASET}" \
      --gpu "${GPU}" \
      --random_seed "${SEED}" \
      --hidden_size 64 \
      --maxlen 20 \
      --batch_size 256 \
      --eval_batch_size 256 \
      --lr 0.001 \
      --l2_emb 1e-6 \
      --num_blocks 2 \
      --num_heads 1 \
      --dropout_rate 0.0 \
      --epoch 200 \
      --early_stop 10 \
      --fixed_eval_neg 1 \
      > "${LOG_FILE}" 2>&1

    echo "============================================================"
    echo "[DONE] DATASET=${DATASET} SEED=${SEED}"
    echo "============================================================"

    grep -E "Best Iter|Test After Training|Recall@5|NDCG@5|HR@5|Recall@10|NDCG@10|HR@10|Recall@20|NDCG@20|HR@20" "${LOG_FILE}" | tail -50 || true
    echo ""
  done
done

echo "All LLMEmb-SASRec experiments finished."

echo ""
echo "Quick summary:"
for f in "${LOG_DIR}"/*.log; do
  echo "============================================================"
  echo "$(basename "$f")"
  grep -E "Best Iter|Test After Training|Recall@5|NDCG@5|HR@5|Recall@10|NDCG@10|HR@10|Recall@20|NDCG@20|HR@20" "$f" | tail -40 || true
done
