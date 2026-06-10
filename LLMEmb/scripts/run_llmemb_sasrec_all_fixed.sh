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

    python main.py \
      --model_name llmemb_sasrec \
      --dataset "${DATASET}" \
      --gpu_id "${GPU}" \
      --seed "${SEED}" \
      --train_batch_size 256 \
      --lr 0.001 \
      --l2 1e-6 \
      --num_train_epochs 200 \
      --patience 10 \
      --num_workers 5 \
      --log \
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

