#!/bin/bash
#SBATCH --job-name=ina-L1-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/ina/logs/%x-%j.out
# Level 1: re-evaluate the 28 released BERT-large GLUE adapters, no training.

source ~/ina/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO"

declare -A TASK=( [CoLA]=cola [SST2]=sst2 [MRPC]=mrpc [STSB]=stsb [QQP]=qqp [MNLI]=mnli [QNLI]=qnli [RTE]=rte )

for ADAPTER in output_final/BERT_large/*/InA*; do
  NAME=$(basename "$(dirname "$ADAPTER")")        # e.g. RTE
  TAG=$(basename "$ADAPTER")                       # e.g. InA30
  echo "=== $NAME $TAG"
  python "$GLUE_PY" \
    --model_name_or_path bert-large-cased \
    --task_name "${TASK[$NAME]}" \
    --do_eval \
    --adapter_path "$ADAPTER" \
    --task_type SEQ_CLS --peft_type LORA \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 4 \
    --output_dir "$RUNS/level1/$NAME/$TAG" --overwrite_output_dir \
    --report_to none
done
