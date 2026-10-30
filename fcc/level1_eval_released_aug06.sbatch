#!/bin/bash
#SBATCH --job-name=ina-L1-aug06
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/ina/logs/%x-%j.out
# Re-score the 20 released adapters that use inhibition (InA10/30/90) with the code they were
# trained with: commit 59021f2 (6 Aug 2025), LeakyReLU(z - Inh*max z), slope 0.01.
# After FCC patch v2, a NEGATIVE inhibition selects exactly that forward pass.
# Each adapter is copied (weights + config only) and its lora_inhibition set to -|value|;
# the originals in the repo are not touched.

source ~/ina/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COPIES=/storage/$USER/ina_adapters_aug06
declare -A TASK=( [CoLA]=cola [SST2]=sst2 [MRPC]=mrpc [STSB]=stsb [QQP]=qqp [MNLI]=mnli [QNLI]=qnli [RTE]=rte )

for ADAPTER in "$REPO"/output_final/BERT_large/*/InA[139]0; do
  NAME=$(basename "$(dirname "$ADAPTER")")        # e.g. RTE
  TAG=$(basename "$ADAPTER")                       # e.g. InA30
  COPY=$COPIES/$NAME/$TAG
  mkdir -p "$COPY"
  cp "$ADAPTER"/adapter_model.safetensors "$COPY"/
  python - "$ADAPTER/adapter_config.json" "$COPY/adapter_config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["lora_inhibition"] = -abs(c["lora_inhibition"])      # negative = 6 Aug 2025 code path
json.dump(c, open(sys.argv[2], "w"), indent=2)
print("lora_inhibition ->", c["lora_inhibition"])
PY
  echo "=== $NAME $TAG (inhibition flipped to negative)"
  python "$GLUE_PY" \
    --model_name_or_path bert-large-cased \
    --task_name "${TASK[$NAME]}" \
    --do_eval \
    --adapter_path "$COPY" \
    --task_type SEQ_CLS --peft_type LORA \
    --per_device_eval_batch_size 64 \
    --dataloader_num_workers 4 \
    --output_dir "$RUNS/level1_aug06/$NAME/$TAG" --overwrite_output_dir \
    --report_to none
done
