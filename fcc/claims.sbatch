#!/bin/bash
#SBATCH --job-name=ina-claims
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-18%1
#SBATCH --output=/home/%u/ina/logs/%x-%A_%a.out
# Level 3: the paper's own recipe (Table 1) on its 3 biggest InA-over-LoRA claims (Table 6).
# One line of claims_runs.txt per run: model task inhibition seed epochs target_modules extra_modules
#   inhibition < 0 = subtract the threshold (paper Eq. 10); 0 = plain LoRA baseline
#   INA_ACT=gelu   = the paper's default activation (paper §6.6.3; Table 5's GELU column = Tables 3 and 6)
# Change the batch size with:  BS=32 sbatch claims.sbatch   (Table 1 allows 16, 32 or 64)
# run_glue.py itself saves every epoch and reloads the epoch with the lowest dev loss for eval_results.json;
# collect_results.py's best_ep column (best dev score over epochs) is the number to compare with the paper.

source ~/ina/env.sh
export INA_ACT=${INA_ACT:-gelu}
BS=${BS:-16}
read -r MODEL TASK INH SEED EP TGT EXTRA <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/ina/scripts/claims_runs.txt)"
TAG=InA$(python -c "print(f'{round(abs($INH)*100):02d}')")
if [ "$TAG" != "InA00" ]; then
  if [[ "$INH" == -* ]]; then TAG="${TAG}_sub_${INA_ACT}"; else TAG="${TAG}_add_${INA_ACT}"; fi
fi
MTS=""; [ "$EXTRA" != "-" ] && MTS="--modules_to_save $EXTRA"
OUT="$RUNS/claims/$(basename "$MODEL")/$TASK/${TAG}_bs$BS/seed$SEED"
echo "=== run $SLURM_ARRAY_TASK_ID: $MODEL $TASK inh=$INH act=$INA_ACT seed=$SEED bs=$BS -> $OUT on $(hostname)"

python "$GLUE_PY" \
  --model_name_or_path "$MODEL" \
  --task_name "$TASK" \
  --do_train --do_eval \
  --num_train_epochs "$EP" \
  --learning_rate 5e-6 --warmup_steps 100 --weight_decay 0.01 \
  --adam_epsilon 1e-6 --adam_beta1 0.9 --adam_beta2 0.999 --max_grad_norm 1.0 \
  --lr_scheduler_type linear --optim adamw_torch \
  --per_device_train_batch_size "$BS" --per_device_eval_batch_size 64 \
  --task_dropout 0.15 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --lora_inhibition "$INH" \
  --target_modules "$TGT" $MTS \
  --task_type SEQ_CLS --peft_type LORA \
  --eval_strategy epoch --logging_strategy epoch \
  --seed "$SEED" \
  --dataloader_num_workers 4 \
  --output_dir "$OUT" --overwrite_output_dir \
  --report_to none
