#!/bin/bash
#SBATCH --job-name=ina-L3-paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/ina/logs/%x-%A_%a.out
# Level 3: the paper's Table 1 recipe. Submit with the array size printed by make_level3_runs.sh:
#   sbatch --array=1-<N>%1 level3_glue_paper.sbatch

source ~/ina/env.sh
read -r MODEL TASK INH SEED EP TGT EXTRA <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/ina/scripts/level3_runs.txt)"
SIGN=$(python -c "print('neg' if $INH < 0 else 'pos')")
TAG=InA$(python -c "print(f'{round(abs($INH)*100):02d}')")_${SIGN}_${INA_ACT:-leaky}
MTS=""; [ "$EXTRA" != "-" ] && MTS="--modules_to_save $EXTRA"
OUT="$RUNS/level3/$(basename "$MODEL")/$TASK/$TAG/seed$SEED"
echo "=== $MODEL $TASK inh=$INH seed=$SEED -> $OUT on $(hostname)"

python "$GLUE_PY" \
  --model_name_or_path "$MODEL" \
  --task_name "$TASK" \
  --do_train --do_eval \
  --num_train_epochs "$EP" \
  --learning_rate 5e-6 --warmup_steps 100 --weight_decay 0.01 \
  --adam_epsilon 1e-6 --max_grad_norm 1.0 --lr_scheduler_type linear \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --lora_inhibition "$INH" \
  --target_modules "$TGT" $MTS \
  --task_type SEQ_CLS --peft_type LORA \
  --eval_strategy epoch \
  --seed "$SEED" \
  --dataloader_num_workers 4 \
  --output_dir "$OUT" --overwrite_output_dir \
  --report_to none
