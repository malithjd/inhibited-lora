#!/bin/bash
#SBATCH --job-name=ina-L2-glue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-28%1
#SBATCH --output=/home/%u/ina/logs/%x-%A_%a.out
# Level 2: re-train exactly the 28 released configurations with the repo's settings
# (HF defaults: lr 5e-5, batch 8, seed 42, 10 epochs; InA on query,key,value like the released adapters).

source ~/ina/env.sh
read -r NAME TASK INH R <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/ina/scripts/level2_runs.txt)"
TAG=InA$(python -c "print(f'{round(abs($INH)*100):02d}')")
echo "=== run $SLURM_ARRAY_TASK_ID: $NAME inh=$INH r=$R -> $TAG on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

python "$GLUE_PY" \
  --model_name_or_path bert-large-cased \
  --task_name "$TASK" \
  --do_train --do_eval \
  --num_train_epochs 10 \
  --lora_r "$R" --lora_alpha 16 --lora_dropout 0.1 \
  --lora_inhibition "$INH" \
  --target_modules query,key,value \
  --task_type SEQ_CLS --peft_type LORA \
  --seed 42 \
  --dataloader_num_workers 4 \
  --output_dir "$RUNS/level2/$NAME/$TAG" --overwrite_output_dir \
  --report_to none
