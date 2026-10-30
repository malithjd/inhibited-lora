#!/bin/bash
#SBATCH --job-name=ina-squad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/%u/ina/logs/%x-%j.out
# usage: sbatch squad.sbatch <model> <squad|squad_v2> <inhibition> <rank>
#   e.g. sbatch squad.sbatch bert-large-cased squad 0.9 2    (old run, current code)
#        sbatch squad.sbatch bert-large-cased squad -0.9 2   (new run, 6 Aug code -> $RUNS/squad_aug06)

source ~/ina/env.sh
MODEL=$1; DATA=$2; INH=$3; R=$4
V2=""; [ "$DATA" = "squad_v2" ] && V2="--version_2_with_negative"
TAG=InA$(python -c "print(f'{round(abs($INH)*100):02d}')")
ROOT=squad; [[ "$INH" == -* ]] && ROOT=squad_aug06   # negative inhibition = 6 Aug code (new run)
OUT="$RUNS/$ROOT/$(basename "$MODEL")/$DATA/${TAG}_r$R"
echo "=== $MODEL $DATA inh=$INH r=$R -> $OUT on $(hostname)"

python "$QA_PY" \
  --model_name_or_path "$MODEL" \
  --dataset_name "$DATA" $V2 \
  --lora_r "$R" --lora_alpha 16 --lora_dropout 0.1 \
  --lora_inhibition "$INH" \
  --target_modules query,key \
  --task_type QUESTION_ANS --peft_type LORA \
  --do_train --do_eval \
  --learning_rate 3e-5 --num_train_epochs 10 \
  --max_seq_length 512 --doc_stride 128 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --dataloader_num_workers 4 \
  --output_dir "$OUT" --overwrite_output_dir \
  --report_to none
