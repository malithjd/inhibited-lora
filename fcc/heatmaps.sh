#!/bin/bash
# Step 9: attention heatmaps for the paper's "red bag" example from the released SQuAD-v2 adapters.
# Run inside a small job:  srun -p cpu -c 4 --mem=24G --time=1:00:00 bash ~/ina/scripts/heatmaps.sh
source ~/ina/env.sh
mkdir -p "$RUNS/heatmaps"
# copy only the adapters, so the authors' own PNGs stay in the repo for side-by-side comparison
for MODEL in google-bert/bert-large-uncased FacebookAI/roberta-large; do
  mkdir -p "$RUNS/heatmaps/$MODEL"
  cp -rn "$REPO/visualization/Output_PEFT/$MODEL"/model_squad_v2_inh* "$RUNS/heatmaps/$MODEL/"
done
cd "$REPO/visualization"
for MODEL in google-bert/bert-large-uncased FacebookAI/roberta-large; do
  for INH in 0.0 0.1 0.3 0.9; do
    echo "=== $MODEL inh=$INH"
    python visualize_lm.py \
      --adapter_name "$RUNS/heatmaps/$MODEL/" \
      --model_name "$MODEL" \
      --task squad_v2 \
      --lora_inhibition "$INH"
  done
done
echo "PNGs: $RUNS/heatmaps/<model>/squad_v2/Inhibition_<inh>/layer_<n>_head_average.png"
