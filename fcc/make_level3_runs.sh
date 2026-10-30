#!/bin/bash
# Writes ~/ina/scripts/level3_runs.txt: one line per run = model task inhibition seed epochs targets extra-modules
# Edit the four lists to shrink or grow the sweep. Small GLUE tasks first so results arrive early.
MODELS="bert-large-cased"                 # add: FacebookAI/roberta-large microsoft/deberta-v3-large
TASKS="rte mrpc stsb cola sst2 qnli"      # add: mnli qqp (about a day each per run)
INHS="0.0 -0.3"                           # LoRA baseline vs the paper's subtractive InA; add -0.1 -0.9 or +0.3 (repo sign)
SEEDS="42 43 44"
OUT=~/ina/scripts/level3_runs.txt; : > "$OUT"
for m in $MODELS; do
  case $m in
    bert-large-cased)            EP=5;  TGT=query,key;            EXTRA=- ;;
    FacebookAI/roberta-large)    EP=10; TGT=query,key;            EXTRA=- ;;
    microsoft/deberta-v3-large)  EP=10; TGT=query_proj,key_proj;  EXTRA=pooler ;;
  esac
  for t in $TASKS; do for i in $INHS; do for s in $SEEDS; do
    echo "$m $t $i $s $EP $TGT $EXTRA" >> "$OUT"
  done; done; done
done
echo "$(wc -l < "$OUT") runs -> $OUT"
