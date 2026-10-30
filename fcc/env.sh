# Sourced by every job:  source ~/ina/env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ina
export HF_HOME=/storage/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export REPO=$HOME/inhibited-lora
export RUNS=/storage/$USER/ina_runs
export GLUE_PY=$REPO/transformers-4.53.0/examples/pytorch/text-classification/run_glue.py
export QA_PY=$REPO/transformers-4.53.0/examples/pytorch/question-answering/run_qa.py
# If compute nodes have no internet (Step 3.1), uncomment:
# export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
