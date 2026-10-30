# Running the InA reproduction on a Slurm cluster (Fordham FCC)

These scripts ran every result in [../REPRODUCTION.md](../REPRODUCTION.md). They expect this layout, which the steps below create:

| Path | What |
|---|---|
| `~/inhibited-lora` | this repo |
| `~/ina/scripts` | this `fcc/` folder (a link, so your edits stay under git) |
| `~/ina/env.sh` | sourced by every job: conda env `ina`, `$REPO`, `$RUNS`, `$HF_HOME` |
| `~/ina/logs` | one log per job |
| `/storage/$USER/ina_runs` (`$RUNS`) | results, one folder per run |
| `/storage/$USER/hf_cache` (`$HF_HOME`) | downloaded models and datasets |

## 1. Set up once

```bash
git clone https://github.com/malithjd/inhibited-lora.git ~/inhibited-lora
mkdir -p ~/ina/logs /storage/$USER/hf_cache /storage/$USER/ina_runs
ln -s ~/inhibited-lora/fcc ~/ina/scripts
cp ~/ina/scripts/env.sh ~/ina/env.sh            # edit if your paths differ

conda create -n ina python=3.10 -y && conda activate ina
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # cu128 runs on L40 and Blackwell
cd ~/inhibited-lora
grep -v -E '^(torch|triton|nvidia-)' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt
pip install -e peft-0.10.0/ -e transformers-4.53.0/

source ~/ina/env.sh
python ~/ina/scripts/prefetch.py                 # models, GLUE, SQuAD and metrics into $HF_HOME (about 6 GB)
python fcc/check_versions.py                     # every line SAME except the two "should differ" lines
```

Do not run `apply_fcc_patches.py` here: its changes are already in this repo. It is kept as the record of what changed, and it can patch a fresh upstream clone.

## 2. Run

| Script | Runs | Results in |
|---|---|---|
| `sbatch ~/ina/scripts/level1_eval_released.sbatch` | test the 28 released checkpoints, current code | `$RUNS/level1` |
| `sbatch ~/ina/scripts/level1_eval_released_aug06.sbatch` | test the 20 inhibited checkpoints, Aug 6 code | `$RUNS/level1_aug06` |
| `sbatch ~/ina/scripts/level2_glue_repo.sbatch` | 28 lines of `level2_runs.txt`, one at a time | `$RUNS/level2` |
| `sbatch ~/ina/scripts/level2_aug06.sbatch` | 20 lines of `level2_aug06_runs.txt` | `$RUNS/level2_aug06` |
| `sbatch ~/ina/scripts/squad.sbatch <model> <squad or squad_v2> <inh> <rank>` | one SQuAD run | `$RUNS/squad` (inh ≥ 0), `$RUNS/squad_aug06` (inh < 0) |
| `bash ~/ina/scripts/make_level3_runs.sh`, then `sbatch --array=1-<N>%1 ~/ina/scripts/level3_glue_paper.sbatch` | paper Table 1 recipe, 3 seeds | `$RUNS/level3` |

Watch: `squeue -u $USER`, `tail -f ~/ina/logs/<job log>`. Tables: `python ~/ina/scripts/compare_all.py $RUNS --released $REPO/output_final/BERT_large` (side by side) or `python ~/ina/scripts/collect_results.py $RUNS/<set>` (one set, with mean ± std over seeds).

## 3. Change a setting

- **Which runs:** edit the run lists. Each line of `level2_runs.txt` and `level2_aug06_runs.txt` is `NAME task inhibition rank`; then change `--array=1-N%1` in the `.sbatch` to the number of lines. For Level 3, edit `MODELS`, `TASKS`, `INHS`, `SEEDS` in `make_level3_runs.sh`.
- **The inhibition sign picks the code version:** negative = shunting inhibition (subtracts the threshold, as in the paper and the released checkpoints), positive = shunting activation (adds it, current upstream code), 0 = plain LoRA.
- **Activation:** `INA_ACT=gelu sbatch ...` or `INA_ACT=relu sbatch ...` (default LeakyReLU).
- **Hyperparameters:** edit the `python ... run_glue.py` flags inside the `.sbatch` file (learning rate, epochs, batch size, `--lora_r`, `--target_modules`). `--target_modules` is `query,key` (paper), `query,key,value` (released checkpoints) or `query_proj,key_proj` (DeBERTa).
- **New output folder:** change the `--output_dir` line, so new results do not overwrite old ones.
- Student accounts get one GPU, so keep `%1` in `--array`; each run waits for the previous one.
