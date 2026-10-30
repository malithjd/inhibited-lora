# Reproducing InA (Kang et al., 2024) on the Fordham FCC cluster

Paper: Kang et al., *InA: Inhibition Adaption on pre-trained language models*, Neural Networks 178 (2024) 106410 (published version, checked Oct 2026).
Code: [ChengKang520/inhibited-lora](https://github.com/ChengKang520/inhibited-lora) at `d1d6685`, plus the changes listed at the top of [README.md](README.md).
Hardware: one GPU per run (NVIDIA L40 48 GB or RTX PRO 6000 Blackwell), Python 3.10, PyTorch 2.7.1 + CUDA 12.8, the repo's bundled peft 0.10.0 and transformers 4.53.0.
Every score below is one run (seed 42, fp32), taken at the last epoch, on the dev set. Raw output: [results/final_comparison.txt](results/final_comparison.txt).

## Summary

1. **The released checkpoints reproduce, with the right code version.** Testing them with the Aug 6, 2025 code (commit `59021f2`) gives exactly the released scores (20 of 20). Training the same 20 settings again with that code lands within 1 point of the released score on 17 of 20 (all within 2.4).
2. **The current upstream code is a different method.** It adds the threshold instead of subtracting it (the README calls this "shunting activation", for a coming paper). It cannot run the released inhibited checkpoints (near-random scores at Inh 0.3 and 0.9), and models trained with it show none of the released drop.
3. **The paper's numbers do not reproduce with the repo's commands.** At the paper's level for classification (Inh 0.3), our new runs are 0.6 to 6.8 points below Table 6 on 5 of 6 tasks. At Inh 0.9 they are 11 to 17 points below on CoLA, MRPC and RTE, where the paper reports almost no drop.
4. **With the repo's settings, inhibition does not beat plain LoRA.** At Inh 0.1 and 0.3 no task gains more than 0.2 points over our own Inh 0 run; Inh 0.9 loses 4.5 to 13 points on CoLA, MRPC, STS-B and RTE, and 5.1 F1 on SQuAD v1.1.
5. **The paper's main BERT numbers come from GELU, which is not in any public version of the code.** Table 5's GELU column equals the InA rows of Tables 3 and 6 digit for digit; the code only has LeakyReLU.
6. **Still open:** the paper's own recipe (Table 1). That is Level 3 below, not run yet.

## What we ran

| Set | What | Code | Settings | Runs |
|---|---|---|---|---|
| L1 cur | Test the 28 released BERT-large GLUE checkpoints (`output_final/BERT_large`) | current upstream | eval only | 28 |
| L1 aug6 | Test the 20 inhibited released checkpoints | Aug 6 code (Inh < 0 in this fork) | eval only | 20 |
| L2 old | Train the released settings again | current upstream (Inh > 0): shunting activation | README command: bert-large-cased, LoRA rank 4 (STS-B 8) on query, key, value, alpha 16, dropout 0.1, lr 5e-5, batch 8, 10 epochs, seed 42 | 26 |
| L2 new | Train the 20 inhibited settings again | Aug 6 code (Inh < 0): shunting inhibition | same as L2 old | 20 |
| SQuAD | README command | both | rank 2 on query, key, lr 3e-5, batch 4, 512 tokens, stride 128, 10 epochs | 5 |
| L3 | Paper Table 1 recipe | to decide | lr 5e-6, warmup 100, weight decay 0.01, batch 32, rank 8 on query, key, 3 seeds | not run |

The L2 settings match the released checkpoints exactly: every released `trainer_state.json` shows batch 8, 10 epochs and lr 5e-5 with linear decay and no warmup, and every `adapter_config.json` shows rank 4 (STS-B 8), alpha 16, dropout 0.1 on query, key, value.

## Final comparison: GLUE dev, BERT-large

Metric: CoLA MCC, STS-B Pearson, all others accuracy (MNLI matched). Inh = inhibition level.

| Task | Inh | Paper (Table 6) | Released | L1 cur | L1 aug6 | L2 old | L2 new | new − released | new − paper |
|---|---|---|---|---|---|---|---|---|---|
| CoLA | 0 | 65.5 | 60.3 | 60.3 | | 60.1 | = old | | |
| CoLA | 0.1 | 65.8 | 59.6 | 62.1 | 59.6 | 59.3 | 58.3 | −1.3 | −7.5 |
| CoLA | 0.3 | 65.9 | 58.8 | 2.5 | 58.8 | 59.1 | 59.1 | +0.3 | −6.8 |
| CoLA | 0.9 | 64.3 | 51.0 | 0.0 | 51.0 | 61.3 | 48.6 | −2.4 | −15.7 |
| SST-2 | 0 | 93.9 | 92.2 | 92.2 | | 92.9 | = old | | |
| SST-2 | 0.1 | 93.5 | 91.5 | 91.7 | 91.5 | 92.0 | 92.4 | +0.9 | −1.1 |
| SST-2 | 0.3 | 94.4 | 92.2 | 69.5 | 92.2 | 92.3 | 92.3 | +0.1 | −2.1 |
| SST-2 | 0.9 | 93.3 | 92.0 | 50.9 | 92.0 | 92.0 | 91.9 | −0.1 | −1.4 |
| MRPC | 0 | 85.0 | 84.6 | 84.6 | | 85.0 | = old | | |
| MRPC | 0.1 | 83.1 | 84.3 | 76.2 | 84.3 | 85.3 | 84.1 | −0.2 | +1.0 |
| MRPC | 0.3 | 84.8 | 80.9 | 31.6 | 80.9 | 85.8 | 79.7 | −1.2 | −5.1 |
| MRPC | 0.9 | 84.3 | 73.0 | 31.6 | 73.0 | 87.7 | 73.0 | 0.0 | −11.3 |
| STS-B | 0 | 88.7 | 90.1 | 90.1 | | 90.1 | = old | | |
| STS-B | 0.1 | 88.9 | 89.9 | 89.4 | 89.9 | 90.0 | 89.9 | 0.0 | +1.0 |
| STS-B | 0.3 | 89.0 | 89.8 | 48.3 | 89.8 | 90.1 | 89.8 | 0.0 | +0.8 |
| STS-B | 0.9 | 88.3 | 85.5 | −18.7 | 85.5 | 90.3 | 85.6 | +0.1 | −2.7 |
| QNLI | 0 | 92.5 | 92.0 | 92.0 | | 91.9 | = old | | |
| QNLI | 0.1 | 92.4 | 91.8 | 88.4 | 91.8 | 91.7 | 91.7 | −0.1 | −0.7 |
| QNLI | 0.3 | 92.7 | 92.2 | 50.6 | 92.2 | 91.7 | 92.1 | −0.1 | −0.6 |
| QNLI | 0.9 | 92.4 | 91.1 | 50.5 | 91.1 | 91.3 | 91.0 | −0.1 | −1.4 |
| RTE | 0 | 66.4 | 70.0 | 70.0 | | 67.5 | = old | | |
| RTE | 0.1 | 70.1 | 68.2 | 49.5 | 68.2 | 67.9 | 67.5 | −0.7 | −2.6 |
| RTE | 0.3 | 69.0 | 66.8 | 49.8 | 66.8 | 69.3 | 67.5 | +0.7 | −1.5 |
| RTE | 0.9 | 71.1 | 54.9 | 47.3 | 54.9 | 70.0 | 54.5 | −0.4 | −16.6 |
| MNLI | 0 | 86.6 | 85.6 | 85.6 | | 85.4 | = old | | |
| MNLI | 0.1 | 86.5 | 85.0 | 81.2 | 85.0 | | 85.2 | +0.2 | −1.3 |
| QQP | 0 | 91.5 | 89.7 | 89.7 | | 89.8 | = old | | |
| QQP | 0.1 | 91.4 | 89.6 | 86.3 | 89.6 | | 89.5 | −0.1 | −1.9 |

At Inh 0 there is no threshold, so old and new code train the same model. The authors released MNLI and QQP only at Inh 0 and 0.1.

## Final comparison: SQuAD dev (F1 / EM)

| Data, model | Inh | Paper (Table 6) | L2 old (adds Th) | L2 new (subtracts Th) |
|---|---|---|---|---|
| v1.1, BERT-large-cased | 0 (plain LoRA) | 91.1 / 84.3 | 85.5 / 76.9 | same |
| v1.1, BERT-large-cased | 0.9 | 91.3 / 84.6 | 85.5 / 77.0 | 80.4 / 70.4 |
| v2.0, BERT-large-cased | 0.9 | 81.5 / 78.1 | | 61.8 / 57.9 |
| v2.0, RoBERTa-large | 0.9 | 89.1 / 86.3 | | 82.2 / 78.6 |

Our plain-LoRA v1.1 run is already 5.6 F1 below the paper, so most of the SQuAD gap is the recipe (rank 2, lr 3e-5), not InA. There is no v2.0 plain-LoRA run yet, so the v2.0 gaps cannot be split into recipe and inhibition.

## Does inhibition help? Each run minus our own Inh 0 run

| Task | new 0.1 | new 0.3 | new 0.9 | old 0.3 | old 0.9 | paper 0.3 | paper 0.9 |
|---|---|---|---|---|---|---|---|
| CoLA | −1.8 | −1.0 | −11.5 | −1.0 | +1.2 | +0.4 | −1.2 |
| SST-2 | −0.5 | −0.6 | −1.0 | −0.6 | −0.9 | +0.5 | −0.6 |
| MRPC | −0.9 | −5.3 | −12.0 | +0.8 | +2.7 | −0.2 | −0.7 |
| STS-B | −0.2 | −0.3 | −4.5 | 0.0 | +0.2 | +0.3 | −0.4 |
| QNLI | −0.2 | +0.2 | −0.9 | −0.2 | −0.6 | +0.2 | −0.1 |
| RTE | 0.0 | 0.0 | −13.0 | +1.8 | +2.5 | +2.6 | +4.7 |
| MNLI | −0.2 | | | | | −0.3 | −0.3 |
| QQP | −0.3 | | | | | 0.0 | −0.1 |
| SQuAD v1.1 F1 | | | −5.1 | | 0.0 | 0.0 | +0.2 |

"new" = subtracts the threshold (the paper's method), "old" = adds it, "paper" = Table 6 minus Table 6 at Inh 0. These are single runs: RTE has 277 dev examples (one example = 0.36 points) and MRPC 408 (0.25 points), so differences under about 2 points on those two are within noise.

## Cross-checks

### This fork's code against the authors' own code

`python fcc/check_versions.py` builds the same small BERT with identical LoRA weights under three versions of the InA layer and compares the outputs:

```
SAME  fork -0.3  vs  Aug 6 code +0.3   (shunting inhibition = released checkpoints)
SAME  fork -0.9  vs  Aug 6 code +0.9
SAME  fork +0.3  vs  upstream +0.3     (shunting activation, unchanged)
SAME  fork +0.9  vs  upstream +0.9
SAME  fork 0     vs  upstream 0        (plain LoRA, unchanged)
SAME  upstream -0.3 vs upstream 0      (upstream silently ignores a negative value)
DIFF  fork -0.3  vs  fork 0            (should differ: the gate runs)
DIFF  fork -0.3  vs  fork +0.3         (should differ: two versions)
```

### How the InA code changed upstream

| Date | Commit | InA forward pass | `run_glue.py` adapts |
|---|---|---|---|
| Apr 24, 2024 | `f39c55f` | first public code: plain LoRA in the main path | |
| Aug 6, 2025 | `59021f2` | LeakyReLU(z − Inh × max z), slope 0.01 (the paper's Eq. 10) | query, key, value |
| Aug 8, 2025 | `d5773d3` | released GLUE checkpoints uploaded | query, key, value |
| Aug 11, 2025 to now | `4409cbb` | LeakyReLU(z + Inh × max z), slope 0.1; gate only when Inh > 0 | query, key |

### Paper (published version) against the code and the README

| Item | Paper | Code / README / released checkpoints |
|---|---|---|
| Threshold | Eq. 10 to 12: f(H W_down − Th), Th = Inh × max(H W_down) per token | Aug 6: same. Current: adds Th |
| Activation | GELU is the default (§1 (b), Fig. 2 caption, §5.2, §6.6.3); LeakyReLU in §4.3, §6.4.1 and Table 2 | LeakyReLU only: slope 0.01 (Aug 6), 0.1 (current) |
| Adapted matrices | query, key (Eq. 11 and 12) | released: query, key, value; current README command: query, key |
| LoRA rank | not stated; 0.8M trainable parameters implies rank 8 on query, key | released: rank 4 (STS-B 8) on query, key, value (about 0.6M) |
| Learning rate | 5e-6 (Table 1) | 5e-5 |
| Epochs | BERT 5, RoBERTa and DeBERTa 10 | 10 |
| Batch size | 16, 32 or 64 | 8 |
| Warmup, weight decay, Adam eps | 100 steps, 0.01, 1e-6 | 0, 0, 1e-8 (Hugging Face defaults) |
| Task-layer dropout | 0.15 | 0.1 (model default) |
| Seeds | several (results given as mean ± std) | one (seed 42) |
| Inh level used | 0.3 for classification, 0.9 for QA and SWAG (§5) | all of 0, 0.1, 0.3, 0.9 released |

### Inside the paper

1. Table 5's GELU column (BERT-large, Inh 0.3) equals Table 3's BERT [InA] row and Table 6's BERT (0.3) row digit for digit, so the main BERT results used GELU.
2. Inh 0 should equal LoRA, but Table 3's BERT [LoRA] row differs from Table 6's BERT (0) row: CoLA 64.2 vs 65.5, SST-2 94.2 vs 93.9, STS-B 89.2 vs 88.7, RTE 69.2 vs 66.4. SQuAD v1.1 also differs: 91.3 / 84.5 (Table 4) vs 91.1 / 84.3 (Table 6).
3. Table 4's BERT [FT] and BERT [LoRA] rows are identical in all five numbers (91.3 / 84.5, 81.7 / 78.4, 86.5).
4. §6.4.1 says "In Table 1, LeakyReLU has no obvious average gap with GeLU"; Table 1 lists hyperparameters, and the comparison is Table 5.
5. The upstream README (Aug 11, 2025) says a negative Th means shunting inhibition, but upstream only runs the gate when Inh > 0.

## Next

- Level 3, the paper's recipe. First decide the activation (GELU or LeakyReLU) and the sign. Proposed check: CoLA with 2 activations × 2 signs plus plain LoRA, 3 seeds each (15 runs, about 3 GPU hours); then the six small GLUE tasks at Inh 0 and 0.3.
- SQuAD v2.0 plain-LoRA baselines for BERT and RoBERTa.
- RoBERTa-large and DeBERTa-v3-large on GLUE; SWAG.

To rebuild the tables from your own runs: `python fcc/compare_all.py $RUNS --released output_final/BERT_large` (see [fcc/README.md](fcc/README.md)).
