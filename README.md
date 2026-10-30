# InA reproduction fork (Fordham University, Oct 2026)

**Fork of [ChengKang520/inhibited-lora](https://github.com/ChengKang520/inhibited-lora) at commit [`d1d6685`](https://github.com/ChengKang520/inhibited-lora/tree/d1d668520d6a3b9a32112bb7a27e88fe981d7fc1)**, the code for Kang et al., [*InA: Inhibition Adaption on pre-trained language models*, Neural Networks 178 (2024) 106410](https://doi.org/10.1016/j.neunet.2024.106410). InA is the original authors' work; this fork only makes their code runnable and checkable. The upstream README follows unchanged below the line.

**Results:** [REPRODUCTION.md](REPRODUCTION.md) · **Rerun or change settings:** [fcc/README.md](fcc/README.md) · **Full diff against upstream:** [d1d6685...main](https://github.com/malithjd/inhibited-lora/compare/d1d6685...main)

### Changes to the original repo

| File | Change | Why |
|---|---|---|
| `peft-0.10.0/src/peft/tuners/lora/layer.py` | The sign of `--lora_inhibition` (Inh) picks the version. **Inh < 0: shunting inhibition**, identical to upstream commit [`59021f2`](https://github.com/ChengKang520/inhibited-lora/commit/59021f2) (Aug 6, 2025): LeakyReLU(z − abs(Inh) × max z), slope 0.01. **Inh > 0: shunting activation**, the unchanged current code: LeakyReLU(z + Inh × max z), slope 0.1. **Inh = 0:** plain LoRA, unchanged. `INA_ACT=gelu` or `INA_ACT=relu` swaps the activation. | Upstream runs the gate only when Inh > 0, so a negative value silently trains plain LoRA. The released checkpoints need the Aug 6 version. |
| `peft-0.10.0/src/peft/tuners/lora/bnb.py` | Per-token `torch.max(z, dim=-1)` and the same sign rule. | The 4-bit path crashed on `torch.max(z)`. |
| `transformers-4.53.0/examples/pytorch/text-classification/run_glue.py` and `.../question-answering/run_qa.py` | New `--target_modules` (default `query,key`, as upstream) and `--adapter_path` (evaluate a saved adapter); `--modules_to_save` is now passed on. | Upstream hard-codes `query,key`; the released checkpoints use `query,key,value`. |
| `visualization/visualize_lm.py` | Removed `merge_and_unload()`. | Merging drops the InA gate from the heatmaps. |
| New: `fcc/`, `results/`, `REPRODUCTION.md` | Slurm scripts, run lists, result tables, and `fcc/check_versions.py`. | So anyone in the group can rerun or change a setting. |

`python fcc/check_versions.py` confirms the fork gives the same outputs as upstream for Inh ≥ 0 and as commit 59021f2 for Inh < 0.

---

#  Shunting Inhibition and Activation Adaption on Pre-Trained Language Models



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](LICENSE)


This repository contains the code necessary to reproduce Shunting Inhibition and Activation on LoRA, the shunting inhibition and activation mechanism that controls the activated information passing from the previous layer to the current layer introduced in the [paper](https://doi.org/10.1016/j.neunet.2024.106410) and one coming paper.

## Updates
- (Aug/11/2025) the experiments of shunting inhibition and shunting activation on LoRA. **Please Notice:**  When the $Th$ is negative, we are using the Shunting Inhibition. Otherwise, when the $Th$ is positive, the Shunting Activation is using on LoRA.
- (Aug/08/2025) Refresh the comparison results.
- (Aug/02/2025) Upload codes and the developing environment.
- (May/13/2024) Upload Adapter weights.
- (Apr/07/2024) Initial release.
## What is Shunting Inhibition? 

Bellow image illustrates how shunting inhibition and activation works, with its inhibition (the red box) and activation (the blue box) states. When the gate of shunting inhibition and activation is open, the signal transmission occurs across the joint, which can be influenced by shunting synapses. These shunting synapses play a crucial role in regulating neuronal function, and their activation can affect signal reception and transmission.

<div align="center">
  <figure>
    <img src="assets/shunting-inhibition-activation.jpg" width="90%"/>
  </figure>
</div>

## Why Shunting Inhibition is needed in Adaption Fine-tuning? 

Bellow is a practical example of InA when using it in the 𝐵𝐸𝑅𝑇-large model, which has been fine-tuned under question-answering datasets. Left panel explains the potential risk of LoRA, and right panel presents the visualization of the attention score on last second attention layer based on prior work Dai et al. (2019). The text is ’I put my red bag in the black bag.’, and the question is ’What is the colour of my bag?’, Therefore, the answer should be ’red’. There are two colours: red and black. Classical fine-tuning and adaption fine-tuning methods, such as LoRA, on downstream NLU tasks tend to choose the proper features from the entire ’redundant’ feature pool. This cannot essentially eliminate the influence of task-irrelevant words, for example, ’I’ and ’My’. After five epochs of InA fine-tuning, our inhibition vector can learn an incomplete intrinsic rank whose sole tail was eliminated by InA. Finally, activated by GeLU, which has a small negative tail, this incomplete intrinsic rank can provide the pre-trained weights with a small negative vector. Thus, these answer-irrelevant parts—’I’ and ’My’—in the intrinsic rank will be weakened or eliminated (see red stars in the right panel). We finally conclude that after InA fine-tuning, attention layers will pay less attention to such task-irrelevant information.

<div align="center">
  <figure>
    <img s[InA_GLUE.batch](..%2F..%2F..%2FDesktop%2FInA_GLUE.batch)rc="assets/lora-problem.jpg" width="100%"/>
  </figure>
</div>

**If only pay attention to the answer-relevant side (we have ignored the direction from the right to the left, as BERT based models have two directions.)**

Although LoRA and InA has the same answer, but from the average attention heatmap of all 24 layers shown bellow can find one solution to optimize LoRA (even one potential solution to overcome the **Hallucination of LLMs**):

**Shunting Inhibition mechanism has the obvious ability to weaken the answer-irrelevant passing information, as well as other attention scores**

<font face="Arial Black" size="3">1</font>

|Layer|Fully FT:|LoRA (no-inhibition):|InA (inhibiiton 10%):|InA (inhibiiton 30%):|InA (inhibiiton 90%):| 
|  :-:   |  :-:   | :-:  |   :-:  |   :-:  |   :-:  |
|1| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer0_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer0_average.png"  align=center > |
|2| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer1_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer1_average.png"  align=center > |
|3| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer2_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer2_average.png"  align=center > |
|4| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer3_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer3_average.png"  align=center > |
|21| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer20_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer20_average.png"  align=center > |
|22| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer21_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer21_average.png"  align=center > |
|23| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer22_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer22_average.png"  align=center > |
|24| <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_no/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_00/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_10/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_30/layer23_average.png"  align=center > |  <img style="height:100px" src="assets/AttentionHeatMap/squad/inhibition_90/layer23_average.png"  align=center > |






## How does Shunting Inhibition effect LoRA?

InA also inserts trainable inhibition matrices into transformer layers to approximate the weight updates. By using a low-rank decomposition $W_0 + \Delta = W_0 + W_{down}W_{up}$, where $W_{down} \in {R^{d\times{r}}}$, $W_{up} \in {R^{r\times{k}}}$, $Th \in {R^{M\times{1}}}$, InA updates the $Query$ and $Key$ projection matrices ($W_{q},W_{k}$) in the multi-head attention sub-layer. For the specific input $H$, InA modifies the projection output $H_{o}$ as:

$$H_{o} \leftarrow H_{o}+s \cdot f(HW_{down}+Th)W_{up},$$

where $s \in \{0, 1\}$ is a tunable scalar hyperparameter, and $Th$ is the threshold. 

**Notation.** We denote input hidden vectors as $H \in R^{M \times {d}}$ and the output of self-attention as $H_o \in R^{M \times {d}}$. $W_{k}, W_{q}, W_{v} \in R^{d \times {d}}$ are the projection matrices.

**Motivation.**  The motivation of InA on Transformer is to assemble a flexible gate with an adjustable inhibition vector to fine-tune downstream tasks. In addition, it should be able to automatically learn to rarefy tense features without sparsity settings. Under transfer learning, pre-trained language models can provide features for downstream tasks. The inhibition vector with a gate mechanism can learn to adjust and inhibit the provided features, and it finally makes tunable weights fit into a specific downstream task by fine-tuning. We formulate the linear InA layer as:

$$I_{k}=f(HW_{k-down}+Th_{k})W_{k-up},$$
$$I_{q}=f(HW_{q-down}+Th_{q})W_{q-up},$$

where $I_{k} \in {R^{M\times{d}}}$ and $I_{q} \in {R^{M\times{d}}}$, respectively, is the $Inhibition$ matrix in $Key$ side and $Query$ side; $f$ is the activation function; $Th_{k} \in {R^{M\times{1}}}$ is the product of $\max(HW_{k-down}) \times Inh_{p}$ in terms of the column-wise maximization and $Th_{q} \in {R^{M\times{1}}}$ is the product of $\max(HW_{q-down}) \times Inh_{p}$ in terms of the column-wise maximization. 


## Reproduction

This repository is a tutorial for finetuning LMs and LLMs with InA on GLUE, SquAD datasets! So here's how to reproduce:

## Installation

1. Install requirements

```bash
$ python -m venv InaEnv
$ source InaEnv/bin/activate
$ pip install -e peft-0.10.0/
$ pip install -e transformers-4.53.0/
```


## Finetune on GLUE Benchmarks

In the paper [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685), We can find that when the low rank $r$ is 2 or 4, the text classification task can get a better performance. Thus, we also set $r$ to 2 and 4 in our experiments. 

  - `BERT-large` needs 1 GPU. The finetuned weights are available at [weights](./output_final/BERT_large/). 
    ```bash
    python transformers-4.53.0/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-large-cased \
    --task_name cola \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --overwrite_output_dir \
    --output_dir output_final/BERT_large/CoLA/InA00/ \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_inhibition 0.0 \
    --lora_dropout 0.1 \
    --task_type "SEQ_CLS" \
    --peft_type "LORA"
    ```
  
  - `Llama2-7B` needs 1 or 2 GPUs
    ```bash
    python transformers-4.53.0/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --task_name cola \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --overwrite_output_dir \
    --output_dir output_final/Llama2/CoLA/ \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_inhibition 0.0 \
    --lora_dropout 0.1 \
    --task_type "SEQ_CLS" \
    --peft_type "LORA" \
    --batch_size 16 \
    --bf16 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2
    ```


### Experiments of GLUE
Our fine-tuning experiments are carried on half a DGX-2 node with 1x40 A100 GPU card, the results may vary due to different GPU models, drivers, CUDA SDK versions, using FP16 or FP32, and random seeds. 
We report our numbers based on multple runs with different random seeds here. Here are the results from the BERT-large model:

|Task	 | Command	                            	| Running Time(1x40G A100 GPUs) |
|--------|-------------------------------------|-------------------------------|
|MNLI large| 	`sbatch batch/InA_GLUE_MNLI.batch` | 	 	21h x 4                    |
|QQP large| 	`sbatch batch/InA_GLUE_QQP.batch`  | 		19h x 4                     |
|QNLI large| 	`sbatch batch/InA_GLUE_QNLI.batch` | 		2h x 4                      |
|MRPC large| 	`sbatch batch/InA_GLUE_MRPC.batch` | 		0.5h x 4                    |
|RTE large| 	`sbatch batch/InA_GLUE_RTE.batch`  | 		0.1h x 4                    |
|SST-2 large| 	`sbatch batch/InA_GLUE_SST2.batch` | 		1h x 4                      |
|STS-b large| 	`sbatch batch/InA_GLUE_STSB.batch` | 		0.5h x 4                    |
|CoLA large| 	`sbatch batch/InA_GLUE_CoLA.batch` | 		0.5h x 4                    |



## Finetune on SQuAD-V1.1 and SQuAD-V2.0 Benchmarks

Here are the tuning time when fine-tuning the BERT-large model:
|Task	 | Command	                            	| Running Time(1x40G A100 GPUs) |
|--------|-------------------------------------|-------------------------------|
|SQuAD v1.1| 	`sbatch batch/InA_SQuADv1.batch` | 	 	19h x 4                    |
|SQuAD v2.0| 	`sbatch batch/InA_SQuADv2.batch`  | 		28h x 4                     |

### 1. Run on 1 or 2 GPUs: 
  - `RoBERTa-large on SQuADv1.1`
    ```bash
    ## SQuADv1.1
    python transformers-4.53.0/examples/pytorch/question-answering/run_qa.py \
    --model_name FacebookAI/roberta-large \
    --dataset_name squad \
    --lora_r 2 \
    --lora_alpha 16 \
    --lora_inhibition 0.9 \
    --lora_dropout 0.1 \
    --task_type "QUESTION_ANS" \
    --peft_type "LORA" \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir output_final/RoBERTa_large/SQuAD_v1/InA90/ \
    --output_dir relative_squad \
    --per_device_eval_batch_size=4 \
    --per_device_train_batch_size=4
    ```
  

  - `RoBERTa-large on SQuADv2.0`
    ```bash
    ## SQuADv2.0
    python transformers-4.53.0/examples/pytorch/question-answering/run_qa.py \
    --model_name FacebookAI/roberta-large \
    --dataset_name squad_v2 \
    --lora_r 2 \
    --lora_alpha 16 \
    --lora_inhibition 0.9 \
    --lora_dropout 0.1 \
    --task_type "QUESTION_ANS" \
    --peft_type "LORA" \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir output_final/RoBERTa_large/SQuAD_v2/InA90/ \
    --output_dir relative_squad \
    --per_device_eval_batch_size=4 \
    --per_device_train_batch_size=4
    ```
  
  
  - `Llama2-7B on SQuADv2.0` needs 2 GPUs
    ```bash
    ## SQuADv2.0
    python transformers-4.53.0/examples/pytorch/question-answering/run_qa.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_name squad_v2 \
    --lora_r 2 \
    --lora_alpha 16 \
    --lora_inhibition 0.9 \
    --lora_dropout 0.1 \
    --task_type "QUESTION_ANS" \
    --peft_type "LORA" \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir output_final/Llama2_7B/SQuAD_v2/InA90/ \
    --output_dir relative_squad \
    --per_device_eval_batch_size=4 \
    --per_device_train_batch_size=4
    ```

  - `Llama3-8B on SQuADv2.0` needs 2 GPUs
    ```bash
    ## SQuADv2.0
    python transformers-4.53.0/examples/pytorch/question-answering/run_qa.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset_name squad_v2 \
    --lora_r 2 \
    --lora_alpha 16 \
    --lora_inhibition 0.9 \
    --lora_dropout 0.1 \
    --task_type "QUESTION_ANS" \
    --peft_type "LORA" \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir output_final/Llama3_8B/SQuAD_v2/InA90/ \
    --output_dir relative_squad \
    --per_device_eval_batch_size=4 \
    --per_device_train_batch_size=4
    ```

### 2. Visualization of Average Attention Heatmap 

  Check out how shunting inhibition can benefit the selection of attention scores from the [README.md](visualization/README.md) 

  - `BERT-large`
    ```bash
    cd visualization/
    python visualize_lm.py --adapter_name=/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/google-bert/bert-large-uncased/ \
    --model_name google-bert/bert-large-uncased \
    --task="squad_v2" \
    --lora_inhibition 0.9
    ```

  - `RoBERTa-large`
    ```bash
    cd visualization/
    python visualize_lm.py --adapter_name=/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/FacebookAI/roberta-large/ \
    --model_name FacebookAI/roberta-large \
    --task="squad_v2" \
    --lora_inhibition 0.9
    ```

  - `Llama2-7B`
    ```bash
    cd visualization/
    python visualize_llm.py --adapter_name=/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/Llama-2-7b-chat-hf/ \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --task="squad_v2" \
    --lora_inhibition 0.9
    ```

## Acknowledgements

Cheng Kang, Jindrich Prokop and Daniel Novak are supported by the Czech Technical University in Prague (grant number: SGS22/165/OHK3/3T/13), the Research Centre for Informatics (grant number: CZ.02.1.01/0.0/0.0/160\_19/0000765), and the Brain Dynamics(grant number: CZ.02.01.01/00/22\_008/0004643). We thank Yong Hu, Huiyu Zhou and Daniel Novak for proofreading the paper and providing insightful comments. We also thank the anonymous reviewers for valuable discussions.

## Contributing
We welcome contributions in any form! Assistance with documentation is always welcome. To contribute, feel free to open an issue or please fork the project make your changes and submit a pull request. We will do our best to work through any issues and requests.

## Citing this work
If our work helped you, please cite the reference:
```
@article{kang2024ina,
  title={InA: Inhibition Adaption on pre-trained language models},
  author={Kang, Cheng and Prokop, Jindrich and Tong, Lei and Zhou, Huiyu and Hu, Yong and Novak, Daniel},
  journal={Neural Networks},
  volume={178},
  pages={106410},
  year={2024},
  publisher={Elsevier}
}
```

