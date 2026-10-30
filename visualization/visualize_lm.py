
# Basic Imports
import time
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import peft
import os
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer, BitsAndBytesConfig




import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
import os
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import json5
from threading import Thread
from typing import Iterator, Optional

from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TextIteratorStreamer,
)

logger = logging.getLogger()
transformers.logging.set_verbosity_error()

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
print(f"device: {device}")

@dataclass
class ScriptArguments:
    lora_inhibition: Optional[float] = field(default=0.3)
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    adapter_name: Optional[str] = field(
        default="/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/FacebookAI/roberta-large/",
    )
    quantize: Optional[bool] = field(default=False)
    task: Optional[str] = field(
        default="squad_v2",
    )
    output_csv_file: Optional[str] = field(default="/home/kangchen/inhibited_lora/LoRA-LM/Output_PEFT/Llama-2-7b-chat-hf/results.csv")


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# logging.basicConfig(level=logging.DEBUG if script_args.debug else logging.INFO)


io_path = pathlib.Path(script_args.adapter_name)
model_id = script_args.model_name  #  "roberta-base" # "roberta-base" or "roberta-large"
# io_path = io_path / model_id

task = script_args.task  #  "squad_v2"
model = peft.AutoPeftModel.from_pretrained(io_path / f'model_{task}_inh{script_args.lora_inhibition}')
# model = model.merge_and_unload()  # FCC-PATCH: merging drops the InA gate; keep the adapter live


# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # manual overwrite
# print(f"device: {device}")

# %%
tokenizer = AutoTokenizer.from_pretrained(model_id)
select_mode = "Inhibition_" + str(script_args.lora_inhibition)

visualize_file = str(io_path) + "/" + task + "/"


input_text = "I put my red bag in the black bag ."
output_text = "What is the colour of my bag ?"
# attention_scores_plot = torch.zeros(input_length, output_length, layers_num, heads_num)

# question, text = "I put my red bag in the black bag.", "What is the colour of my bag?"
question, text = input_text, output_text
inputs = tokenizer(question, text, return_tensors="pt")

# print('#####################################')
# print(inputs['input_ids'][0])
# print('#####################################')

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)
attention_scores_draw = outputs['attentions']  # Retrieve attention from model outputs

layers_num = len(attention_scores_draw)

attention_scores_plot = torch.zeros(inputs['input_ids'].size()[1], inputs['input_ids'].size()[1], layers_num)


for plot_layer in range(len(attention_scores_draw)):
    attention_heads = torch.squeeze(attention_scores_draw[plot_layer])
    ## Plot Attention Scores
    attention_scores_plot[:, :, plot_layer] = torch.squeeze(torch.mean(attention_heads, 0))

##  *************************************************************
# print('#####################################')
attention_scores_plot = torch.squeeze(attention_scores_plot).detach().numpy()
attention_scores_size = attention_scores_plot.shape
# print(attention_scores_size)

for plot_layer in range(layers_num):
    ## Plot Attention Scores
    # for plot_head in range(heads_num):

    plot_head = "average"
    file_name = "layer_" + str(plot_layer) + "_head_" + str(plot_head)
    path = os.path.join(visualize_file, select_mode)

    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    attention_heads = torch.squeeze(attention_scores_draw[plot_layer])
    attention_heads_size = attention_heads.size()

    text_plot = []
    for i_token in range(len(inputs['input_ids'][0])):
        text_plot.append(tokenizer.decode(inputs['input_ids'][0][i_token]))

    plot_0 = plt
    fig = plot_0.figure()
    imgplot = plot_0.imshow(attention_scores_plot[:, :, plot_layer], cmap='BuGn')  #  , vmin=-1.0, vmax=5.0
    plot_0.xticks(np.arange(0, len(text_plot)), text_plot, rotation='vertical')
    plot_0.yticks(np.arange(0, len(text_plot)), text_plot, rotation='horizontal')
    plot_0.colorbar(orientation='vertical')
    plot_0.show()
    save_file = path + '/' + file_name + '.png'
    # print(save_file)
    plot_0.savefig(save_file)
    plot_0.close()
    # print('#####################################')


answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
output_text = tokenizer.decode(predict_answer_tokens)
print(output_text)

## *********************************************
## Plot Loss
# loss = outputs.loss
# print(loss)

