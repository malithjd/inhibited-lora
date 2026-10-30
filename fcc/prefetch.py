# Fill $HF_HOME with every model, dataset and metric the runs need (run once on fly, Step 3).
from huggingface_hub import snapshot_download
from datasets import load_dataset
import evaluate

SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "tf_model*", "flax_model*", "rust_model*", "coreml/*"]
MODELS = [
    "bert-large-cased",                # GLUE + SQuAD, and the base of the 28 released adapters
    "FacebookAI/roberta-large",        # paper Tables 3-4
    "microsoft/deberta-v3-large",      # paper Tables 3-4 (needs the Step 4 patch)
    "google-bert/bert-large-uncased",  # base of the released SQuAD-v2 heatmap adapters
]
for m in MODELS:
    print("model", m, snapshot_download(m, ignore_patterns=SKIP))

for task in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]:
    print("glue", task, load_dataset("nyu-mll/glue", task))
    evaluate.load("glue", task)                # caches the metric script too

for name in ["squad", "squad_v2"]:
    print(name, load_dataset(name))
    evaluate.load(name)
for name in ["accuracy", "mse"]:
    evaluate.load(name)
print("done")
