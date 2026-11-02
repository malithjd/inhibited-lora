#!/usr/bin/env python
"""Add --task_dropout to run_glue.py: the dropout before the classifier head ("Dropout of task layer",
paper Table 1: 0.15). Without the flag nothing changes. Safe to run twice.

usage: python patch_task_dropout.py <repo folder>      e.g. ~/inhibited-lora
"""
import pathlib, sys

repo = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
p = repo / "transformers-4.53.0" / "examples" / "pytorch" / "text-classification" / "run_glue.py"
s = p.read_text()
if "task_dropout" in s:
    sys.exit(f"already patched: {p}")

field_anchor = "    lora_inhibition: Optional[float] = field("
assert s.count(field_anchor) == 1, "run_glue.py differs from the expected version"
s = s.replace(field_anchor, '''    task_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "FCC-PATCH: dropout before the classifier head (paper Table 1: 0.15); default = the model's own"},
    )
''' + field_anchor, 1)

i = s.index("    config = AutoConfig.from_pretrained(")
j = s.index("\n    )\n", i) + len("\n    )\n")
s = s[:j] + '''    if peft_args.task_dropout is not None:  # FCC-PATCH: paper Table 1 "Dropout of task layer"
        config.classifier_dropout = peft_args.task_dropout  # BERT and RoBERTa heads
        config.cls_dropout = peft_args.task_dropout  # DeBERTa-v2/v3 head
''' + s[j:]
p.write_text(s)
print(f"patched: {p}")
