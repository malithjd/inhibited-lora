#!/usr/bin/env python
"""Apply the FCC reproduction patches to an inhibited-lora checkout (checked against commit d1d6685).

usage:  python apply_fcc_patches.py ~/inhibited-lora

Every edit is an exact text replacement with an expected match count. If the repo
has changed and a count is off, the script stops before touching that file.
Running it twice is safe: files already carrying the FCC-PATCH marker are skipped.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser().resolve()
MARK = "FCC-PATCH"


def patch(rel, edits):
    path = ROOT / rel
    text = path.read_text()
    if MARK in text:
        print(f"skip (already patched): {rel}")
        return
    for old, new, count in edits:
        found = text.count(old)
        if found != count:
            sys.exit(f"STOP {rel}: expected {count} match(es) of {old[:60]!r}, found {found}. Repo version differs.")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched: {rel} ({len(edits)} edits)")


# 1) run_glue.py / run_qa.py: choose target modules + extra trainable modules from the command line,
#    and load an existing adapter for evaluation-only runs.
NEW_ARGS = '''    target_modules: str = field(
        default="query,key",
        metadata={"help": "FCC-PATCH: comma-separated modules to adapt: query,key (paper) | query,key,value (released adapters) | query_proj,key_proj (DeBERTa)"},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "FCC-PATCH: folder of a trained adapter to load instead of creating a new one (use with --do_eval only)"},
    )
    lora_r: int = field(
        default=16,'''

OLD_CFG = '''        target_modules=["query", "key"]
        # modules_to_save=model_args.modules_to_save,
    )
    model = get_peft_model(model_lm, peft_config)'''

NEW_CFG = '''        target_modules=[m.strip() for m in peft_args.target_modules.split(",") if m.strip()],  # FCC-PATCH
        modules_to_save=[m.strip() for m in peft_args.modules_to_save.split(",")] if peft_args.modules_to_save else None,  # FCC-PATCH
    )
    if peft_args.adapter_path:  # FCC-PATCH: evaluate a released / earlier adapter as-is (gate stays active)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model_lm, peft_args.adapter_path)
        print(f"FCC-PATCH loaded adapter {peft_args.adapter_path}: {model.peft_config['default']}")
    else:
        model = get_peft_model(model_lm, peft_config)
        model.print_trainable_parameters()  # FCC-PATCH: log the trainable-parameter count'''

for script in ["transformers-4.53.0/examples/pytorch/text-classification/run_glue.py",
               "transformers-4.53.0/examples/pytorch/question-answering/run_qa.py"]:
    patch(script, [
        ("    lora_r: int = field(\n        default=16,", NEW_ARGS, 1),
        (OLD_CFG, NEW_CFG, 1),
    ])

# 2) peft LoRA layer: a negative --lora_inhibition now runs the gate (subtractive inhibition, paper Eq. 10)
#    instead of silently falling back to plain LoRA; the activation is selectable with INA_ACT.
ACT_DEF = '''from .config import LoraConfig

# FCC-PATCH: InA activation. Default = the repo's LeakyReLU(0.1); INA_ACT=gelu gives the paper's stated default.
import os as _os
_INA_ACT = _os.environ.get("INA_ACT", "leaky_relu").lower()


def _ina_act(x):
    if _INA_ACT == "gelu":
        return torch.nn.functional.gelu(x)
    if _INA_ACT == "relu":
        return torch.nn.functional.relu(x)
    return torch.nn.functional.leaky_relu(x, negative_slope=0.1)
'''
patch("peft-0.10.0/src/peft/tuners/lora/layer.py", [
    ("from .config import LoraConfig\n", ACT_DEF, 1),
    ("if lora_inhibition > 0.0:", "if lora_inhibition != 0.0:  # FCC-PATCH: <0 = inhibition, >0 = activation", 5),
    ("torch.nn.functional.leaky_relu(lora_A_temp + out_ * lora_inhibition, negative_slope=0.1)",
     "_ina_act(lora_A_temp + out_ * lora_inhibition)", 3),
    ("torch.nn.functional.leaky_relu(after_A + out_ * lora_inhibition, negative_slope=0.1)",
     "_ina_act(after_A + out_ * lora_inhibition)", 2),
])

# 3) 4-bit / 8-bit path: torch.max(z) without dim returns one tensor, so "out_, inds_ = ..." crashes.
#    Use the per-token max like layer.py, and the same sign + activation, so 4-bit and full precision agree.
patch("peft-0.10.0/src/peft/tuners/lora/bnb.py", [
    ("from .layer import LoraLayer\n", "from .layer import LoraLayer, _ina_act  # FCC-PATCH\n", 1),
    ("out_, inds_ = torch.max(lora_A_temp)\n", "out_, inds_ = torch.max(lora_A_temp, dim=-1)  # FCC-PATCH: per-token max over r\n", 4),
    ("if lora_inhibition > 0.0:", "if lora_inhibition != 0.0:", 4),
    ("torch.nn.functional.leaky_relu(lora_A_temp - out_ * lora_inhibition)",
     "_ina_act(lora_A_temp + out_ * lora_inhibition)", 4),
])

# 4) heatmaps: merge_and_unload() folds only B @ A into W, so the merged model has no InA gate.
patch("visualization/visualize_lm.py", [
    ("model = model.merge_and_unload()",
     "# model = model.merge_and_unload()  # FCC-PATCH: merging drops the InA gate; keep the adapter live", 1),
])


# ---------------------------------------------------------------------------------------------
# v2 (22 Oct 2026): make the sign of --lora_inhibition select one exact version of the authors' code.
#   inhibition < 0 -> commit 59021f2 (6 Aug 2025): LeakyReLU(z - |Inh| * max z), slope 0.01.
#                     The released GLUE adapters (uploaded 8 Aug 2025) were trained with this.
#   inhibition > 0 -> commit 4409cbb (11 Aug 2025, still current): LeakyReLU(z + Inh * max z), slope 0.1.
# Runs on files that already carry the v1 patch; safe to re-run.
MARK2 = "FCC-PATCH-v2"


def patch_v2(rel, edits):
    path = ROOT / rel
    text = path.read_text()
    if MARK2 in text:
        print(f"skip (v2 already applied): {rel}")
        return
    for old, new, count in edits:
        found = text.count(old)
        if found != count:
            sys.exit(f"STOP {rel}: expected {count} match(es) of {old[:60]!r}, found {found}.")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched v2: {rel} ({len(edits)} edits)")


patch_v2("peft-0.10.0/src/peft/tuners/lora/layer.py", [
    ("def _ina_act(x):\n",
     "def _ina_act(x, inhibition=1.0):  # FCC-PATCH-v2: sign of inhibition picks the code version\n", 1),
    ("    return torch.nn.functional.leaky_relu(x, negative_slope=0.1)\n",
     "    # <0: commit 59021f2 (6 Aug 2025, released adapters) used F.leaky_relu's default slope 0.01\n"
     "    # >0: commit 4409cbb (11 Aug 2025, current) uses slope 0.1\n"
     "    return torch.nn.functional.leaky_relu(x, negative_slope=0.01 if inhibition < 0 else 0.1)\n", 1),
    ("_ina_act(lora_A_temp + out_ * lora_inhibition)", "_ina_act(lora_A_temp + out_ * lora_inhibition, lora_inhibition)", 3),
    ("_ina_act(after_A + out_ * lora_inhibition)", "_ina_act(after_A + out_ * lora_inhibition, lora_inhibition)", 2),
])
patch_v2("peft-0.10.0/src/peft/tuners/lora/bnb.py", [
    ("from .layer import LoraLayer, _ina_act  # FCC-PATCH\n", "from .layer import LoraLayer, _ina_act  # FCC-PATCH FCC-PATCH-v2\n", 1),
    ("_ina_act(lora_A_temp + out_ * lora_inhibition)", "_ina_act(lora_A_temp + out_ * lora_inhibition, lora_inhibition)", 4),
])

print("all patches applied")
