#!/usr/bin/env python
"""Check that this fork's InA layer reproduces the authors' own code, version by version.

usage (CPU is enough, about 1 minute):  python fcc/check_versions.py      # run from the repo root

It builds a tiny BERT with LoRA/InA on query, key, value, gives every version the same LoRA weights,
and compares the classifier logits of:
  this fork (working tree)            peft-0.10.0/src
  upstream commit 59021f2 (Aug 6 2025, the code that trained the released checkpoints)
  upstream commit d1d6685 (current upstream code, this fork's starting point)
Expected: every line SAME, except the two "should differ" lines, which print DIFF.
"""
import copy, os, shutil, subprocess, sys, tempfile

AUG06, UPSTREAM = "59021f2", "d1d6685"


def forward(tag, inhs):
    """Runs inside a child process whose PYTHONPATH points at one peft version."""
    import torch, peft
    from transformers import BertConfig, BertForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType
    torch.manual_seed(0)
    base = BertForSequenceClassification(BertConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
                                                    num_attention_heads=2, intermediate_size=128)).eval()
    torch.manual_seed(7)
    ids = torch.randint(0, 100, (3, 9))
    out = {"peft": peft.__file__}
    for inh in inhs:
        m = get_peft_model(copy.deepcopy(base), LoraConfig(task_type=TaskType.SEQ_CLS, r=4, lora_alpha=16, lora_dropout=0.0,
                           lora_inhibition=inh, target_modules=["query", "key", "value"])).eval()
        torch.manual_seed(1)                                  # identical LoRA weights for every version and level
        for n, p in sorted(m.named_parameters()):
            if "lora_A" in n or "lora_B" in n:
                p.data.normal_(0, 0.5)
        with torch.no_grad():
            out[inh] = m(input_ids=ids).logits
    torch.save(out, tag)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return forward(sys.argv[2], [float(x) for x in sys.argv[3:]])
    import torch
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    tmp = tempfile.mkdtemp()
    srcs = {"fork": os.path.join(repo, "peft-0.10.0", "src")}
    for name, commit in (("aug06", AUG06), ("upstream", UPSTREAM)):
        d = os.path.join(tmp, name); os.makedirs(d)
        archive = subprocess.run(["git", "-C", repo, "archive", commit, "peft-0.10.0/src"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=archive, check=True)
        srcs[name] = os.path.join(d, "peft-0.10.0", "src")
    levels = {"fork": ["-0.3", "-0.9", "0.3", "0.9", "0.0"], "aug06": ["0.3", "0.9", "0.0"], "upstream": ["0.3", "0.9", "-0.3", "0.0"]}
    res = {}
    for name, src in srcs.items():
        f = os.path.join(tmp, name + ".pt")
        env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
        subprocess.run([sys.executable, os.path.abspath(__file__), "--child", f, *levels[name]], env=env, check=True)
        res[name] = torch.load(f)
        print(f"{name:9s} peft from {res[name]['peft']}")
    a, u, k = res["aug06"], res["upstream"], res["fork"]
    same = lambda x, y: torch.allclose(x, y, atol=1e-6)
    print()
    for label, x, y in [
        ("fork -0.3  vs  Aug 6 code +0.3   (shunting inhibition = released checkpoints)", k[-0.3], a[0.3]),
        ("fork -0.9  vs  Aug 6 code +0.9", k[-0.9], a[0.9]),
        ("fork +0.3  vs  upstream +0.3     (shunting activation, unchanged)", k[0.3], u[0.3]),
        ("fork +0.9  vs  upstream +0.9", k[0.9], u[0.9]),
        ("fork 0     vs  upstream 0        (plain LoRA, unchanged)", k[0.0], u[0.0]),
        ("upstream -0.3 vs upstream 0      (upstream silently ignores a negative value)", u[-0.3], u[0.0]),
        ("fork -0.3  vs  fork 0            (should differ: the gate runs)", k[-0.3], k[0.0]),
        ("fork -0.3  vs  fork +0.3         (should differ: two versions)", k[-0.3], k[0.3]),
    ]:
        print(("SAME  " if same(x, y) else "DIFF  ") + label)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
