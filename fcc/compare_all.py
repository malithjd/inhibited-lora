#!/usr/bin/env python
"""Side-by-side comparison: one row per (model, task, inhibition level), one column per run set.

usage:
  python compare_all.py $RUNS --released $REPO/output_final/BERT_large
  python compare_all.py $RUNS --released $REPO/output_final/BERT_large --markdown   # paste into notes/slides
  python compare_all.py $RUNS --released $REPO/output_final/BERT_large --csv all.csv

Columns
  paper      Kang et al. 2024, Table 6 (same model, task, inhibition level)
  released   the authors' released adapter (output_final/BERT_large)
  L1 cur     released adapter scored with the current code        ($RUNS/level1)
  L1 aug6    released adapter scored with the Aug 6, 2025 code    ($RUNS/level1_aug06)
  L2 old     trained with the current code, adds the threshold    ($RUNS/level2, $RUNS/squad)
  L2 new     trained with the Aug 6 code, subtracts the threshold ($RUNS/level2_aug06, $RUNS/squad_aug06)
  new-rel    L2 new minus released;  new-paper  L2 new minus paper
  check      "L1 off" when L1 cur is more than 1 point away from released
At inhibition 0 there is no threshold, so old and new code are the same model: L2 new shows "= old".
SQuAD cells are F1 / EM; every other cell is the paper's metric (CoLA MCC, STS-B Pearson, others accuracy).
"""
import argparse, csv, pathlib, signal, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from collect_results import load_run, describe, paper_score, stats, PAPER, _COLS  # noqa: E402

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

SETS = [("level1", "L1 cur"), ("level1_aug06", "L1 aug6"), ("level2", "L2 old"),
        ("level2_aug06", "L2 new"), ("squad", "L2 old"), ("squad_aug06", "L2 new")]
ORDER = ["cola", "sst2", "mrpc", "stsb", "qnli", "rte", "mnli", "qqp", "squad", "squad_v2"]
MODELS = ["BERT", "RoBERTa", "DeBERTaV3"]
NAMES = {"cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC", "stsb": "STS-B", "qnli": "QNLI", "rte": "RTE",
         "mnli": "MNLI", "qqp": "QQP", "squad": "SQuAD v1.1", "squad_v2": "SQuAD v2.0"}


def cell(runs, squad):
    mean, txt = stats([r["final"] for r in runs])
    if squad and mean is not None:
        em, _ = stats([r.get("EM") for r in runs])
        txt = f"{mean:.1f}/{em:.1f}" if em is not None else txt
    return mean, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", help="$RUNS (the folder holding level1, level2, squad, ...)")
    ap.add_argument("--released", help="$REPO/output_final/BERT_large")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()
    runs_root = pathlib.Path(a.runs).expanduser()

    table = {}   # (model, task, inh) -> {column: [run dicts]}
    for folder, col in SETS:
        root = runs_root / folder
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("eval_results.json")):
            if f.parent.name.startswith("checkpoint-"):
                continue
            r = load_run(f.parent)
            parts = f.parent.relative_to(root).parts
            model, task, inh = describe(parts)
            if r and task is not None and inh is not None:
                table.setdefault((model, task, inh), {}).setdefault(col, []).append(r)

    released = {}
    if a.released:
        rel_root = pathlib.Path(a.released).expanduser()
        for f in rel_root.rglob("eval_results.json"):
            r = load_run(f.parent)
            model, task, inh = describe(f.parent.relative_to(rel_root).parts)
            if r and task is not None and inh is not None:
                released[("BERT", task, inh)] = r["final"]

    cols = ["setting", "metric", "paper", "released", "L1 cur", "L1 aug6", "L2 old", "L2 new",
            "new-rel", "new-paper", "check"]
    rows = []
    for key in sorted(table, key=lambda k: (MODELS.index(k[0]) if k[0] in MODELS else 9,
                                             ORDER.index(k[1]) if k[1] in ORDER else 99, k[2])):
        model, task, inh = key
        sets = table[key]
        squad = task.startswith("squad")
        metric = next(iter(sets.values()))[0]["metric"]
        p_val, p_txt = paper_score(model, task, inh)
        rel = released.get(key)
        vals = {c: cell(sets[c], squad) if c in sets else (None, "") for c in ("L1 cur", "L1 aug6", "L2 old", "L2 new")}
        new = vals["L2 new"][0]
        if inh == 0 and new is None and vals["L2 old"][0] is not None:
            vals["L2 new"] = (None, "= old")
        l1 = vals["L1 cur"][0]
        rows.append({
            "setting": f"{'' if model == 'BERT' else model + ' '}{NAMES.get(task, task)}, {inh:g}",
            "metric": metric,
            "paper": p_txt.replace(" / ", "/"),
            "released": f"{rel:.1f}" if rel is not None else "",
            **{c: vals[c][1] for c in vals},
            "new-rel": f"{new - rel:+.1f}" if (new is not None and rel is not None) else "",
            "new-paper": f"{new - p_val:+.1f}" if (new is not None and p_val is not None) else "",
            "check": "L1 off" if (l1 is not None and rel is not None and abs(l1 - rel) > 1.0) else "",
            "_task": (model, task),
        })

    if a.markdown:
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(r[c] for c in cols) + " |")
    else:
        w = {c: max([len(c)] + [len(r[c]) for r in rows]) for c in cols}
        print("  ".join(c.ljust(w[c]) for c in cols))
        print("  ".join("-" * w[c] for c in cols))
        last = None
        for r in rows:
            if last is not None and r["_task"] != last:
                print()
            print("  ".join(r[c].ljust(w[c]) for c in cols))
            last = r["_task"]
    print("\nL1 = released adapters re-scored; L2 old = current code (adds Th); L2 new = Aug 6 code (subtracts Th)."
          "\npaper = Kang et al. 2024, Table 6; SQuAD cells are F1/EM.")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
