#!/usr/bin/env python
"""One table of every finished run, next to the released adapters and the paper.

usage:
  python collect_results.py $RUNS                       # everything under ina_runs
  python collect_results.py $RUNS --released $REPO/output_final/BERT_large
  python collect_results.py $RUNS/level3 --csv level3.csv
  python collect_results.py $RUNS --markdown            # pipe table to paste into notes

A run = a folder holding eval_results.json. Folders named seed<k> are pooled into mean ± std.
Columns
  yours     final dev score (x100); mean ± std (n) over seeds
  best_ep   best per-epoch dev score (only with --eval_strategy epoch)
  EM/mm     SQuAD exact match, or MNLI mismatched accuracy
  released  score of the authors' released adapter for the same task + level
  paper     Kang et al. 2024, Table 6, same model + task + inhibition level
  diff      yours - paper (primary metric)
"""
import argparse, csv, json, math, pathlib, re, signal, statistics as st

signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet exit when piped into head / less

# the paper's metric per task: CoLA MCC, STS-B Pearson, other GLUE accuracy, SQuAD F1 (+ EM)
PRIMARY = [("eval_matthews_correlation", "MCC"), ("eval_pearson", "Pearson"),
           ("eval_accuracy", "Acc"), ("eval_f1", "F1")]

# Paper Table 6 (GLUE dev, SQuAD dev F1/EM) by model and inhibition level.
_T6 = {
    "BERT": {
        0.0: "65.5 91.5 86.6 93.9 88.7 92.5 66.4 85.0 91.1/84.3 81.6/78.9",
        0.1: "65.8 91.4 86.5 93.5 88.9 92.4 70.1 83.1 91.1/84.4 81.3/78.5",
        0.3: "65.9 91.5 86.3 94.4 89.0 92.7 69.0 84.8 91.1/84.4 81.4/78.1",
        0.9: "64.3 91.4 86.3 93.3 88.3 92.4 71.1 84.3 91.3/84.6 81.5/78.1"},
    "RoBERTa": {
        0.0: "64.1 92.2 90.2 95.8 92.0 94.1 85.2 89.0 93.9/88.4 88.3/84.7",
        0.1: "65.5 92.0 89.5 95.6 92.4 94.4 83.4 91.7 94.1/88.8 88.5/85.5",
        0.3: "68.5 92.2 90.2 96.4 92.0 94.4 85.2 90.8 94.2/88.8 88.7/85.3",
        0.9: "67.5 92.1 89.6 95.8 91.6 94.1 85.2 89.7 94.7/89.2 89.1/86.3"},
    "DeBERTaV3": {
        0.0: "73.2 93.1 90.9 96.6 93.2 95.5 90.3 91.4 95.2/89.7 90.8/88.5",
        0.1: "76.5 93.2 90.8 96.2 93.2 96.0 90.0 92.3 95.3/89.9 91.2/88.7",
        0.3: "76.4 93.2 90.9 96.6 93.2 96.1 90.7 93.1 95.4/89.9 91.1/88.4",
        0.9: "72.8 93.0 90.9 96.2 92.6 95.5 89.5 90.7 95.4/90.0 91.6/89.0"},
}
_COLS = ["cola", "qqp", "mnli", "sst2", "stsb", "qnli", "rte", "mrpc", "squad", "squad_v2"]
PAPER = {m: {lvl: dict(zip(_COLS, row.split())) for lvl, row in levels.items()} for m, levels in _T6.items()}
TASKS = set(_COLS)


def pct(v):
    """Scores arrive as fractions (GLUE: 0.856) or percents (SQuAD: 85.45); report percents."""
    return v if abs(v) > 1.0 else 100 * v


def primary(d):
    for k, label in PRIMARY:
        if k in d:
            return k, label
    return None, None


def load_run(folder):
    ev = json.loads((folder / "eval_results.json").read_text())
    key, label = primary(ev)
    if key is None:
        return None
    row = {"metric": label, "final": pct(ev[key])}
    for em_key in ("eval_exact_match", "eval_exact"):   # SQuAD v1.1 / v2.0 name EM differently
        if em_key in ev:
            row["EM"] = pct(ev[em_key])
            break
    if "eval_accuracy_mm" in ev:
        row["mm"] = pct(ev["eval_accuracy_mm"])
    ts = folder / "trainer_state.json"
    if ts.exists():
        hist = [h[key] for h in json.loads(ts.read_text()).get("log_history", []) if key in h]
        if hist:
            row["best_ep"] = pct(max(hist))
    return row


def describe(parts):
    """model, task, inhibition level read from the run's folder names."""
    low = [p.lower() for p in parts]
    model = "RoBERTa" if any("roberta" in p for p in low) else "DeBERTaV3" if any("deberta" in p for p in low) else "BERT"
    task = next((p.replace("-", "") for p in reversed(low) if p.replace("-", "") in TASKS), None)
    inh = next((int(m.group(1)) / 100 for p in reversed(parts) for m in [re.match(r"InA(\d\d)", p)] if m), None)
    return model, task, inh


def paper_score(model, task, inh):
    cell = PAPER.get(model, {}).get(inh, {}).get(task)
    if cell is None:
        return None, ""
    return float(cell.split("/")[0]), cell.replace("/", " / ")


def stats(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None, ""
    if len(vals) == 1:
        return vals[0], f"{vals[0]:.1f}"
    return st.mean(vals), f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f} (n={len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--released", help="output_final/BERT_large, to add the released score as a column")
    ap.add_argument("--csv")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown pipe table instead of aligned text")
    a = ap.parse_args()

    groups = {}
    roots = [pathlib.Path(p).expanduser() for p in a.roots]
    for root in roots:
        for f in sorted(root.rglob("eval_results.json")):
            run = f.parent
            if run.name.startswith("checkpoint-"):
                continue
            rel = run.relative_to(root).parts
            key = tuple(p for p in rel if not p.startswith("seed"))
            r = load_run(run)
            if r:
                prefix = (root.name,) if len(roots) > 1 else ()   # name the root only when several are given
                groups.setdefault(prefix + key, []).append(r)

    released = {}
    if a.released:
        rel_root = pathlib.Path(a.released).expanduser()
        for f in rel_root.rglob("eval_results.json"):
            r = load_run(f.parent)
            if r:
                released[f.parent.relative_to(rel_root).parts] = r["final"]

    rows = []
    for key, runs in groups.items():
        model, task, inh = describe(key)
        mean, yours = stats([r["final"] for r in runs])
        p_val, p_txt = paper_score(model, task, inh)
        rows.append({
            "_sort": (model, _COLS.index(task) if task in TASKS else 99, inh if inh is not None else 9, key),
            "run": "/".join(key),
            "metric": runs[0]["metric"],
            "yours": yours,
            "best_ep": stats([r.get("best_ep") for r in runs])[1],
            "EM/mm": stats([r.get("EM", r.get("mm")) for r in runs])[1],
            "released": f"{released[key[-2:]]:.1f}" if key[-2:] in released else "",
            "paper": p_txt,
            "diff": f"{mean - p_val:+.1f}" if (mean is not None and p_val is not None) else "",
        })
    rows.sort(key=lambda r: r["_sort"])

    cols = ["run", "metric", "yours", "best_ep", "EM/mm", "released", "paper", "diff"]
    if a.markdown:
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(r[c] for c in cols) + " |")
    else:
        w = {c: max(len(c), *(len(r[c]) for r in rows)) if rows else len(c) for c in cols}
        print("  ".join(c.ljust(w[c]) for c in cols))
        print("  ".join("-" * w[c] for c in cols))
        last = None
        for r in rows:
            group = r["_sort"][:3]
            if last is not None and group[:2] != last[:2]:
                print()                                   # blank line between tasks
            print("  ".join(r[c].ljust(w[c]) for c in cols))
            last = group
    print("\npaper = Kang et al. 2024, Table 6 (same model, task and inhibition level); diff = yours - paper")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
