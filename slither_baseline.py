"""Slither reentrancy baseline on the same benchmark test set (M4).

Runs Slither's reentrancy detectors on the 81 positives + 130 sampled clean
contracts (identical sampling to codebert_baseline.py, seed 0) and reports
precision / recall / F1. Contracts Slither cannot compile are counted and
excluded from the metric denominators, matching the paper's "graph
construction succeeds for 162 of 211" disclosure.
"""
import os
import sys
import json
import glob
import random
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

DETECTORS = "reentrancy-eth,reentrancy-no-eth"  # serious reentrancy only
SOLC_DIR = os.path.expanduser("~/.solc-select/artifacts")
_SOLC_ORDER = ["0.5.17", "0.4.26"]  # most common; try these first


def _solc_candidates():
    cands = []
    for v in _SOLC_ORDER:
        p = os.path.join(SOLC_DIR, "solc-" + v, "solc-" + v)
        if os.path.exists(p):
            cands.append(p)
    for d in sorted(os.listdir(SOLC_DIR)):
        if d.startswith("solc-"):
            v = d[len("solc-"):]
            p = os.path.join(SOLC_DIR, d, d)
            if v not in _SOLC_ORDER and os.path.exists(p):
                cands.append(p)
    return cands

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data_src", "ge-sc-master", "data")
POS_DIRS = [
    os.path.join(DATA, "solidifi_buggy_contracts", "reentrancy"),
    os.path.join(DATA, "smartbug-dataset", "reentrancy"),
]
NEG_DIR = os.path.join(DATA, "smartbugs-wild-clean-contracts")
SLITHER = os.path.join(sys.prefix, "Scripts", "slither.exe")
OUT = os.path.join(ROOT, "results", "slither_baseline.json")


def collect(seed=0, limit=130):
    rng = random.Random(seed)
    items = []
    for d in POS_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.sol"))):
            items.append((f, 1))
    neg = [(f, 0) for f in sorted(glob.glob(os.path.join(NEG_DIR, "*.sol")))]
    neg = rng.sample(neg, min(limit, len(neg)))
    items += neg
    rng.shuffle(items)
    return items


def run_slither(fpath):
    out_json = os.path.join(ROOT, "results", "_slither_tmp.json")
    for solc in _solc_candidates():
        try:
            if os.path.exists(out_json):
                os.remove(out_json)  # avoid reading a stale result
            subprocess.run(
                [SLITHER, fpath, "--detect", DETECTORS, "--solc", solc,
                 "--json", out_json],
                capture_output=True, text=True, timeout=300,
            )
            # Slither's Windows console script exits -1 (0xFFFFFFFF) even on a
            # successful run, so trust the JSON (`success` field) over the code.
            if not os.path.exists(out_json):
                continue  # compile/version failure -> try next solc version
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("success", True):
                continue  # analysis reported failure -> try next solc version
            dets = data.get("results", {}).get("detectors", [])
            return any(d.get("check", "").startswith("reentrancy") for d in dets)
        except Exception:
            continue
    return None


def main():
    items = collect()
    n_pos = sum(1 for _, l in items if l == 1)
    n_neg = sum(1 for _, l in items if l == 0)
    print(f"analyzing {len(items)} contracts ({n_pos} pos / {n_neg} neg)", flush=True)

    preds, truths, analyzed, failed = [], [], 0, 0
    results = []  # incremental save
    for i, (fpath, label) in enumerate(items):
        p = run_slither(fpath)
        if p is None:
            failed += 1
            results.append({"file": fpath, "label": label, "pred": None})
        else:
            analyzed += 1
            preds.append(int(p))
            truths.append(label)
            results.append({"file": fpath, "label": label, "pred": int(p)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)} analyzed={analyzed} failed={failed}", flush=True)
            with open(OUT, "w") as f:
                json.dump({"analyzed": analyzed, "failed": failed,
                           "results": results}, f)

    tp = sum(1 for t, p in zip(truths, preds) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truths, preds) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truths, preds) if t == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    summary = {"n_total": len(items), "n_analyzed": analyzed, "n_failed": failed,
               "tp": tp, "fp": fp, "fn": fn,
               "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    print(f"RESULT slither analyzed={analyzed} failed={failed} "
          f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}", flush=True)
    with open(OUT, "w") as f:
        json.dump({**summary, "results": results}, f)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
