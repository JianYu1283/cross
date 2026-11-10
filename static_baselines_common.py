"""Run Slither + SmartCheck on the EXACT 199-instance HGT test set (M4).

The HGT/GAT experiments subsample negatives via np.random.RandomState(0).choice
over the cache's 469 graph-buildable clean contracts (69 pos + 130 neg = 199),
whereas the earlier static runs used random.Random(0).sample over the full 2742
sorted glob. To make the static-vs-learned comparison apples-to-apples, this
script re-runs both static tools on the identical 199-instance list used by
experiments.py, so all methods share one test set.

Writes results/slither_common.json and results/smartcheck_common.json.
"""
import os
import sys
import json
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from slither_baseline import run_slither  # noqa: E402
from smartcheck_baseline import run_smartcheck, is_reentrancy  # noqa: E402


def smartcheck_pred(fpath):
    """run_smartcheck returns (counts, ok); map to bool/None like run_slither."""
    counts, ok = run_smartcheck(fpath)
    if not ok or counts is None:
        return None
    return 1 if is_reentrancy(counts) else 0

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "cache_full.pkl")
OLD = "data/"
NEW = os.path.join(ROOT, "data_src", "ge-sc-master", "data")


def common_items(subsample_neg=130, seed=0):
    cache = pickle.load(open(CACHE, "rb"))
    pos, neg = [], []
    for k in cache.keys():
        if "solidifi_buggy_contracts/reentrancy" in k or "smartbug-dataset/reentrancy" in k:
            pos.append((k, 1))
        elif "smartbugs-wild-clean-contracts" in k:
            neg.append((k, 0))
    if subsample_neg and len(neg) > subsample_neg:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(neg), subsample_neg, replace=False)
        neg = [neg[i] for i in idx]
    items = []
    for old, lab in pos + neg:
        rel = old[len(OLD):].replace("\\", "/")
        items.append((os.path.join(NEW, rel), lab))
    return items


def evaluate(tool_name, runner, items):
    tp = fp = fn = 0
    analyzed = failed = 0
    results = []
    for i, (fpath, label) in enumerate(items):
        p = runner(fpath)
        if p is None:
            failed += 1
            results.append({"file": fpath, "label": label, "pred": None})
        else:
            analyzed += 1
            pred = int(p)
            results.append({"file": fpath, "label": label, "pred": pred})
            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 1 and pred == 0:
                fn += 1
        if (i + 1) % 20 == 0:
            print(f"  [{tool_name}] {i+1}/{len(items)} analyzed={analyzed} failed={failed}", flush=True)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    summary = {"n_total": len(items), "n_analyzed": analyzed, "n_failed": failed,
               "tp": tp, "fp": fp, "fn": fn,
               "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    print(f"RESULT {tool_name} common: analyzed={analyzed} failed={failed} "
          f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} (tp={tp} fp={fp} fn={fn})", flush=True)
    with open(os.path.join(ROOT, "results", f"{tool_name}_common.json"), "w") as f:
        json.dump({**summary, "results": results}, f)


def main():
    subsample_neg = 130
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        subsample_neg = 0
    items = common_items(subsample_neg=subsample_neg)
    n_pos = sum(1 for _, l in items if l == 1)
    print(f"test set: {len(items)} items ({n_pos} pos / {len(items)-n_pos} neg)", flush=True)
    evaluate("slither", run_slither, items)
    evaluate("smartcheck", smartcheck_pred, items)


if __name__ == "__main__":
    main()
