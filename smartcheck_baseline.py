"""SmartCheck baseline on the same benchmark test set (M4).

SmartCheck (v2.0.3, smartdec/smartcheck) is a 2018-era XPath/ANTLR static
analyser. It has NO dedicated "reentrancy" rule; its reentrancy-relevant
signals are the low-level-call family:
  - SOLIDITY_UNCHECKED_CALL   (call/delegatecall/send/callcode with no return check)
  - SOLIDITY_CALL_WITHOUT_DATA (call.value(...)("")  / call() with empty data)
We therefore define the "reentrancy proxy" as firing either rule, and record
per-rule counts so the Discussion can be precise. Contracts that SmartCheck
cannot parse at all (no summary block) are excluded from the denominators,
mirroring the Slither and graph-construction disclosures.
"""
import os
import sys
import json
import glob
import random
import subprocess
import re

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data_src", "ge-sc-master", "data")
POS_DIRS = [
    os.path.join(DATA, "solidifi_buggy_contracts", "reentrancy"),
    os.path.join(DATA, "smartbug-dataset", "reentrancy"),
]
NEG_DIR = os.path.join(DATA, "smartbugs-wild-clean-contracts")

# jdeploy launcher of the npm-installed SmartCheck (runs the bundled jar with `java`)
JDEPLOY = os.path.join(
    ROOT, "tools", "smartcheck-install", "node_modules",
    "@smartdec", "smartcheck", "jdeploy-bundle", "jdeploy.js",
)
OUT = os.path.join(ROOT, "results", "smartcheck_baseline.json")

# Rules that jointly signal a reentrancy-prone low-level call pattern.
REENTRANCY_RULES = ("SOLIDITY_UNCHECKED_CALL", "SOLIDITY_CALL_WITHOUT_DATA")
_SUMMARY_RE = re.compile(r"^(SOLIDITY_[A-Z_0-9]+|VYPER_[A-Z_0-9]+)\s*:(\d+)\s*$")


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


def run_smartcheck(fpath):
    """Return (counts_dict, ok) where counts_dict maps rule -> int.
    ok=False means SmartCheck produced no parseable summary (hard failure)."""
    try:
        p = subprocess.run(
            ["node", JDEPLOY, "-p", fpath],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "NO_PROXY": "*", "no_proxy": "*"},
        )
    except Exception:
        return None, False
    counts = {}
    for line in (p.stdout or "").splitlines():
        m = _SUMMARY_RE.match(line.strip())
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + int(m.group(2))
    # A hard failure = no summary block at all. Empty summary (clean file) is valid.
    return counts, bool(counts or _SUMMARY_RE.search(p.stdout or ""))


def is_reentrancy(counts):
    return any(counts.get(r, 0) > 0 for r in REENTRANCY_RULES)


def main():
    items = collect()
    n_pos = sum(1 for _, l in items if l == 1)
    n_neg = sum(1 for _, l in items if l == 0)
    print(f"analyzing {len(items)} contracts ({n_pos} pos / {n_neg} neg)", flush=True)

    preds, truths = [], []
    analyzed, failed = 0, 0
    results = []
    rule_totals = {}
    for i, (fpath, label) in enumerate(items):
        counts, ok = run_smartcheck(fpath)
        if not ok or counts is None:
            failed += 1
            results.append({"file": fpath, "label": label, "pred": None, "rules": {}})
        else:
            analyzed += 1
            preds.append(1 if is_reentrancy(counts) else 0)
            truths.append(label)
            for r, c in counts.items():
                rule_totals[r] = rule_totals.get(r, 0) + c
            results.append({"file": fpath, "label": label,
                            "pred": int(is_reentrancy(counts)), "rules": counts})
        if (i + 1) % 20 == 0:
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
               "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
               "reentrancy_rules": list(REENTRANCY_RULES),
               "rule_totals": rule_totals}
    print(f"RESULT smartcheck analyzed={analyzed} failed={failed} "
          f"P={prec:.4f} R={rec:.4f} F1={f1:.4f}", flush=True)
    print("rule totals:", json.dumps(rule_totals, indent=2), flush=True)
    with open(OUT, "w") as f:
        json.dump({**summary, "results": results}, f)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
