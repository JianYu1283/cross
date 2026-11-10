"""Item 3: per-fold F1 dump + contract-clustered bootstrap CIs (no retraining).

Reconstructs the per-contract (fold,seed) predictions from the saved pooled
preds, then runs a cluster bootstrap that resamples CONTRACTS (the independent
unit) rather than fold x seed observations, to answer reviewer #4.
"""
import os
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "data", "cache_full.pkl")


def load_items(subsample_neg=130, seed=0):
    """Exact copy of experiments.load_items (what generated the GNN pooled preds)."""
    cache = pickle.load(open(CACHE, "rb"))
    pos, neg = [], []
    for k in cache.keys():
        s = str(k).lower()
        if ("solidifi_buggy_contracts" in s) or ("smartbug-dataset" in s):
            pos.append((k, 1))
        elif "smartbugs-wild-clean-contracts" in s:
            neg.append((k, 0))
    if subsample_neg and len(neg) > subsample_neg:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(neg), subsample_neg, replace=False)
        neg = [neg[i] for i in idx]
    return pos + neg


def load_json(tag):
    with open(os.path.join(RES, f"full_{tag}.json")) as f:
        return json.load(f)


def reconstruct_per_contract(pooled, labels, n_seeds=5, n_folds=5):
    """Map pooled[seed]['preds'] back to per-contract per-seed predictions."""
    n = len(labels)
    out = {i: {} for i in range(n)}
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        te_flat = []
        for _, te in skf.split(np.arange(n), labels):
            te_flat.extend(te.tolist())
        pred = pooled[s]["preds"]
        assert len(pred) == len(te_flat), f"seed {s}: {len(pred)} != {len(te_flat)}"
        for j, item_i in enumerate(te_flat):
            out[item_i][s] = pred[j]
    return out


def cluster_bootstrap(predsA, predsB, labels, n_iter=10000, seed=0):
    """Resample CONTRACTS (not fold x seed) -> 95% CI + two-sided sign-flip p."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    item_idx = np.arange(n)
    diffs = []
    for _ in range(n_iter):
        sample = rng.choice(item_idx, n, replace=True)
        pa, pb, tt = [], [], []
        for i in sample:
            for s in range(5):
                pa.append(predsA[i][s]); pb.append(predsB[i][s]); tt.append(labels[i])
        f1a = f1_score(tt, pa, zero_division=0)
        f1b = f1_score(tt, pb, zero_division=0)
        diffs.append(f1a - f1b)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return lo, hi, p


def report(ta, tb):
    A = load_json(ta); B = load_json(tb)
    labels = np.array([lab for _, lab in load_items()])
    pA = reconstruct_per_contract(A["pooled"], labels)
    pB = reconstruct_per_contract(B["pooled"], labels)
    lo, hi, p = cluster_bootstrap(pA, pB, labels)
    fd = np.mean(A["per_fold_f1"]) - np.mean(B["per_fold_f1"])
    print(f"{ta:22s} vs {tb:22s}: foldxseed_diff={fd:+.4f} "
          f"cluster-boot 95%CI [{lo:+.4f},{hi:+.4f}] signflip_p={p:.4f}")


if __name__ == "__main__":
    print("=== per-fold F1 (25 fold x seed values) ===")
    for tag in ("hgt_rich_low1", "hgt_rich_low0", "gat_rich_low1"):
        d = load_json(tag)
        vals = np.array(d["per_fold_f1"])
        print(f"{tag:20s} min={vals.min():.3f} q1={np.percentile(vals,25):.3f} "
              f"med={np.median(vals):.3f} q3={np.percentile(vals,75):.3f} "
              f"max={vals.max():.3f}  mean={vals.mean():.4f} std={vals.std():.4f}")

    print("\n=== contract-clustered bootstrap (resample contracts) ===")
    report("hgt_rich_low1", "hgt_rich_low0")
    report("hgt_rich_low1", "gat_rich_low1")
    report("hgt_sparse_low1", "gat_sparse_low1")
    report("hgt_rich_low1", "graphcodebert_common")
