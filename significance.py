"""Paired significance tests + bootstrap CIs over the GNN experiment JSONs.

For each config pair (same graph set, same fold/seed splits) report:
  - paired Wilcoxon signed-rank p on per-(fold,seed) F1
  - paired t-test p
  - bootstrap 95% CI on the F1 difference (mean_a - mean_b)

Reads results/full_<model>_<mode>_low<lowcall>.json produced by experiments.py.
"""
import os
import json
import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def load(tag):
    p = os.path.join(RES, f"full_{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def bootstrap_ci(a, b, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.array(a) - np.array(b)
    diffs = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return np.percentile(diffs, [2.5, 97.5])


def compare(ta, tb):
    a, b = load(ta), load(tb)
    if a is None or b is None:
        print(f"  SKIP {ta} vs {tb} (missing)")
        return
    fa = np.array(a["per_fold_f1"]); fb = np.array(b["per_fold_f1"])
    w = stats.wilcoxon(fa, fb, zero_method="zsplit")
    t = stats.ttest_rel(fa, fb)
    lo, hi = bootstrap_ci(fa, fb)
    print(f"  {ta} vs {tb}: n={len(fa)} mean_diff={fa.mean()-fb.mean():+.4f} "
          f"Wilcoxon p={w.pvalue:.4f} t-test p={t.pvalue:.4f} "
          f"95%CI [{lo:+.4f},{hi:+.4f}]")


def main():
    print("=== Architecture (HGT vs GAT), paired on fold*seed F1 ===")
    for mode in ("onehot", "sparse", "rich"):
        for low in ("0", "1"):
            compare(f"hgt_{mode}_low{low}", f"gat_{mode}_low{low}")
    print("=== Low-level call (on vs off) ===")
    for model in ("hgt", "gat"):
        for mode in ("onehot", "sparse", "rich"):
            compare(f"{model}_{mode}_low1", f"{model}_{mode}_low0")
    print("=== HGT vs GraphCodeBERT on common 199-instance subset (the 'ties' test) ===")
    compare("hgt_rich_low1", "graphcodebert_common")
    compare("hgt_sparse_low1", "graphcodebert_common")
    compare("hgt_onehot_low1", "graphcodebert_common")


if __name__ == "__main__":
    main()
