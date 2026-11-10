"""Comment 2: source-grouped (leave-one-source-out) evaluation.

The main 0.951 result is random stratified 5-fold CV, which can leak sibling
contracts (same SolidiFI injection template / same SmartBugs project) across
folds. We cannot recover the finer base-contract grouping from the SolidiFI
filenames (buggy_N.sol), so the coarsest reliable group is the *source*:
SolidiFI positives, SmartBugs positives, and SmartBugs-wild negatives.

This script computes the three source-grouped cells (leave-one-source-out):
  1. sf_to_sb   : train SolidiFI pos + 130 neg(seed0) -> test SmartBugs pos + 130 neg(seed1)
  2. sb_to_sf   : train SmartBugs pos + 130 neg(seed0) -> test SolidiFI pos + 130 neg(seed1)
  3. wild_holdout: train all 69 pos + 130 neg(seed0)  -> test a disjoint 130 neg(seed1)

Cell 3 reports clean-contract generalization (negative-only test set).
The positive-source transfer F1 (mean of cells 1-2) is the leakage-robust
estimate of positive generalization, to contrast with the random-stratified 0.951.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import HGTClassifier
from experiments import load_items, make_batch

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "cache_full.pkl")


def is_sf(k):
    return "solidifi_buggy_contracts" in str(k).lower()


def is_sb(k):
    return "smartbug-dataset" in str(k).lower()


def neg_items(seed, n=130):
    items, _ = load_items(CACHE, None, 0)
    negs = [(k, 0) for k, l in items if l == 0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(negs), n, replace=False)
    return [negs[i] for i in idx]


def train_eval(tr_items, te_items, cache, lowcall, mode, epochs=50, seed=0):
    tg, tl = make_batch(cache, tr_items, lowcall, mode, True, True)
    vg, vl = make_batch(cache, te_items, lowcall, mode, True, True)
    dim = tg[0].ndata["feat"].shape[1]
    torch.manual_seed(seed); np.random.seed(seed)
    m = HGTClassifier(dim, 64, 2, 4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tr_g = dgl.batch(tg)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(tr_g), tl)
        loss.backward(); opt.step()
    m.eval()
    preds = []
    with torch.no_grad():
        for g in vg:
            preds.append(m(g).argmax(1).item())
    truths = vl.tolist()
    p = np.array(preds); t = np.array(truths)
    return dict(acc=float(accuracy_score(t, p)),
                n_test=int(len(t)), n_pos_test=int((t == 1).sum()),
                n_neg_test=int((t == 0).sum()),
                # positive-side metrics (undefined if no positives)
                precision=float(precision_score(t, p, zero_division=0)),
                recall=float(recall_score(t, p, zero_division=0)),
                f1=float(f1_score(t, p, zero_division=0)),
                # negative-side (specificity)
                neg_acc=float(accuracy_score(1 - t, 1 - p)))


def main():
    items, cache = load_items(CACHE, None, 0)
    sf_pos = [(k, 1) for k, l in items if l == 1 and is_sf(k)]
    sb_pos = [(k, 1) for k, l in items if l == 1 and is_sb(k)]
    all_pos = sf_pos + sb_pos

    out = {}
    for mode, lowcall in (("rich", 1), ("sparse", 1), ("rich", 0), ("sparse", 0)):
        cells = {}
        # cell 1: sf -> sb
        t0 = time.time()
        cells["sf_to_sb"] = train_eval(sf_pos + neg_items(0), sb_pos + neg_items(1),
                                       cache, lowcall, mode)
        cells["sf_to_sb"]["wall_sec"] = round(time.time() - t0, 1)
        # cell 2: sb -> sf
        t0 = time.time()
        cells["sb_to_sf"] = train_eval(sb_pos + neg_items(0), sf_pos + neg_items(1),
                                       cache, lowcall, mode)
        cells["sb_to_sf"]["wall_sec"] = round(time.time() - t0, 1)
        # cell 3: all pos -> disjoint wild negatives
        t0 = time.time()
        cells["wild_holdout"] = train_eval(all_pos + neg_items(0), neg_items(1),
                                           cache, lowcall, mode)
        cells["wild_holdout"]["wall_sec"] = round(time.time() - t0, 1)
        cells["pos_transfer_f1"] = round((cells["sf_to_sb"]["f1"] +
                                          cells["sb_to_sf"]["f1"]) / 2.0, 4)
        out[f"{mode}_low{lowcall}"] = cells
        print(f"RESULT {mode}_low{lowcall}: "
              f"sf_to_sb F1={cells['sf_to_sb']['f1']:.3f} "
              f"sb_to_sf F1={cells['sb_to_sf']['f1']:.3f} "
              f"wild_neg_acc={cells['wild_holdout']['neg_acc']:.3f} "
              f"pos_transfer_F1={cells['pos_transfer_f1']}", flush=True)

    json.dump(out, open(os.path.join(ROOT, "results", "grouped_cv.json"), "w"))
    print("saved results/grouped_cv.json")


if __name__ == "__main__":
    main()
