"""Multi-seed cross-dataset transfer (reviewer: single-run is underpowered).

Re-runs the decisive sb_to_sf / sf_to_sb transfer cells with 5 seeds each and
reports mean +/- std F1/recall, so the asymmetry is not carried by one run.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score, precision_score, recall_score

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


def train_eval(tr_items, te_items, cache, mode, lowcall, seed):
    tg, tl = make_batch(cache, tr_items, lowcall, mode, True, True)
    vg, vl = make_batch(cache, te_items, lowcall, mode, True, True)
    dim = tg[0].ndata["feat"].shape[1]
    torch.manual_seed(seed); np.random.seed(seed)
    m = HGTClassifier(dim, 64, 2, 4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tr_g = dgl.batch(tg)
    for _ in range(50):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(tr_g), tl)
        loss.backward(); opt.step()
    m.eval()
    preds = []
    with torch.no_grad():
        for g in vg:
            preds.append(m(g).argmax(1).item())
    t = vl.tolist(); p = preds
    return dict(f1=float(f1_score(t, p, zero_division=0)),
                rec=float(recall_score(t, p, zero_division=0)),
                prec=float(precision_score(t, p, zero_division=0)))


def main():
    items, cache = load_items(CACHE, None, 0)
    sf = [(k, 1) for k, l in items if l == 1 and is_sf(k)]
    sb = [(k, 1) for k, l in items if l == 1 and is_sb(k)]
    out = {}
    for name, tr_pos, te_pos in (("sb_to_sf", sb, sf), ("sf_to_sb", sf, sb)):
        for mode, low in (("sparse", 1), ("rich", 1)):
            vals = []
            for s in range(5):
                tr = tr_pos + neg_items(2 * s)
                te = te_pos + neg_items(2 * s + 1)
                vals.append(train_eval(tr, te, cache, mode, low, s))
            f1 = [v["f1"] for v in vals]
            rec = [v["rec"] for v in vals]
            out[f"{name}_{mode}_low{low}"] = {
                "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)),
                "rec_mean": float(np.mean(rec)), "rec_std": float(np.std(rec)),
                "per_seed": vals}
            print(f"RESULT {name} {mode} low{low}: "
                  f"F1={np.mean(f1):.3f}±{np.std(f1):.3f} rec={np.mean(rec):.3f}±{np.std(rec):.3f} "
                  f"[{', '.join(f'{x:.2f}' for x in f1)}]", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "transfer_multiseed.json"), "w"))
    print("saved results/transfer_multiseed.json")


if __name__ == "__main__":
    main()
