"""Item 2: per-source (SolidiFI / SmartBugs) and cross-dataset transfer evaluation.

Answers reviewer #2's leakage concern: rich features could capture SolidiFI
injection artifacts. If HGT (rich) transfers from SolidiFI (injection-generated)
to SmartBugs (not injection-generated) positives, the signal is real reentrancy,
not template artifacts.

Protocols:
  sf_only   : 5-fold CV on SolidiFI positives + 130 negatives
  sb_only   : 5-fold CV on SmartBugs positives + 130 negatives
  sf_to_sb  : train SolidiFI pos + 130 neg (seed 0) -> test SmartBugs pos + 130 neg (seed 1)
  sb_to_sf  : train SmartBugs pos + 130 neg (seed 0) -> test SolidiFI pos + 130 neg (seed 1)
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F, dgl
from sklearn.model_selection import StratifiedKFold
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


def split_pos(items):
    sf = [(k, 1) for k, l in items if l == 1 and is_sf(k)]
    sb = [(k, 1) for k, l in items if l == 1 and is_sb(k)]
    return sf, sb


def neg_items(subsample_seed, n=130):
    items, _ = load_items(CACHE, None, 0)          # all 69 pos + 469 neg
    negs = [(k, 0) for k, l in items if l == 0]
    rng = np.random.RandomState(subsample_seed)
    idx = rng.choice(len(negs), n, replace=False)
    return [negs[i] for i in idx]


def train_eval(train_items, test_items, epochs=50, seed=0):
    _, cache = load_items(CACHE, None, 0)
    tg, tl = make_batch(cache, train_items, lowcall=True, mode="rich",
                        include_cross=True, delegatecall=True)
    vg, vl = make_batch(cache, test_items, lowcall=True, mode="rich",
                        include_cross=True, delegatecall=True)
    torch.manual_seed(seed); np.random.seed(seed)
    m = HGTClassifier(32, 64, 2, 4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tr_g = dgl.batch(tg)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(tr_g), tl)
        loss.backward(); opt.step()
    m.eval()
    preds, truths = [], []
    with torch.no_grad():
        for g in vg:
            preds.append(m(g).argmax(1).item())
    truths = vl.tolist()
    return np.array(preds), np.array(truths)


def metrics(p, t):
    return dict(precision=float(precision_score(t, p, zero_division=0)),
                recall=float(recall_score(t, p, zero_division=0)),
                f1=float(f1_score(t, p, zero_division=0)),
                acc=float(accuracy_score(t, p)))


def cv_on(items, n_folds=5, epochs=50):
    _, cache = load_items(CACHE, None, 0)
    graphs, labels = make_batch(cache, items, lowcall=True, mode="rich",
                                include_cross=True, delegatecall=True)
    f1s = []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        idx = np.arange(len(labels))
        for tr, te in skf.split(idx, labels.numpy()):
            m = HGTClassifier(32, 64, 2, 4, 2)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            tr_g = dgl.batch([graphs[i] for i in tr])
            for _ in range(epochs):
                m.train(); opt.zero_grad()
                loss = F.cross_entropy(m(tr_g), labels[tr]); loss.backward(); opt.step()
            m.eval(); preds, truths = [], []
            with torch.no_grad():
                for i in te:
                    preds.append(m(graphs[i]).argmax(1).item()); truths.append(labels[i].item())
            f1s.append(f1_score(truths, preds, zero_division=0))
    return float(np.mean(f1s)), float(np.std(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items, _ = load_items(CACHE, None, 0)          # 69 pos + 469 neg (no subsample)
    sf_pos, sb_pos = split_pos(items)
    print(f"SolidiFI pos={len(sf_pos)}  SmartBugs pos={len(sb_pos)}", flush=True)

    out = {}
    # internal
    for name, pos in (("sf_only", sf_pos), ("sb_only", sb_pos)):
        its = pos + neg_items(0)
        t0 = time.time()
        m, s = cv_on(its, epochs=args.epochs)
        out[name] = {"f1_mean": m, "f1_std": s, "n_pos": len(pos), "wall_sec": round(time.time()-t0,1)}
        print(f"RESULT {name}: F1={m:.4f}+-{s:.4f} (pos={len(pos)})", flush=True)

    # cross-dataset
    for name, tr_pos, te_pos in (("sf_to_sb", sf_pos, sb_pos), ("sb_to_sf", sb_pos, sf_pos)):
        tr = tr_pos + neg_items(0)
        te = te_pos + neg_items(1)
        t0 = time.time()
        p, t = train_eval(tr, te, epochs=args.epochs)
        m = metrics(p, t)
        m["wall_sec"] = round(time.time()-t0, 1)
        out[name] = m
        print(f"RESULT {name}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (train_pos={len(tr_pos)} test_pos={len(te_pos)})", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"))
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
