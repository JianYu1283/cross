"""Item 1: class-weighted HGT, PR-AUC/ROC-AUC, and repeated negative subsampling.

Runs HGT (rich features, lowcall on) with an optional class weight on the
cross-entropy loss, saving per-instance softmax probabilities so PR-AUC and
ROC-AUC can be computed (the original harness saved only argmax labels).

Usage:
  python src/class_weighted.py --subsample-neg 0 --pos-weight -1   # balanced full-469
  python src/class_weighted.py --subsample-neg 0 --pos-weight 0    # unweighted full-469
  python src/class_weighted.py --subsample-neg 130 --subsample-seed K --pos-weight 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import HGTClassifier
from experiments import load_items, make_batch

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "cache_full.pkl")


def run(subsample_neg, subsample_seed, pos_weight, n_seeds=5, n_folds=5, epochs=50):
    items, cache = load_items(CACHE, subsample_neg or None, subsample_seed)
    graphs, labels = make_batch(cache, items, lowcall=True, mode="rich",
                                include_cross=True, delegatecall=True)
    n = len(graphs)
    per_fold_f1, pooled = [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        idx = np.arange(n)
        seed_probs, seed_preds, seed_truths = [], [], []
        for tr, te in skf.split(idx, labels.numpy()):
            m = HGTClassifier(32, 64, 2, 4, 2)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            tr_g = dgl.batch([graphs[i] for i in tr])
            tr_l = labels[tr]
            w = torch.tensor([1.0, pos_weight], dtype=torch.float) if pos_weight > 0 else None
            for _ in range(epochs):
                m.train(); opt.zero_grad()
                loss = F.cross_entropy(m(tr_g), tr_l, weight=w)
                loss.backward(); opt.step()
            m.eval()
            fold_preds, fold_truths = [], []
            with torch.no_grad():
                for i in te:
                    logits = m(graphs[i])
                    p = torch.softmax(logits, 1)[:, 1]
                    pred = logits.argmax(1).item()
                    seed_probs.extend(p.tolist()); seed_preds.append(pred)
                    seed_truths.append(labels[i].item())
                    fold_preds.append(pred); fold_truths.append(labels[i].item())
            per_fold_f1.append(f1_score(fold_truths, fold_preds, zero_division=0))
        pooled.append({"probs": seed_probs, "preds": seed_preds, "truths": seed_truths})

    P = np.concatenate([p["probs"] for p in pooled])
    Y = np.concatenate([p["preds"] for p in pooled])
    T = np.concatenate([p["truths"] for p in pooled])
    rec_pos = recall_score(T, Y, zero_division=0)
    rec_neg = recall_score(1 - T, 1 - Y, zero_division=0)
    return {
        "n": n, "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
        "f1_mean": float(np.mean(per_fold_f1)), "f1_std": float(np.std(per_fold_f1)),
        "precision": float(precision_score(T, Y, zero_division=0)),
        "recall": rec_pos, "f1": float(f1_score(T, Y, zero_division=0)),
        "mcc": float(matthews_corrcoef(T, Y)),
        "balanced_acc": float((rec_pos + rec_neg) / 2),
        "pr_auc": float(average_precision_score(T, P)),
        "roc_auc": float(roc_auc_score(T, P)),
        "pos_weight": pos_weight, "subsample_neg": subsample_neg,
        "subsample_seed": subsample_seed, "wall_sec": 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample-neg", type=int, default=0)
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="-1 = auto n_neg/n_pos, 0 = unweighted, >0 = fixed")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    # resolve pos_weight: -1 -> n_neg/n_pos, needs item counts
    pw = args.pos_weight
    if pw < 0:
        items, _ = load_items(CACHE, args.subsample_neg or None, args.subsample_seed)
        npos = sum(1 for _, l in items if l == 1)
        nneg = sum(1 for _, l in items if l == 0)
        pw = nneg / npos
    res = run(args.subsample_neg, args.subsample_seed, pw,
              args.seeds, args.folds, args.epochs)
    res["wall_sec"] = round(time.time() - t0, 1)
    tag = (f"cw_s{args.subsample_neg}_seed{args.subsample_seed}_pw{pw:.3f}"
           .replace(".", "p"))
    print(f"RESULT {tag}: F1={res['f1_mean']:.4f}±{res['f1_std']:.4f} "
          f"P={res['precision']:.3f} R={res['recall']:.3f} MCC={res['mcc']:.3f} "
          f"BA={res['balanced_acc']:.3f} PR-AUC={res['pr_auc']:.3f} "
          f"ROC-AUC={res['roc_auc']:.3f} [{res['wall_sec']}s]", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f)
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
