"""Fine-tune a code transformer on the SAME 199 graph-buildable instances as HGT.

Re-derives the exact 69 positive + 130 subsampled-negative contract list used by
experiments.py (cache_full.pkl, np.RandomState(0) subsample, pos-first ordering)
and maps the stale legacy paths to the current data_src layout. Uses the same
StratifiedKFold(n_splits, shuffle=True, random_state=seed) as experiments.py so
that per-(fold, seed) F1 pairs one-to-one with the HGT results, enabling a valid
paired test (M3: HGT vs GraphCodeBERT "ties").

Writes results/full_<model>_common.json with a flat per_fold_f1 list, matching
the schema that src/significance.py reads.
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from codebert_baseline import SolDataset, TransformerClassifier  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/graphcodebert-base")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subsample-neg", type=int, default=130,
                    help="0 = use all graph-buildable negatives")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = common_items(subsample_neg=args.subsample_neg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = SolDataset(items, tokenizer, args.max_len)
    labels = ds.labels.numpy()
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    print(f"dataset: {len(ds)} items (pos={n_pos}, neg={n_neg}) on {device}", flush=True)

    per_fold_f1, pooled = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        idx = np.arange(len(ds))
        seed_preds, seed_truths = [], []
        for fold, (tr, te) in enumerate(skf.split(idx, labels)):
            model = TransformerClassifier(args.model).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            tr_loader = DataLoader(torch.utils.data.Subset(ds, tr), batch_size=args.batch, shuffle=True)
            model.train()
            for _ in range(args.epochs):
                for ids, mask, lab in tr_loader:
                    ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(model(ids, mask), lab)
                    loss.backward()
                    opt.step()
            model.eval()
            preds, truths = [], []
            te_loader = DataLoader(torch.utils.data.Subset(ds, te), batch_size=args.batch)
            with torch.no_grad():
                for ids, mask, lab in te_loader:
                    ids, mask = ids.to(device), mask.to(device)
                    preds.extend(model(ids, mask).argmax(1).cpu().tolist())
                    truths.extend(lab.tolist())
            f1 = float(f1_score(truths, preds, zero_division=0))
            per_fold_f1.append(f1)
            seed_preds.extend(preds); seed_truths.extend(truths)
            print(f"  seed={seed} fold={fold} f1={f1:.4f}", flush=True)
        pooled.append({"preds": seed_preds, "truths": seed_truths})

    arr = np.array(per_fold_f1)
    out = {
        "model": args.model,
        "per_fold_f1": per_fold_f1,
        "f1_mean": float(arr.mean()), "f1_std": float(arr.std()),
        "n_graphs": len(ds), "n_pos": n_pos, "n_neg": n_neg,
        "pooled": pooled,
    }
    outp = args.out or os.path.join(ROOT, "results",
                                    f"full_{args.model.split('/')[-1]}_common.json")
    with open(outp, "w") as f:
        json.dump(out, f)
    print(f"RESULT {args.model} common f1={arr.mean():.4f}±{arr.std():.4f} "
          f"(n={len(ds)}, pos={n_pos}, neg={n_neg})", flush=True)
    print("saved", outp, flush=True)


if __name__ == "__main__":
    main()
