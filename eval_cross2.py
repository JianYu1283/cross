"""E3 (diversified): precision/recall on the multi-template cross-contract benchmark.

Trains HGT on standard reentrancy (SolidiFI + smartbugs) vs clean, then evaluates
on the diversified injected set: 3 vulnerable templates (positive) vs 2 safe
external-call patterns (negative). Reports precision, recall, F1, and per-template
recall.
"""
import os
import sys
import glob
import torch
import torch.nn.functional as F
import dgl

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (HGTClassifier, FEAT_DIM, collect_dataset, build_and_cache,
                      make_dgl_batch)

M = "data"


def main():
    pos = [M + "/data/solidifi_buggy_contracts/reentrancy",
           M + "/data/smartbug-dataset/reentrancy"]
    neg = [M + "/data/smartbugs-wild-clean-contracts"]
    train_items = collect_dataset(pos, neg, limit=130, seed=0)
    cache = build_and_cache(train_items, "data/cache_full.pkl")
    train_graphs, train_labels = make_dgl_batch(cache, train_items, lowcall=True)
    print(f"train graphs: {len(train_graphs)}", flush=True)

    pos_files = sorted(glob.glob("data/injected_diverse/positive/*.sol"))
    neg_files = sorted(glob.glob("data/injected_diverse/negative/*.sol"))
    eval_items = [(f, 1) for f in pos_files] + [(f, 0) for f in neg_files]
    inj_cache = build_and_cache(eval_items, "data/cache_inj2.pkl")
    eval_graphs, eval_labels = make_dgl_batch(inj_cache, eval_items, lowcall=True)
    print(f"eval graphs: {len(eval_graphs)} (pos {len(pos_files)} / neg {len(neg_files)})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HGTClassifier(FEAT_DIM, 64, 2, 4, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_g = dgl.batch(train_graphs).to(device)
    tr_l = train_labels.to(device)
    for _ in range(50):
        model.train()
        opt.zero_grad()
        logits = model(tr_g)
        loss = F.cross_entropy(logits, tr_l)
        loss.backward()
        opt.step()
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for g in eval_graphs:
            preds.append(model(g.to(device)).argmax(1).item())
    truths = eval_labels.tolist()
    tp = sum(1 for p, t in zip(preds, truths) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, truths) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, truths) if p == 0 and t == 1)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    print(f"E3-diverse: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
          f"(tp={tp} fp={fp} fn={fn})", flush=True)

    # per-template recall
    for tag in ("t1", "t2", "t3"):
        ids = [i for i, f in enumerate(eval_items) if os.path.basename(f[0]).startswith(tag)]
        if not ids:
            continue
        r = sum(1 for i in ids if preds[i] == 1) / len(ids)
        print(f"  template {tag}: recall={r:.4f} ({len(ids)} samples)", flush=True)


if __name__ == "__main__":
    main()
