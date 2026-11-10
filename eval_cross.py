"""E3: hold-out evaluation on injected cross-contract reentrancy pairs.

Trains HGT on standard reentrancy (SolidiFI + smartbugs) vs clean, then
evaluates recall on the injected victim+attacker pairs, comparing the
single-contract (victim-only) and cross-contract (victim+attacker) representations.
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
    print(f"train graphs: {len(train_graphs)}")

    victim_files = sorted(glob.glob("data/injected/victim/*.sol"))
    pair_files = sorted(glob.glob("data/injected/pair/*.sol"))
    inj_cache = build_and_cache([(f, 1) for f in victim_files + pair_files], "data/cache_inj.pkl")
    vg, _ = make_dgl_batch(inj_cache, [(f, 1) for f in victim_files], lowcall=True)
    pg, _ = make_dgl_batch(inj_cache, [(f, 1) for f in pair_files], lowcall=True)

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
    for name, gs in [("single", vg), ("cross", pg)]:
        preds = []
        for g in gs:
            with torch.no_grad():
                logits = model(g.to(device))
            preds.append(logits.argmax(1).item())
        recall = sum(1 for p in preds if p == 1) / max(1, len(preds))
        print(f"E3 {name}: recall_on_cross_contract_reentrancy = {recall:.4f} ({len(preds)} samples)")


if __name__ == "__main__":
    main()
