"""Diagnostic: is MANDO lowcall=0 collapse a genuine majority-class collapse or NaN?"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from mando_baseline import (make_mando_graphs, precompute_metapaths,
                            get_length_2_metapath, MANDOClassifier)
from pipeline import collect_dataset, build_and_cache
from sklearn.metrics import f1_score

M = "data"
P1 = f"{M}/data/solidifi_buggy_contracts/reentrancy"
P2 = f"{M}/data/smartbug-dataset/reentrancy"
N1 = f"{M}/data/smartbugs-wild-clean-contracts"

torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

items = collect_dataset([P1, P2], [N1], 130, 0)
cache = build_and_cache(items, "data/cache_full.pkl")
graphs, labels = make_mando_graphs(cache, items, lowcall=False)

ntype_counts = Counter()
for g in graphs:
    for t in g.ntypes:
        if g.num_nodes(t) > 0:
            ntype_counts[t] += 1
print("graphs:", len(graphs), "labels:", Counter(labels.tolist()))
print("graphs with non-empty node type:", dict(ntype_counts))

rep = next(g for g in graphs if g.number_of_nodes() > 0)
meta_paths = get_length_2_metapath(rep)
in_size = len(rep.ntypes)
print("metapaths:", len(meta_paths), "src types:", sorted(set(mp[0][0] for mp in meta_paths)))

info = precompute_metapaths(graphs, meta_paths, device)
labels = labels.to(device)
model = MANDOClassifier(meta_paths, in_size, 32, 2, 8, 0.6).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

idx = np.arange(len(labels))
rng = np.random.RandomState(0)
tr = rng.choice(idx, int(len(idx) * 0.8), replace=False)
for ep in range(20):
    model.train(); opt.zero_grad()
    loss = 0.0
    for i in tr:
        loss = loss + F.cross_entropy(model(info[i]), labels[i].unsqueeze(0))
    loss = loss / len(tr)
    loss.backward(); opt.step()

model.eval()
logits_list, truths = [], []
with torch.no_grad():
    for i in idx:
        logits_list.append(model(info[i]).detach().cpu())
        truths.append(labels[i].item())
all_logits = torch.cat(logits_list)
print("logits finite:", bool(torch.isfinite(all_logits).all().item()),
      "| min:", round(all_logits.min().item(), 4), "max:", round(all_logits.max().item(), 4))
preds = all_logits.argmax(1).tolist()
print("pred distribution:", Counter(preds), "| truth:", Counter(truths))
print("f1:", f1_score(truths, preds, zero_division=0))
