"""Noise-node control for the ext_sink symmetry-break confound (Threats Sec.7).

Under one-hot features, ext_sink raises HGT F1 from 0.455 to 0.723 (+26.8 pts).
Does that gain come from the *low-level-call semantics*, or merely from adding a
third node type that breaks the function/stmt symmetry? We attach a *noise* node
(uninformative one-hot identity, same slot as ext_sink) to every function and
measure the F1 under identical fold x seed CV:

  - noise ~ 0.455  => the gain is semantic (position on low-call statements matters)
  - noise ~ 0.723  => the gain is (at least partly) symmetry-breaking

Run:  python src/noise_node_control.py --seeds 5 --folds 5 --out data/noise_control.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from experiments import load_items, train_fold  # noqa: E402
from pipeline import HGTClassifier  # noqa: E402

# Node types: function, stmt, noise (3 types -> matches HGTClassifier num_ntypes)
NT = {"function": 0, "stmt": 1, "noise": 2}
# 6 edge types -> matches HGTClassifier num_etypes
ETYPES = [
    ("function", "cfg", "stmt"),
    ("stmt", "cfg", "stmt"),
    ("function", "internal_call", "function"),
    ("function", "external_call", "function"),
    ("function", "delegatecall", "function"),
    ("function", "noise_edge", "noise"),
]


def nx_to_noise_dgl(G):
    """Build a one-hot (3-dim) DGL graph where ext_sink is replaced by a noise
    node attached to every function (uninformative location)."""
    f_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "function"]
    s_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "stmt"]
    fid = {n: i for i, n in enumerate(f_nodes)}
    sid = {n: i for i, n in enumerate(s_nodes)}
    n_f, n_s = len(f_nodes), len(s_nodes)
    n_n = n_f  # one noise node per function

    edges = {et: ([], []) for et in ETYPES}
    for u, v, ed in G.edges(data=True):
        tu = G.nodes[u]["node_type"]
        tv = G.nodes[v]["node_type"]
        et = ed["edge_type"]
        if et == "extcall_sink":
            continue  # ext_sink nodes dropped entirely (lowcall off)
        canon = None
        if et == "cfg":
            canon = (tu, "cfg", tv)
        elif et == "internal_call":
            canon = ("function", "internal_call", "function")
        elif et == "external_call":
            canon = ("function", "external_call", "function")
        elif et == "delegatecall":
            canon = ("function", "delegatecall", "function")
        if canon is None or canon not in edges:
            continue
        if tu == "function" and tv == "function":
            if u not in fid or v not in fid:
                continue
            edges[canon][0].append(fid[u]); edges[canon][1].append(fid[v])
        elif tu == "function" and tv == "stmt":
            if u not in fid or v not in sid:
                continue
            edges[canon][0].append(fid[u]); edges[canon][1].append(sid[v])
        elif tu == "stmt" and tv == "stmt":
            if u not in sid or v not in sid:
                continue
            edges[canon][0].append(sid[u]); edges[canon][1].append(sid[v])

    # noise edge: function i -> noise i
    edges[("function", "noise_edge", "noise")] = (list(range(n_f)), list(range(n_n)))

    data_dict = {}
    for et, (srcs, dsts) in edges.items():
        data_dict[et] = (torch.tensor(srcs, dtype=torch.long),
                         torch.tensor(dsts, dtype=torch.long))
    num_nodes = {"function": n_f, "stmt": n_s, "noise": n_n}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)
    # one-hot features (3-dim)
    feat = {}
    f = torch.zeros(n_f, 3); f[:, 0] = 1.0
    s = torch.zeros(n_s, 3); s[:, 1] = 1.0
    n = torch.zeros(n_n, 3); n[:, 2] = 1.0
    for t, v in (("function", f), ("stmt", s), ("noise", n)):
        g.nodes[t].data["feat"] = v
    return g


def make_noise_batch(cache, items):
    graphs, labels = [], []
    for fpath, label in items:
        G = cache.get(fpath)
        if G is None:
            continue
        g = nx_to_noise_dgl(G)
        hg = dgl.to_homogeneous(g, ndata=["feat"])
        graphs.append(hg)
        labels.append(label)
    return graphs, torch.tensor(labels, dtype=torch.long)


def run(cache_path, seeds, folds, epochs, hidden, heads, layers, subsample_neg):
    items, cache = load_items(cache_path, subsample_neg=subsample_neg or None)
    graphs, labels = make_noise_batch(cache, items)
    per_fold_f1, pooled = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        cnt = Counter(labels.tolist())
        k = max(2, min(folds, min(cnt.values())))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        idx = np.arange(len(labels))
        seed_preds, seed_truths = [], []
        for tr, te in skf.split(idx, labels.numpy()):
            m = HGTClassifier(3, hidden, 2, heads, layers)
            preds, truths = train_fold(m, graphs, labels, tr, te, epochs, 1e-3)
            per_fold_f1.append(f1_score(truths, preds, zero_division=0))
            seed_preds.extend(preds); seed_truths.extend(truths)
        pooled.append({"preds": seed_preds, "truths": seed_truths})
    f1s = np.array(per_fold_f1)
    all_p = np.concatenate([p["preds"] for p in pooled])
    all_t = np.concatenate([p["truths"] for p in pooled])
    return {
        "n_graphs": len(graphs),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "f1_pooled": float(f1_score(all_t, all_p, zero_division=0)),
        "per_fold_f1": per_fold_f1,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache_full.pkl")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--subsample-neg", type=int, default=130)
    ap.add_argument("--out", default="data/noise_control.json")
    args = ap.parse_args()

    t0 = time.time()
    res = run(args.cache, list(range(args.seeds)), args.folds, args.epochs,
              args.hidden, args.heads, args.layers, args.subsample_neg)
    res["wall_sec"] = time.time() - t0
    print(f"NOISE CONTROL (one-hot + noise node per function): "
          f"F1={res['f1_mean']:.4f}±{res['f1_std']:.4f} "
          f"(n={res['n_graphs']}, pos={res['n_pos']}, neg={res['n_neg']}) "
          f"[{res['wall_sec']:.0f}s]", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("saved", args.out, flush=True)
