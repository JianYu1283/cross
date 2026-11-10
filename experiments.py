"""Unified experiment harness for the JISA revision (three-factor decomposition).

Reconstructs the dataset from cache_full.pkl (no source .sol needed), supports
three feature modes (onehot / sparse / rich), lowcall on/off, delegatecall
on/off, and returns per-fold results + per-instance predictions so downstream
significance tests (paired t/Wilcoxon, bootstrap, permutation) can be run.

Usage:
  python src/experiments.py --model hgt --mode rich --lowcall 1 --seeds 5
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import HGTClassifier, GATClassifier, NODE_TYPES, NT, ALL_ETYPES, CROSS_ETYPES  # noqa: E402

STMT_TYPES = ["EXPRESSION", "ENTRYPOINT", "RETURN", "VARIABLE", "IF", "ENDIF",
              "OTHER_ENTRYPOINT", "PLACEHOLDER", "STARTLOOP", "IFLOOP", "ENDLOOP",
              "ASSEMBLY", "ENDASSEMBLY", "THROW", "CONTINUE"]
FEAT_DIM = {"onehot": 3, "sparse": 8, "rich": 32}
NT_IDX = {"function": 0, "stmt": 1, "ext_sink": 2}


def load_items(cache_path, subsample_neg=None, seed=0):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    pos, neg = [], []
    for k in cache.keys():
        s = str(k).lower()
        if ("solidifi_buggy_contracts" in s) or ("smartbug-dataset" in s):
            pos.append((k, 1))
        elif "smartbugs-wild-clean-contracts" in s:
            neg.append((k, 0))
    if subsample_neg and len(neg) > subsample_neg:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(neg), subsample_neg, replace=False)
        neg = [neg[i] for i in idx]
    return pos + neg, cache


def node_feat(ntype, data, lowcall, mode):
    dim = FEAT_DIM[mode]
    v = np.zeros(dim, dtype=np.float32)
    if mode == "onehot":
        v[NT_IDX[ntype]] = 1.0
        return v
    # sparse / rich share the first 8 dims
    if ntype == "function":
        v[0] = 1.0
        v[5] = 1.0 if data.get("payable") else 0.0
        v[6] = 1.0 if data.get("constructor") else 0.0
        v[7] = 1.0 if data.get("fallback") else 0.0
    elif ntype == "stmt":
        v[1] = 1.0
        if lowcall:
            v[3] = 1.0 if data.get("is_ext_call") else 0.0
            v[4] = 1.0 if data.get("is_delegate") else 0.0
    elif ntype == "ext_sink":
        v[2] = 1.0
        if lowcall:
            v[3] = 1.0
            v[4] = 1.0 if data.get("is_delegate") else 0.0
    if mode == "rich":
        if ntype == "stmt":
            st = data.get("stmt_type", "").replace("NodeType.", "")
            if st in STMT_TYPES:
                v[8 + STMT_TYPES.index(st)] = 1.0
            expr = data.get("expression", "") or ""
            if "require" in expr:
                v[23] = 1.0
            if "revert" in expr or "assert" in expr:
                v[24] = 1.0
    return v


def nx_to_dgl(G, lowcall=True, mode="rich", include_cross=True, delegatecall=True):
    """Convert a networkx MultiDiGraph to a DGL heterograph with the given features."""
    def active(n):
        return not (G.nodes[n]["node_type"] == "ext_sink" and not lowcall)

    id_map = {}
    for ntype in NODE_TYPES:
        nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype and active(n)]
        for i, n in enumerate(nodes):
            id_map[n] = (ntype, i)

    feats = {NT[t]: [] for t in NODE_TYPES}
    for ntype in NODE_TYPES:
        nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype and active(n)]
        for n in nodes:
            feats[NT[ntype]].append(node_feat(ntype, G.nodes[n], lowcall, mode))

    edges = {et: ([], []) for et in ALL_ETYPES}
    for u, v, ed in G.edges(data=True):
        if u not in id_map or v not in id_map:
            continue
        tu = G.nodes[u]["node_type"]
        tv = G.nodes[v]["node_type"]
        et = ed["edge_type"]
        canon = (NT[tu], et, NT[tv])
        if canon not in edges:
            continue
        if canon in CROSS_ETYPES and not include_cross:
            continue
        if canon == (NT["function"], "delegatecall", NT["function"]) and not delegatecall:
            continue
        edges[canon][0].append(id_map[u][1])
        edges[canon][1].append(id_map[v][1])

    data_dict = {}
    for et, (srcs, dsts) in edges.items():
        data_dict[et] = (torch.tensor(srcs, dtype=torch.long),
                         torch.tensor(dsts, dtype=torch.long))
    num_nodes = {NT[t]: len(feats[NT[t]]) for t in NODE_TYPES}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)
    dim = FEAT_DIM[mode]
    for t in NODE_TYPES:
        f = feats[NT[t]]
        g.nodes[NT[t]].data["feat"] = torch.tensor(f) if f else torch.zeros(0, dim)
    return g


def make_batch(cache, items, lowcall, mode, include_cross, delegatecall):
    graphs, labels = [], []
    for fpath, label in items:
        G = cache.get(fpath)
        if G is None:
            continue
        g = nx_to_dgl(G, lowcall=lowcall, mode=mode,
                      include_cross=include_cross, delegatecall=delegatecall)
        hg = dgl.to_homogeneous(g, ndata=["feat"])
        graphs.append(hg)
        labels.append(label)
    return graphs, torch.tensor(labels, dtype=torch.long)


def train_fold(model, graphs, labels, tr, te, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_g = dgl.batch([graphs[i] for i in tr])
    tr_l = labels[tr]
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(tr_g)
        loss = F.cross_entropy(logits, tr_l)
        loss.backward()
        opt.step()
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in te:
            logits = model(graphs[i])
            preds.append(logits.argmax(1).item())
            truths.append(labels[i].item())
    return preds, truths


def run_config(model_name, items, cache, mode, lowcall, include_cross, delegatecall,
               epochs, hidden, heads, layers, seeds, n_folds):
    """Run a full config; return per-(seed,fold) F1 + pooled predictions."""
    graphs, labels = make_batch(cache, items, lowcall, mode, include_cross, delegatecall)
    if len(graphs) == 0:
        return None
    dim = FEAT_DIM[mode]
    per_fold_f1 = []      # list of f1 per (seed, fold)
    pooled = []           # list of dicts {preds, truths} per seed
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        cnt = Counter(labels.tolist())
        k = max(2, min(n_folds, min(cnt.values())))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        idx = np.arange(len(labels))
        seed_preds, seed_truths = [], []
        for tr, te in skf.split(idx, labels.numpy()):
            if model_name == "hgt":
                m = HGTClassifier(dim, hidden, 2, heads, layers)
            else:
                m = GATClassifier(dim, hidden, 2, heads, layers)
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
        "per_fold_f1": per_fold_f1,
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "f1_pooled": float(f1_score(all_t, all_p, zero_division=0)),
        "prec_pooled": float(precision_score(all_t, all_p, zero_division=0)),
        "recall_pooled": float(recall_score(all_t, all_p, zero_division=0)),
        "acc_pooled": float(accuracy_score(all_t, all_p)),
        "pooled": pooled,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hgt", "gat"], default="hgt")
    ap.add_argument("--mode", choices=["onehot", "sparse", "rich"], default="rich")
    ap.add_argument("--lowcall", type=int, default=1)
    ap.add_argument("--include-cross", type=int, default=1)
    ap.add_argument("--delegatecall", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--subsample-neg", type=int, default=0)
    ap.add_argument("--cache", default="data/cache_full.pkl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    items, cache = load_items(args.cache, subsample_neg=args.subsample_neg or None)
    print(f"dataset: {len(items)} items (cache has {len(cache)} graphs)", flush=True)
    res = run_config(args.model, items, cache, args.mode, bool(args.lowcall),
                     bool(args.include_cross), bool(args.delegatecall),
                     args.epochs, args.hidden, args.heads, args.layers,
                     list(range(args.seeds)), args.folds)
    res["config"] = vars(args)
    res["wall_sec"] = time.time() - t0
    tag = (f"{args.model}_{args.mode}_low{args.lowcall}_cross{args.include_cross}"
           f"_del{args.delegatecall}")
    print(f"RESULT {tag}: F1={res['f1_mean']:.4f}±{res['f1_std']:.4f} "
          f"(n={res['n_graphs']}, pos={res['n_pos']}, neg={res['n_neg']}) "
          f"[{res['wall_sec']:.0f}s]", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f)
        print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
