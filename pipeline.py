"""End-to-end cross-contract reentrancy detection pipeline.

Graph construction -> DGL heterogeneous graphs -> HGT / GAT -> train+eval.

Run:  python src/pipeline.py --pos-dirs <dir> --neg-dirs <dir> --limit N
"""
import os
import sys
import glob
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

sys.path.insert(0, os.path.dirname(__file__))
from build_graphs import build_graph  # noqa: E402

FEAT_DIM = 32
STMT_TYPES = ["EXPRESSION", "ENTRYPOINT", "RETURN", "VARIABLE", "IF", "ENDIF",
              "OTHER_ENTRYPOINT", "PLACEHOLDER", "STARTLOOP", "IFLOOP", "ENDLOOP",
              "ASSEMBLY", "ENDASSEMBLY", "THROW", "CONTINUE"]
NODE_TYPES = ["function", "stmt", "ext_sink"]
NT = {"function": "f", "stmt": "s", "ext_sink": "e"}
# canonical edge types (src_ntype, relation, dst_ntype)
ALL_ETYPES = [
    ("f", "cfg", "s"), ("s", "cfg", "s"),
    ("f", "internal_call", "f"),
    ("f", "external_call", "f"),
    ("s", "extcall_sink", "e"),
    ("f", "delegatecall", "f"),
]
# cross-contract edge relations (dropped for single-contract ablation)
CROSS_ETYPES = {("f", "external_call", "f"), ("f", "delegatecall", "f")}


def node_feat(ntype, data, lowcall=True):
    v = np.zeros(FEAT_DIM, dtype=np.float32)
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
        st = data.get("stmt_type", "").replace("NodeType.", "")
        if st in STMT_TYPES:
            v[8 + STMT_TYPES.index(st)] = 1.0
        expr = data.get("expression", "") or ""
        if "require" in expr:
            v[23] = 1.0
        if "revert" in expr or "assert" in expr:
            v[24] = 1.0
    elif ntype == "ext_sink":
        v[2] = 1.0
        if lowcall:
            v[3] = 1.0
            v[4] = 1.0 if data.get("is_delegate") else 0.0
    return v


def nx_to_dgl(G, include_cross=True, lowcall=True):
    """Convert a networkx graph to a DGL heterograph with full schema."""
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
            feats[NT[ntype]].append(node_feat(ntype, G.nodes[n], lowcall))

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
        edges[canon][0].append(id_map[u][1])
        edges[canon][1].append(id_map[v][1])

    data_dict = {}
    for et, (srcs, dsts) in edges.items():
        if et in CROSS_ETYPES and not include_cross:
            srcs, dsts = [], []  # drop cross-contract edges but keep schema
        data_dict[et] = (torch.tensor(srcs, dtype=torch.long),
                         torch.tensor(dsts, dtype=torch.long))
    num_nodes = {NT[t]: len(feats[NT[t]]) for t in NODE_TYPES}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)
    for t in NODE_TYPES:
        f = feats[NT[t]]
        g.nodes[NT[t]].data["feat"] = torch.tensor(f) if f else torch.zeros(0, FEAT_DIM)
    return g


class HGTClassifier(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, num_heads, n_layers):
        super().__init__()
        self.num_ntypes = len(NODE_TYPES)
        self.num_etypes = len(ALL_ETYPES)
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(dglnn.HGTConv(hidden, hidden // num_heads, num_heads,
                                             self.num_ntypes, self.num_etypes, use_norm=True))
        self.classifier = nn.Linear(hidden, n_classes)
        self.pool = dgl.nn.AvgPooling()

    def forward(self, hg):
        x = self.input_proj(hg.ndata["feat"])
        ntype = hg.ndata["_TYPE"]
        etype = hg.edata["_TYPE"]
        for layer in self.layers:
            x = layer(hg, x, ntype, etype)
        pooled = self.pool(hg, x)
        return self.classifier(pooled)


class GATClassifier(nn.Module):
    """Homogeneous GAT baseline (type-agnostic)."""
    def __init__(self, in_dim, hidden, n_classes, num_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_dim, hidden, num_heads, allow_zero_in_degree=True))
        for _ in range(n_layers - 1):
            self.layers.append(dglnn.GATConv(hidden * num_heads, hidden, num_heads, allow_zero_in_degree=True))
        self.classifier = nn.Linear(hidden * num_heads, n_classes)
        self.pool = dgl.nn.AvgPooling()

    def forward(self, hg):
        x = hg.ndata["feat"]
        for layer in self.layers:
            x = layer(hg, x).flatten(1)
        pooled = self.pool(hg, x)
        return self.classifier(pooled)


def collect_dataset(pos_dirs, neg_dirs, limit=None, seed=0):
    rng = random.Random(seed)
    items = []
    for d in pos_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.sol"))):
            items.append((f, 1))
    for d in neg_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.sol"))):
            items.append((f, 0))
    if limit:
        pos = [x for x in items if x[1] == 1]
        neg = [x for x in items if x[1] == 0]
        pos = rng.sample(pos, min(limit, len(pos)))
        neg = rng.sample(neg, min(limit, len(neg)))
        items = pos + neg
    rng.shuffle(items)
    return items


def build_and_cache(items, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    ok = 0
    for i, (fpath, label) in enumerate(items):
        key = fpath
        if key in cache:
            ok += 1
            continue
        G = build_graph(fpath)
        if G is not None and G.number_of_nodes() > 1:
            cache[key] = G
            ok += 1
        if (i + 1) % 25 == 0:
            print(f"  built {i + 1}/{len(items)} (ok={ok})", flush=True)
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return cache


def make_dgl_batch(cache, items, include_cross=True, lowcall=True):
    graphs, labels = [], []
    for fpath, label in items:
        G = cache.get(fpath)
        if G is None:
            continue
        g = nx_to_dgl(G, include_cross=include_cross, lowcall=lowcall)
        hg = dgl.to_homogeneous(g, ndata=["feat"])
        graphs.append(hg)
        labels.append(label)
    return graphs, torch.tensor(labels, dtype=torch.long)


def train_eval(model, graphs, labels, epochs, lr, device, n_folds=5, seed=0):
    """Stratified k-fold CV; returns mean acc/F1."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score
    from collections import Counter
    cnt = Counter(labels.numpy().tolist())
    n_folds = max(2, min(n_folds, min(cnt.values())))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = np.arange(len(labels))
    accs, f1s = [], []
    for tr, te in skf.split(idx, labels.numpy()):
        model_copy = rebuild(model)
        opt = torch.optim.Adam(model_copy.parameters(), lr=lr)
        tr_g = dgl.batch([graphs[i] for i in tr]).to(device)
        tr_l = labels[tr].to(device)
        for _ in range(epochs):
            model_copy.train()
            opt.zero_grad()
            logits = model_copy(tr_g)
            loss = F.cross_entropy(logits, tr_l)
            loss.backward()
            opt.step()
        model_copy.eval()
        preds, truths = [], []
        for i in te:
            with torch.no_grad():
                logits = model_copy(graphs[i].to(device))
            preds.append(logits.argmax(1).item())
            truths.append(labels[i].item())
        accs.append(accuracy_score(truths, preds))
        f1s.append(f1_score(truths, preds, zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def rebuild(model):
    import copy
    return copy.deepcopy(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-dirs", nargs="+", required=True)
    ap.add_argument("--neg-dirs", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--model", choices=["hgt", "gat"], default="hgt")
    ap.add_argument("--include-cross", type=int, default=1)
    ap.add_argument("--lowcall", type=int, default=1)
    ap.add_argument("--cache", default="data/cache.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = collect_dataset(args.pos_dirs, args.neg_dirs, args.limit, args.seed)
    print(f"dataset: {len(items)} contracts "
          f"({sum(1 for _, l in items if l == 1)} pos / {sum(1 for _, l in items if l == 0)} neg)")
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    cache = build_and_cache(items, args.cache)
    graphs, labels = make_dgl_batch(cache, items, include_cross=bool(args.include_cross),
                                    lowcall=bool(args.lowcall))
    print(f"built DGL graphs: {len(graphs)} / {len(items)}")

    if args.model == "hgt":
        model = HGTClassifier(FEAT_DIM, args.hidden, 2, args.heads, args.layers)
    else:
        model = GATClassifier(FEAT_DIM, args.hidden, 2, args.heads, args.layers)
    model = model.to(device)
    acc, f1 = train_eval(model, graphs, labels, args.epochs, 1e-3, device, seed=args.seed)
    print(f"RESULT model={args.model} include_cross={args.include_cross} "
          f"acc={acc:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
