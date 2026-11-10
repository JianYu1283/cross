"""Comment 1: cross-dataset transfer across feature regimes and lowcall settings.

Finding 5 reports the cross-source collapse for rich features only and then
claims it "strengthens the case for explicit low-level-call modeling". That
inference is unsupported until we show that a low-level-call-only (sparse +
ext_sink) representation transfers *better* than rich features. This script
generalizes per_dataset.py's sf_to_sb / sb_to_sf protocols across
{onehot, sparse, stmtonly, rronly, rich} x {lowcall 0/1}.

Protocol (same as per_dataset.py):
  sf_to_sb : train SolidiFI pos + 130 neg (seed 0) -> test SmartBugs pos + 130 neg (seed 1)
  sb_to_sf : train SmartBugs pos + 130 neg (seed 0) -> test SolidiFI pos + 130 neg (seed 1)

The test set always includes a disjoint 130-negative draw, so precision/F1 are
well-defined (not positive-only).
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import HGTClassifier, NODE_TYPES, NT, ALL_ETYPES
from experiments import load_items, STMT_TYPES, NT_IDX

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "cache_full.pkl")

FEAT_DIM = {"onehot": 3, "sparse": 8, "stmtonly": 32, "rronly": 32, "rich": 32}


def node_feat(ntype, data, lowcall, mode):
    dim = FEAT_DIM[mode]
    v = np.zeros(dim, dtype=np.float32)
    if mode == "onehot":
        v[NT_IDX[ntype]] = 1.0
        return v
    # sparse base (dims 0-7), shared by sparse/stmtonly/rronly/rich
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
    # rich added groups
    if mode in ("rich", "stmtonly") and ntype == "stmt":
        st = data.get("stmt_type", "").replace("NodeType.", "")
        if st in STMT_TYPES:
            v[8 + STMT_TYPES.index(st)] = 1.0
    if mode in ("rich", "rronly") and ntype == "stmt":
        expr = data.get("expression", "") or ""
        if "require" in expr:
            v[23] = 1.0
        if "revert" in expr or "assert" in expr:
            v[24] = 1.0
    return v


def nx_to_dgl(G, lowcall, mode):
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
        tu = G.nodes[u]["node_type"]; tv = G.nodes[v]["node_type"]
        canon = (NT[tu], ed["edge_type"], NT[tv])
        if canon not in edges:
            continue
        edges[canon][0].append(id_map[u][1]); edges[canon][1].append(id_map[v][1])
    data_dict = {et: (torch.tensor(s, dtype=torch.long),
                      torch.tensor(t, dtype=torch.long)) for et, (s, t) in edges.items()}
    num_nodes = {NT[t]: len(feats[NT[t]]) for t in NODE_TYPES}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)
    dim = FEAT_DIM[mode]
    for t in NODE_TYPES:
        f = feats[NT[t]]
        g.nodes[NT[t]].data["feat"] = torch.tensor(f) if f else torch.zeros(0, dim)
    return g


def make_batch(cache, items, lowcall, mode):
    graphs, labels = [], []
    for fpath, label in items:
        G = cache.get(fpath)
        if G is None:
            continue
        graphs.append(dgl.to_homogeneous(nx_to_dgl(G, lowcall, mode), ndata=["feat"]))
        labels.append(label)
    return graphs, torch.tensor(labels, dtype=torch.long)


def is_sf(k):
    return "solidifi_buggy_contracts" in str(k).lower()


def is_sb(k):
    return "smartbug-dataset" in str(k).lower()


def neg_items(subsample_seed, n=130):
    items, _ = load_items(CACHE, None, 0)
    negs = [(k, 0) for k, l in items if l == 0]
    rng = np.random.RandomState(subsample_seed)
    idx = rng.choice(len(negs), n, replace=False)
    return [negs[i] for i in idx]


def train_eval(tr_items, te_items, cache, lowcall, mode, epochs=50, seed=0):
    tg, tl = make_batch(cache, tr_items, lowcall, mode)
    vg, vl = make_batch(cache, te_items, lowcall, mode)
    torch.manual_seed(seed); np.random.seed(seed)
    m = HGTClassifier(FEAT_DIM[mode], 64, 2, 4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tr_g = dgl.batch(tg)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(tr_g), tl)
        loss.backward(); opt.step()
    m.eval()
    preds = []
    with torch.no_grad():
        for g in vg:
            preds.append(m(g).argmax(1).item())
    truths = vl.tolist()
    p = np.array(preds); t = np.array(truths)
    return dict(precision=float(precision_score(t, p, zero_division=0)),
                recall=float(recall_score(t, p, zero_division=0)),
                f1=float(f1_score(t, p, zero_division=0)),
                acc=float(accuracy_score(t, p)),
                n_test=int(len(t)), n_pos_test=int((t == 1).sum()))


def main():
    items, cache = load_items(CACHE, None, 0)
    sf_pos = [(k, 1) for k, l in items if l == 1 and is_sf(k)]
    sb_pos = [(k, 1) for k, l in items if l == 1 and is_sb(k)]
    print(f"SolidiFI pos={len(sf_pos)}  SmartBugs pos={len(sb_pos)}", flush=True)

    out = {}
    for mode in ("onehot", "sparse", "stmtonly", "rronly", "rich"):
        for lowcall in (0, 1):
            for name, tr_pos, te_pos in (("sf_to_sb", sf_pos, sb_pos),
                                          ("sb_to_sf", sb_pos, sf_pos)):
                tr = tr_pos + neg_items(0)
                te = te_pos + neg_items(1)
                t0 = time.time()
                m = train_eval(tr, te, cache, lowcall, mode)
                key = f"{name}_{mode}_low{lowcall}"
                out[key] = m
                out[key]["wall_sec"] = round(time.time() - t0, 1)
                print(f"RESULT {key}: P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} acc={m['acc']:.3f} "
                      f"(n_test={m['n_test']}, pos_test={m['n_pos_test']})", flush=True)

    json.dump(out, open(os.path.join(ROOT, "results", "cross_dataset_regimes.json"), "w"))
    print("saved results/cross_dataset_regimes.json")


if __name__ == "__main__":
    main()
