"""Item 8: feature-group ablation of the rich features.

The 32-dim rich vector = sparse base (dims 0-7, incl. ext_call flags)
+ Slither statement-type one-hot (dims 8-22) + require/revert flags (dims 23-24).
Ablates which rich group closes the gap sparse(0.604) -> rich(0.910) at lowcall=0.

Groups:
  full     = sparse + stmt types + require/revert   (== rich, existing 0.910)
  stmtonly = sparse + stmt types                    (no require/revert)
  rronly   = sparse + require/revert                (no stmt types)
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F, dgl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import HGTClassifier, NODE_TYPES, NT, ALL_ETYPES, CROSS_ETYPES
from experiments import load_items, STMT_TYPES, FEAT_DIM

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "data", "cache_full.pkl")


def node_feat(ntype, data, lowcall, feat_group):
    v = np.zeros(FEAT_DIM["rich"], dtype=np.float32)
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
        if feat_group in ("full", "stmtonly"):
            st = data.get("stmt_type", "").replace("NodeType.", "")
            if st in STMT_TYPES:
                v[8 + STMT_TYPES.index(st)] = 1.0
        if feat_group in ("full", "rronly"):
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


def nx_to_dgl(G, lowcall, feat_group):
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
            feats[NT[ntype]].append(node_feat(ntype, G.nodes[n], lowcall, feat_group))
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
    dim = FEAT_DIM["rich"]
    for t in NODE_TYPES:
        f = feats[NT[t]]
        g.nodes[NT[t]].data["feat"] = torch.tensor(f) if f else torch.zeros(0, dim)
    return g


def run(group, lowcall, epochs=50, seeds=5, folds=5):
    items, cache = load_items(CACHE, 130, 0)
    graphs, labels = [], []
    for fpath, label in items:
        G = cache.get(fpath)
        if G is None:
            continue
        g = nx_to_dgl(G, lowcall, group)
        graphs.append(dgl.to_homogeneous(g, ndata=["feat"]))
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.long)
    per_fold = []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
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
            per_fold.append(f1_score(truths, preds, zero_division=0))
    return float(np.mean(per_fold)), float(np.std(per_fold)), per_fold


if __name__ == "__main__":
    out = {}
    for group in ("stmtonly", "rronly"):
        for low in (0, 1):
            t0 = time.time()
            m, s, pf = run(group, low)
            out[f"{group}_low{low}"] = {"f1_mean": m, "f1_std": s, "per_fold_f1": pf, "wall_sec": round(time.time()-t0,1)}
            print(f"RESULT {group} low{low}: F1={m:.4f}+-{s:.4f} [{out[f'{group}_low{low}']['wall_sec']}s]", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "feature_ablation.json"), "w"))
    print("saved results/feature_ablation.json")
