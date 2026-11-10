"""Faithful re-implementation of MANDO (DSAA 2022) as a baseline.

MANDO = HAN (metapath-GAT + semantic attention) on a symmetric heterogeneous
graph, with NODE-TYPE ONE-HOT features, and NO low-level call modeling (its
call graph omits call.value/delegatecall). Ported from ge-sc-master's
sco_models/model_hetero.py + graph_utils.py, adapted to DGL 2.2.

Speed note: MANDO's original runs on ONE global graph, so its per-metapath
cache never invalidates. Our per-contract adaptation recomputes the metapath
reachable graphs every forward, so we precompute them once here (they are
structure-only and independent of model weights).
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import nx_to_dgl, collect_dataset, build_and_cache  # noqa: E402


# --- ported from MANDO sco_models/model_hetero.py ---
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        # z: (N, M, D); aggregate over metapaths (dim 1)
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D)


class HANLayer(nn.Module):
    """One GATConv + (degenerate) semantic attention over a SINGLE metapath.

    MANDO instantiates one HANLayer per metapath (see MANDOGraphClassifier in
    model_hetero.py, lines 216-220); with a single metapath the semantic
    attention is a softmax over one element (identity), kept here for fidelity.
    """

    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()
        self.gat_layer = GATConv(in_size, out_size, num_heads, dropout, dropout,
                                 activation=F.elu, allow_zero_in_degree=True)
        self.semantic_attention = SemanticAttention(in_size=out_size * num_heads)

    def forward(self, mp_g, h):
        emb = self.gat_layer(mp_g, h).flatten(1)          # (N, D*K)
        return self.semantic_attention(emb.unsqueeze(1)).squeeze(1)  # (N, D*K)


class MANDOClassifier(nn.Module):
    """One HANLayer per metapath, node-type one-hot features, per-type readout."""

    def __init__(self, meta_paths, in_size, hidden, out_size, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [HANLayer(in_size, hidden, num_heads, dropout) for _ in meta_paths])
        self.src_types = [mp[0][0] for mp in meta_paths]
        self.classify = nn.Linear(hidden * num_heads, out_size)

    def forward(self, per):
        # per: list of (layer_idx, mp_graph, feat) for non-empty metapaths only
        type_features = {}
        for li, mp_g, feat in per:
            src_type = self.src_types[li]
            h = self.layers[li](mp_g, feat)
            type_features.setdefault(src_type, []).append(h)
        graph_emb = None
        for t, outs in type_features.items():
            # mean over metapaths, then mean over nodes (matches get_assemble_node_features)
            ft = torch.mean(torch.stack(outs), dim=0).mean(0)
            graph_emb = ft if graph_emb is None else graph_emb + ft
        return self.classify(graph_emb.unsqueeze(0))


# --- ported from MANDO sco_models/graph_utils.py ---
def get_length_2_metapath(g):
    begin_by = {}
    for mt in g.canonical_etypes:
        begin_by.setdefault(mt[0], []).append(mt)
    metapath_list = []
    for mt_0 in g.canonical_etypes:
        source, dest = mt_0[0], mt_0[-1]
        if source == dest:
            metapath_list.append([mt_0])
        if dest in begin_by:
            for mt_1 in begin_by[dest]:
                if mt_1 != mt_0 and mt_1[-1] == source:
                    metapath_list.append([mt_0, mt_1])
    return metapath_list


def reflect_graph(g):
    """Symmetrize a heterograph (add reverse edges), matching MANDO's reflect_graph."""
    data_dict = {}
    for cetype in g.canonical_etypes:
        src, dst = g.edges(etype=cetype)
        if cetype[0] == cetype[-1]:
            data_dict[cetype] = (torch.cat([src, dst]), torch.cat([dst, src]))
        else:
            rev = (cetype[-1], cetype[1] + "_r", cetype[0])
            data_dict[cetype] = (src, dst)
            data_dict[rev] = (dst, src)
    return dgl.heterograph(data_dict, num_nodes_dict={t: g.num_nodes(t) for t in g.ntypes})


def set_onehot_feat(g):
    for i, t in enumerate(g.ntypes):
        n = g.num_nodes(t)
        onehot = torch.zeros(n, len(g.ntypes))
        onehot[:, i] = 1.0
        g.nodes[t].data["feat"] = onehot
    return g


def make_mando_graphs(cache, items, lowcall=True):
    graphs, labels = [], []
    for fpath, label in items:
        G = cache.get(fpath)
        if G is None:
            continue
        g = nx_to_dgl(G, include_cross=True, lowcall=lowcall)
        g = reflect_graph(g)
        g = set_onehot_feat(g)
        graphs.append(g)
        labels.append(label)
    return graphs, torch.tensor(labels, dtype=torch.long)


def precompute_metapaths(graphs, meta_paths, device):
    """Precompute each graph's metapath reachable graphs + source features ONCE.

    metapath_reachable_graph is structure-only (independent of model weights),
    so unlike MANDO's per-forward cache this computes them a single time.
    Metapaths whose source node type is empty in a graph (e.g. `ext_sink` with
    0 nodes under lowcall=0) are skipped — the original MANDO never sees this
    because it operates on one global graph, but per-contract adaptation can.
    Returns info[i] = list of (layer_idx, mp_graph, feat).
    """
    info = []
    for g in graphs:
        gd = g.to(device)
        per = []
        for li, mp in enumerate(meta_paths):
            src_type = mp[0][0]
            if gd.num_nodes(src_type) == 0:
                continue
            mp_g = dgl.metapath_reachable_graph(gd, list(mp))
            feat = gd.nodes[src_type].data["feat"]
            per.append((li, mp_g, feat))
        info.append(per)
    return info


def train_eval(model_factory, meta_paths, in_size, graphs, labels, epochs, lr, device, seed=0):
    labels = labels.to(device)
    info = precompute_metapaths(graphs, meta_paths, device)
    cnt = Counter(labels.cpu().numpy().tolist())
    n_folds = max(2, min(5, min(cnt.values())))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    idx = np.arange(len(labels))
    accs, f1s = [], []
    for tr, te in skf.split(idx, labels.cpu().numpy()):
        model = model_factory(meta_paths, in_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            losses = 0.0
            for i in tr:
                logits = model(info[i])
                losses = losses + F.cross_entropy(logits, labels[i].unsqueeze(0))
            losses = losses / len(tr)
            losses.backward()
            opt.step()
        model.eval()
        preds, truths = [], []
        for i in te:
            with torch.no_grad():
                logits = model(info[i])
            preds.append(logits.argmax(1).item())
            truths.append(labels[i].item())
        accs.append(accuracy_score(truths, preds))
        f1s.append(f1_score(truths, preds, zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-dirs", nargs="+", required=True)
    ap.add_argument("--neg-dirs", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=130)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--lowcall", type=int, default=0)
    ap.add_argument("--cache", default="data/cache_full.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = collect_dataset(args.pos_dirs, args.neg_dirs, args.limit, args.seed)
    cache = build_and_cache(items, args.cache)
    graphs, labels = make_mando_graphs(cache, items, lowcall=bool(args.lowcall))

    rep = next((g for g in graphs if g.number_of_nodes() > 0), None)
    if rep is None:
        print("no graphs")
        return
    meta_paths = get_length_2_metapath(rep)
    in_size = len(rep.ntypes)
    print(f"MANDO: {len(graphs)} graphs, {len(meta_paths)} metapaths, in_size={in_size}, lowcall={args.lowcall}",
          flush=True)

    def factory(mps, insz):
        return MANDOClassifier(mps, insz, args.hidden, 2, args.heads, args.dropout)

    acc, f1 = train_eval(factory, meta_paths, in_size, graphs, labels, args.epochs, 1e-3, device, seed=args.seed)
    print(f"RESULT model=mando lowcall={args.lowcall} acc={acc:.4f} f1={f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
