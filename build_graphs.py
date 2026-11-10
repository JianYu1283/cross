"""Build heterogeneous smart-contract graphs via slither.

Produces, per source file, a networkx MultiDiGraph with heterogeneous
node_type / edge_type attributes, plus a cross-contract option.

Node types:
  'function'            -- function-level entry node
  'stmt'                -- CFG statement/expression node
  'ext_sink'            -- unresolved low-level external call (call/delegatecall/send/transfer)
Edge types:
  'cfg'                 -- intra-function control flow (next / if_true / if_false)
  'internal_call'       -- function -> internal function
  'external_call'       -- function -> external contract function (resolved, function-level)
  'extcall_sink'        -- stmt -> ext_sink (low-level call marked)
  'delegatecall'        -- function -> external function via delegatecall (resolved high-level)
"""
import os
import re
import networkx as nx
from slither.slither import Slither
from slither.core.declarations.function import Function

SOLC_DIR = os.path.expanduser("~/.solc-select/artifacts")

_EXT_CALL_TOKENS = ["call", "delegatecall", "callcode", "send(", "transfer("]
_DELEGATE_TOKENS = ["delegatecall", "callcode"]


def pragma_version(src_path):
    with open(src_path, "r", errors="ignore") as f:
        txt = f.read(2000)
    m = re.search(r"pragma\s+solidity\s*([^;]+);", txt)
    if not m:
        return None
    return m.group(1).strip()


def _solc_path(v):
    return os.path.join(SOLC_DIR, "solc-" + v, "solc-" + v)


def _installed_solc():
    vers = []
    if os.path.isdir(SOLC_DIR):
        for d in os.listdir(SOLC_DIR):
            if d.startswith("solc-") and os.path.exists(os.path.join(SOLC_DIR, d, d)):
                vers.append(d[len("solc-"):])
    return sorted(vers, key=lambda s: [int(x) for x in s.split(".")])


def resolve_solc(src_path):
    ver = pragma_version(src_path) or ""
    installed = _installed_solc()
    if not installed:
        return None
    # exact pragma (e.g. "0.4.24")
    m = re.match(r"^(\d+\.\d+\.\d+)$", ver)
    if m and m.group(1) in installed:
        return _solc_path(m.group(1))
    # caret / range: pick highest installed with same major.minor, else same major
    mm = re.search(r"(\d+\.\d+)", ver)
    if mm:
        maj_min = mm.group(1)
        cands = [v for v in installed if v.startswith(maj_min + ".")]
        if cands:
            return _solc_path(cands[-1])
        maj = maj_min.split(".")[0]
        cands = [v for v in installed if v.startswith(maj + ".")]
        if cands:
            return _solc_path(cands[-1])
    return _solc_path(installed[-1])


def _fn_id(contract, fn):
    return f"{contract.name}.{fn.full_name}"


def _is_external_call_expr(expr):
    s = str(expr)
    return any(t in s for t in _EXT_CALL_TOKENS)


def _is_delegatecall_expr(expr):
    s = str(expr)
    return any(t in s for t in _DELEGATE_TOKENS)


def build_graph(src_path, cross_contract=True):
    """Parse one source file and return an nx.MultiDiGraph (or None on failure)."""
    solc = resolve_solc(src_path)
    if solc is None:
        return None
    try:
        slither = Slither(src_path, solc=solc)
    except Exception:
        return None

    G = nx.MultiDiGraph()
    fname = os.path.basename(src_path)

    for c in slither.contracts:
        for fn in c.functions + c.modifiers:
            fnid = _fn_id(c, fn)
            G.add_node(fnid, node_type="function", label=fnid,
                       function_fullname=fn.full_name, contract_name=c.name,
                       source_file=fname, payable=bool(fn.payable),
                       constructor=fn.is_constructor, fallback=fn.is_fallback,
                       visibility=str(fn.visibility))

            # --- CFG statement nodes ---
            prev = fnid  # entry: function -> first stmt
            for node in fn.nodes:
                sid = f"{fnid}::{node.node_id}"
                expr = str(node.expression) if node.expression else ""
                G.add_node(sid, node_type="stmt", label=expr,
                           stmt_type=str(node.type), expression=expr,
                           is_ext_call=_is_external_call_expr(expr),
                           is_delegate=_is_delegatecall_expr(expr),
                           function_fullname=fn.full_name, contract_name=c.name,
                           source_file=fname)
                G.add_edge(prev, sid, edge_type="cfg")
                prev = sid
                for s in node.sons:
                    if s is None:
                        continue
                    t = f"{fnid}::{s.node_id}"
                    if t not in G:
                        G.add_node(t, node_type="stmt", label=str(s.expression) if s.expression else "",
                                   stmt_type=str(s.type), expression=str(s.expression) if s.expression else "",
                                   is_ext_call=_is_external_call_expr(str(s.expression) if s.expression else ""),
                                   is_delegate=_is_delegatecall_expr(str(s.expression) if s.expression else ""),
                                   function_fullname=fn.full_name, contract_name=c.name, source_file=fname)
                    G.add_edge(sid, t, edge_type="cfg")
                if node.expression is not None and _is_external_call_expr(expr):
                    sink = f"{sid}::sink"
                    G.add_node(sink, node_type="ext_sink", label=expr,
                               is_delegate=_is_delegatecall_expr(expr),
                               function_fullname=fn.full_name, contract_name=c.name,
                               source_file=fname)
                    G.add_edge(sid, sink, edge_type="extcall_sink")

            # --- internal calls ---
            for callee in fn.internal_calls:
                if isinstance(callee, Function):
                    callee_id = _fn_id(callee.contract_declarer, callee)
                    G.add_edge(fnid, callee_id, edge_type="internal_call")

            # --- external (high-level) calls ---
            for (ec, ef) in fn.high_level_calls:
                if hasattr(ef, "function"):
                    ef = ef.function  # slither >=0.11 wraps in HighLevelCall
                if not isinstance(ef, Function):
                    continue
                eid = _fn_id(ec, ef)
                if eid not in G:
                    G.add_node(eid, node_type="function", label=eid,
                               function_fullname=ef.full_name, contract_name=ec.name,
                               source_file=fname, payable=bool(ef.payable),
                               constructor=ef.is_constructor, fallback=ef.is_fallback,
                               visibility=str(ef.visibility))
                et = "external_call"
                for node in fn.nodes:
                    if node.expression is not None and _is_delegatecall_expr(str(node.expression)):
                        et = "delegatecall"
                        break
                G.add_edge(fnid, eid, edge_type=et)

    return G


def build_graph_pair(main_src, callee_src, cross_contract=True):
    """Build a cross-contract graph: main + an explicitly-paired callee file."""
    g1 = build_graph(main_src, cross_contract=cross_contract)
    g2 = build_graph(callee_src, cross_contract=cross_contract)
    if g1 is None:
        return None
    G = nx.disjoint_union(g1, g2) if g2 is not None else g1
    return G


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    if src is None:
        src = "data/reentrancy/source_code/Bank.sol"
    G = build_graph(src)
    if G is None:
        print("FAILED to parse", src)
        sys.exit(1)
    from collections import Counter
    nt = Counter(d["node_type"] for _, d in G.nodes(data=True))
    et = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    print("file:", os.path.basename(src))
    print("nodes:", dict(nt))
    print("edges:", dict(et))
    print("num_nodes:", G.number_of_nodes(), "num_edges:", G.number_of_edges())
