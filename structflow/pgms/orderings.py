"""Block orderings for a compiled structure, and the statistics that grade them.

A :class:`~pgms.structure.BlockTriangularStructure` is a variable order plus a
dependency map. The compiler derives the second from the first, so the order is
the design variable: it fixes each block's estimation target, the depth of the
generation chain, and how much two compiled structures can share.

Everything here is generic over graphs. It parses no variable names and assumes
no application. Naming conventions -- an unrolled decision process, say, whose
nodes are ``s3`` and ``a0`` -- belong in the caller, which supplies an ordered
node sequence or a key function rather than asking this module to guess.
"""

from __future__ import annotations

import networkx as nx

__all__ = ["to_nx", "dependencies_for_order", "sequential_depth",
           "bisect_indices", "bisection_order", "tying_classes",
           "block_key", "reuse_fraction", "structure_stats"]


def to_nx(dag):
    """{node: [parents]} -> a networkx DiGraph."""
    G = nx.DiGraph()
    for node, parents in dag.items():
        G.add_node(node)
        for p in parents:
            G.add_edge(p, node)
    return G


def dependencies_for_order(dag, block_order):
    """Prune dependencies by d-separation on the original DAG.

    Mirrors step 3 of ``pgm_compiler.compile_inference_structure``: a block
    depends on a predecessor iff the two are *not* d-separated given all other
    predecessors. The result is the pruning-minimal dependency map for this
    order, which is what makes a block's target a well-defined conditional of
    the joint rather than an artefact of where the order happened to put it.
    """
    G = to_nx(dag)
    deps = {}
    for i, block in enumerate(block_order):
        predecessors = set(block_order[:i])
        required = [p for p in predecessors
                    if not nx.is_d_separator(G, {block}, {p}, predecessors - {p})]
        required.sort(key=block_order.index)
        deps[block] = required
    return deps


def sequential_depth(deps, block_order):
    """Longest chain in the block-dependency DAG.

    The number of block solves that cannot be parallelised: blocks at equal
    depth are conditionally independent given everything above them. This, not
    edge count, is what governs error accumulation along a conditioning chain,
    because only a directed chain gives an error the chance to be re-consumed
    as a condition.
    """
    depth = {}
    for b in block_order:
        depth[b] = 1 + max((depth[d] for d in deps[b]), default=-1)
    return max(depth.values()) + 1


def bisect_indices(pinned, lo=None, hi=None):
    """Midpoint-first emission order for the gaps between ``pinned`` indices.

    Given the already-placed indices, return the remaining indices in the order
    a bisection emits them: every gap's midpoint first, then the midpoints of
    the halves, and so on. Breadth-first across gaps, so the result is grouped
    by level and the k-th level has bracket width n/2^k.

    ``lo``/``hi`` extend the pinned set to the ends of the range when the
    caller wants the endpoints treated as pinned without listing them.
    """
    pins = sorted(set(pinned))
    if lo is not None and (not pins or pins[0] != lo):
        pins = [lo] + pins
    if hi is not None and (not pins or pins[-1] != hi):
        pins = pins + [hi]

    out = []
    intervals = [(pins[k], pins[k + 1]) for k in range(len(pins) - 1)]
    while intervals:
        nxt = []
        for a, b in intervals:
            if b - a < 2:
                continue
            mid = (a + b) // 2
            out.append(mid)
            nxt += [(a, mid), (mid, b)]
        intervals = nxt
    return out


def bisection_order(sequence, evidence=(), tail=()):
    """Midpoint-first order over a chain of variables.

    Args:
        sequence: the chain's variables, in chain order.
        evidence: variables to pin at the front. Conditioning by clamping is
            exact only when the evidence precedes everything else, so pinning
            it is a correctness requirement and not a heuristic.
        tail: variables to append after the chain, for nodes that are
            d-separated from the rest given their neighbours in the chain and
            can therefore wait until both brackets exist.

    Gives sequential depth O(log n) rather than O(n) while keeping each block's
    in-degree at two, because on a chain a bracketing pair d-separates the
    variable between them from everything else.
    """
    seq = list(sequence)
    idx = {v: i for i, v in enumerate(seq)}
    ev = [idx[v] for v in evidence if v in idx]

    order, seen = [], set()

    def emit(i):
        v = seq[i]
        if v not in seen:
            seen.add(v)
            order.append(v)

    for i in sorted(ev):
        emit(i)
    pinned = sorted(ev) or []
    if not pinned:
        pinned = [len(seq) - 1]
        emit(pinned[0])
    if pinned[0] != 0:
        emit(0)
        pinned = [0] + pinned
    if pinned[-1] != len(seq) - 1:
        emit(len(seq) - 1)
        pinned = pinned + [len(seq) - 1]

    for i in bisect_indices(pinned):
        emit(i)
    return order + [v for v in tail if v not in seen]


def tying_classes(deps, key_fn):
    """Group blocks by ``key_fn(block, deps[block])``.

    Weight tying is only sound within a class, and a class is only as good as
    the key. Two keys in use elsewhere: dependency-set equality, which makes
    the blocks' targets provably identical; and structural resemblance, which
    does not and needs its own justification.
    """
    classes = {}
    for b, ps in deps.items():
        classes.setdefault(key_fn(b, ps), []).append(b)
    return classes


def block_key(node, deps, dims=None):
    """The identity of a block's estimation target.

    In a valid structure a block's target is the conditional of the joint given
    exactly its dependency set, so two blocks with equal keys estimate the same
    object and may share one parameter vector outright.
    """
    d = tuple(deps)
    if dims is None:
        return (node, d)
    return (node, d, dims[node], tuple(dims[p] for p in d))


def reuse_fraction(dep_maps, dims=None):
    """Share of block slots that collapse across a set of compiled structures.

    ``dep_maps`` is one {node: deps} per structure. Requires no data and no
    training: the answer is fixed by the graph and the orders, so the cost of a
    query set can be priced before anything is initialised.
    """
    keys, slots = set(), 0
    for deps in dep_maps:
        for v, ps in deps.items():
            keys.add(block_key(v, ps, dims))
            slots += 1
    if not slots:
        return 0.0, 0, 0
    return 1.0 - len(keys) / slots, len(keys), slots


def structure_stats(deps, block_order, tying_key=None):
    """(edges, max in-degree, sequential depth, n_blocks[, n_tying_classes])."""
    stats = {
        "edges": sum(len(v) for v in deps.values()),
        "max_degree": max((len(v) for v in deps.values()), default=0),
        "depth": sequential_depth(deps, block_order),
        "blocks": len(deps),
    }
    if tying_key is not None:
        stats["tying_classes"] = len(tying_classes(deps, tying_key))
    return stats
