from typing import List, Dict, Callable, Optional, Set, Tuple, Union
import networkx as nx

from structflow.pgms.structure import BlockTriangularStructure, topological_sort

def dag_to_structure(
    pgm_dag: Dict[str, List[str]], 
    dims: Optional[Dict[str, int]] = None
) -> BlockTriangularStructure:
    """
    Parses a PGM DAG into a causal BlockTriangularStructure.
    
    The resulting structure maintains the original dependencies and 
    uses a topological sort for the block order. This is suitable
    for causal modeling where we want to generate effects from causes.
    
    Args:
        pgm_dag: Dictionary mapping each variable to its parents in the PGM.
        dims: Dictionary mapping each variable to its dimension.
        
    Returns:
        BlockTriangularStructure with topological order and DAG dependencies.
    """
    variables = set(pgm_dag.keys())
    if dims is None:
        dims = {v: 1 for v in variables}
    
    # Causal mode: use topological order and original DAG parents
    try:
        block_order = topological_sort(pgm_dag)
    except ValueError as e:
        raise nx.NetworkXUnfeasible(str(e))
        
    return BlockTriangularStructure(
        block_order=block_order,
        dependencies=pgm_dag,
        dims=dims
    )

def compile_inference_structure(
    causal_structure: BlockTriangularStructure,
    targets: Optional[Set[str]] = None,
    conditions: Optional[Set[str]] = None
) -> BlockTriangularStructure:
    """
    Compiles a causal structure into an inference structure for a given query.
    
    This uses a moralization and d-separation strategy to determine an optimal 
    ordering and dependency set for the given targets and conditions. 
    The resulting structure starts from conditions and traverses to targets, 
    accounting for explaining-away effects via moralization.
    
    Args:
        causal_structure: The underlying causal BlockTriangularStructure (DAG).
        targets: Set of variables we want to infer/sample.
        conditions: Set of variables we observe.
        
    Returns:
        BlockTriangularStructure optimized for the specified inference query.
    """
    pgm_dag = causal_structure.dependencies
    variables = set(pgm_dag.keys())
    dims = causal_structure.dims

    # 1. Validation
    if targets is None:
        targets = set()
    if conditions is None:
        conditions = set()

    if not targets.issubset(variables):
        raise ValueError(f"Targets must be variables in the PGM. Got {targets}. Valid variables: {variables}")
    if not conditions.issubset(variables):
        raise ValueError(f"Conditions must be variables in the PGM. Got {conditions}. Valid variables: {variables}")
    if not targets.isdisjoint(conditions):
        raise ValueError(f"Targets and conditions must be disjoint. Got {targets} and {conditions}. Valid variables: {variables}")
        
    # 2. Determine Block Order
    # Build Moral Graph
    G_dag = nx.DiGraph()
    for node, parents in pgm_dag.items():
        G_dag.add_node(node)
        for parent in parents:
            G_dag.add_edge(parent, node)
            
    G_moral = G_dag.to_undirected()
    
    # Marry parents
    for node, parents in pgm_dag.items():
        if len(parents) > 1:
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    G_moral.add_edge(parents[i], parents[j])
    
    # 2. Determine Block Order
    # We want to traverse "backwards" from conditions/children to targets/parents.
    # We use reverse topological order as the primary heuristic.
    try:
        topo_order = topological_sort(pgm_dag)
    except ValueError as e:
        raise nx.NetworkXUnfeasible(str(e))
        
    topo_index = {node: i for i, node in enumerate(topo_order)}
    
    # BFS Ordering from Conditions
    # We sort conditions by reverse topological order (descendants first)
    ordered_conditions = sorted(list(conditions), key=lambda x: topo_index[x], reverse=True)
    visited = set(ordered_conditions)
    block_order = list(ordered_conditions)
    
    queue = list(ordered_conditions)
    while queue:
        current = queue.pop(0)
        # Explore neighbors in reverse topological order
        neighbors = sorted(list(G_moral.neighbors(current)), key=lambda x: topo_index[x], reverse=True)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                block_order.append(neighbor)
                queue.append(neighbor)
                
    # Append remaining variables using Reverse Topological Sort
    # (for nodes not reachable from conditions in the moral graph)
    for node in reversed(topo_order):
        if node not in visited:
            block_order.append(node)
            visited.add(node)
            
    # 3. Determine Dependencies
    dependencies = {}
    for i, block in enumerate(block_order):
        predecessors = set(block_order[:i])
        required_deps = []
        
        for pred in predecessors:
            conditioning_set = predecessors - {pred}
            
            # Check d-separation on Original DAG
            if not nx.is_d_separator(G_dag, {block}, {pred}, conditioning_set):
                required_deps.append(pred)
        
        required_deps.sort(key=lambda x: block_order.index(x))
        dependencies[block] = required_deps
        
    return BlockTriangularStructure(
        block_order=block_order,
        dependencies=dependencies,
        dims=dims
    )

class PGMCompiler:
    """
    Compiles a PGM DAG and a target query into a BlockTriangularStructure.
    Provides backward compatibility for the older API while supporting new modes.
    """
    
    def __init__(self, pgm_dag: Dict[str, List[str]], dims: Optional[Dict[str, int]] = None):
        """
        Args:
            pgm_dag: Dictionary mapping each variable to its parents in the PGM.
            dims: Dictionary mapping each variable to its dimension.
        """
        self.pgm_dag = pgm_dag
        self.dims = dims

    def compile_causal(self) -> BlockTriangularStructure:
        """
        Returns the causal structure (Topological sort of original DAG).
        """
        return dag_to_structure(self.pgm_dag, self.dims)

    def compile_inference(
        self, 
        targets: Optional[Set[str]] = None, 
        conditions: Optional[Set[str]] = None
    ) -> BlockTriangularStructure:
        """
        Compiles the structure for inference (starting from conditions).
        """
        # We first build the causal structure to get dims and dependencies handled consistently
        causal = dag_to_structure(self.pgm_dag, self.dims)
        return compile_inference_structure(
            causal, 
            targets=targets, 
            conditions=conditions
        )
