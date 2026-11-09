"""
Factory functions for creating Structural PGMs.

This module provides functions to generate PGMs with various structures:
    - Chain: x0 -> x1 -> x2 -> ... -> x_{n-1}
    - Fork: z -> x1, z -> x2, ...
    - Collider: x1 -> y, x2 -> y, ...
    - Random DAG: Random sparse lower-triangular structure

Each factory can be parameterized with:
    - A mechanism type (linear, nonlinear_parent, nonlinear_noise, universal)
    - Dimensions for each variable
    - Sparsity for random DAGs
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Callable, Optional, Literal

from structflow.pgms.structure import BlockTriangularStructure
from structflow.pgms.structural_pgm import StructuralPGM, Mechanism
from structflow.pgms.mechanisms import (
    LinearMechanism,
    NonlinearParentMechanism,
    NonlinearNoiseMechanism,
    UniversalMechanism,
    make_linear_mechanism,
    make_nonlinear_parent_mechanism,
    make_nonlinear_noise_mechanism,
    make_universal_mechanism,
)


# Type alias for mechanism constructors
MechanismType = Literal["linear", "nonlinear_parent", "nonlinear_noise", "universal"]


def get_mechanism_constructor(
    mechanism_type: MechanismType,
    **kwargs
) -> Callable[[List[str], List[int], int, jax.Array], Mechanism]:
    """
    Get a mechanism constructor function for the given type.
    
    Args:
        mechanism_type: One of "linear", "nonlinear_parent", "nonlinear_noise", "universal"
        **kwargs: Additional arguments passed to the mechanism constructor
        
    Returns:
        A function (parent_names, parent_dims, output_dim, key) -> Mechanism
    """
    if mechanism_type == "linear":
        ws = kwargs.get("weight_scale")
        ns = kwargs.get("noise_scale")
        rs = kwargs.get("root_scale")
        return lambda pn, pd, od, k: make_linear_mechanism(pn, pd, od, k,
                                                           weight_scale=ws, noise_scale=ns,
                                                           root_scale=rs)
    elif mechanism_type == "nonlinear_parent":
        hw = kwargs.get("hidden_width", 32)
        return lambda pn, pd, od, k: make_nonlinear_parent_mechanism(pn, pd, od, k, hidden_width=hw)
    elif mechanism_type == "nonlinear_noise":
        nb = kwargs.get("num_bins", 8)
        return lambda pn, pd, od, k: make_nonlinear_noise_mechanism(pn, pd, od, k, num_bins=nb)
    elif mechanism_type == "universal":
        nb = kwargs.get("num_bins", 8)
        hw = kwargs.get("hidden_width", 32)
        return lambda pn, pd, od, k: make_universal_mechanism(pn, pd, od, k, num_bins=nb, hidden_width=hw)
    else:
        raise ValueError(f"Unknown mechanism type: {mechanism_type}")


def make_chain_structure(
    n_nodes: int,
    dims: Optional[Dict[str, int]] = None,
    node_dim: int = 1
) -> BlockTriangularStructure:
    """
    Create a chain structure: x0 -> x1 -> ... -> x_{n-1}
    
    Args:
        n_nodes: Number of nodes in the chain
        dims: Optional dict mapping node names to dimensions
        node_dim: Default dimension for all nodes (used if dims is None)
        
    Returns:
        BlockTriangularStructure for the chain
    """
    node_names = [f"x{i}" for i in range(n_nodes)]
    
    dependencies = {}
    dependencies[node_names[0]] = []
    for i in range(1, n_nodes):
        dependencies[node_names[i]] = [node_names[i-1]]
    
    if dims is None:
        dims = {name: node_dim for name in node_names}
    
    return BlockTriangularStructure(
        block_order=node_names,
        dependencies=dependencies,
        dims=dims
    )


def make_fork_structure(
    n_children: int = 2,
    root_dim: int = 1,
    child_dim: int = 1
) -> BlockTriangularStructure:
    """
    Create a fork structure: z -> x1, z -> x2, ..., z -> x_n
    
    Args:
        n_children: Number of child nodes
        root_dim: Dimension of root node z
        child_dim: Dimension of each child node
        
    Returns:
        BlockTriangularStructure for the fork
    """
    node_names = ["z"] + [f"x{i}" for i in range(1, n_children + 1)]
    
    dependencies = {"z": []}
    for i in range(1, n_children + 1):
        dependencies[f"x{i}"] = ["z"]
    
    dims = {"z": root_dim}
    for i in range(1, n_children + 1):
        dims[f"x{i}"] = child_dim
    
    return BlockTriangularStructure(
        block_order=node_names,
        dependencies=dependencies,
        dims=dims
    )


def make_collider_structure(
    n_causes: int = 2,
    cause_dim: int = 1,
    effect_dim: int = 1
) -> BlockTriangularStructure:
    """
    Create a collider structure: x1 -> y, x2 -> y, ..., x_n -> y
    
    Args:
        n_causes: Number of cause nodes
        cause_dim: Dimension of each cause node
        effect_dim: Dimension of the effect node y
        
    Returns:
        BlockTriangularStructure for the collider
    """
    cause_names = [f"x{i}" for i in range(1, n_causes + 1)]
    node_names = cause_names + ["y"]
    
    dependencies = {name: [] for name in cause_names}
    dependencies["y"] = cause_names
    
    dims = {name: cause_dim for name in cause_names}
    dims["y"] = effect_dim
    
    return BlockTriangularStructure(
        block_order=node_names,
        dependencies=dependencies,
        dims=dims
    )


def make_random_dag_structure(
    n_nodes: int,
    sparsity: float = 0.3,
    key: jax.Array = None,
    dims: Optional[Dict[str, int]] = None,
    node_dim: int = 1
) -> BlockTriangularStructure:
    """
    Create a random DAG structure with specified sparsity.
    
    The DAG is lower-triangular (node i can only have parents j < i).
    
    Args:
        n_nodes: Number of nodes
        sparsity: Fraction of possible edges to remove (0 = dense, 1 = empty)
        key: Random key (required)
        dims: Optional dict mapping node names to dimensions
        node_dim: Default dimension for all nodes
        
    Returns:
        BlockTriangularStructure for the random DAG
    """
    if key is None:
        raise ValueError("key is required for random DAG generation")
    
    node_names = [f"x{i}" for i in range(n_nodes)]
    
    # Number of possible edges in lower triangular DAG
    n_possible = (n_nodes * (n_nodes - 1)) // 2
    density = 1.0 - sparsity
    n_edges = int(jnp.round(n_possible * density))
    n_edges = int(jnp.clip(n_edges, 0, n_possible))
    
    # Get all valid (child, parent) pairs
    rows, cols = jnp.tril_indices(n_nodes, k=-1)
    
    # Randomly select edges
    if n_edges > 0 and n_possible > 0:
        chosen_indices = random.choice(key, n_possible, shape=(n_edges,), replace=False)
        
        # Build adjacency
        dependencies = {name: [] for name in node_names}
        for idx in chosen_indices:
            child = node_names[int(rows[idx])]
            parent = node_names[int(cols[idx])]
            dependencies[child].append(parent)
        
        # Sort parents for consistency
        for name in node_names:
            dependencies[name] = sorted(dependencies[name])
    else:
        dependencies = {name: [] for name in node_names}
    
    if dims is None:
        dims = {name: node_dim for name in node_names}
    
    return BlockTriangularStructure(
        block_order=node_names,
        dependencies=dependencies,
        dims=dims
    )


def make_dense_structure(
    n_nodes: int,
    dims: Optional[Dict[str, int]] = None,
    node_dim: int = 1
) -> BlockTriangularStructure:
    """
    Create a fully connected DAG (each node depends on all previous nodes).
    
    Args:
        n_nodes: Number of nodes
        dims: Optional dict mapping node names to dimensions
        node_dim: Default dimension for all nodes
        
    Returns:
        BlockTriangularStructure for the dense DAG
    """
    node_names = [f"x{i}" for i in range(n_nodes)]
    
    dependencies = {}
    for i, name in enumerate(node_names):
        dependencies[name] = node_names[:i]  # All previous nodes
    
    if dims is None:
        dims = {name: node_dim for name in node_names}
    
    return BlockTriangularStructure(
        block_order=node_names,
        dependencies=dependencies,
        dims=dims
    )


# --- PGM Factories ---

def make_structural_pgm(
    structure: BlockTriangularStructure,
    mechanism_type: MechanismType,
    key: jax.Array,
    **mechanism_kwargs
) -> StructuralPGM:
    """
    Create a StructuralPGM from a structure and mechanism type.
    
    Args:
        structure: BlockTriangularStructure defining the DAG
        mechanism_type: Type of mechanism to use
        key: Random key for initialization
        **mechanism_kwargs: Additional arguments for mechanism constructors
        
    Returns:
        StructuralPGM instance
    """
    mechanism_fn = get_mechanism_constructor(mechanism_type, **mechanism_kwargs)
    
    mechanisms = {}
    keys = random.split(key, len(structure.block_order))
    
    for i, node in enumerate(structure.block_order):
        parent_names = structure.dependencies.get(node, [])
        parent_dims = [structure.dims[p] for p in parent_names]
        output_dim = structure.dims[node]
        
        mechanisms[node] = mechanism_fn(parent_names, parent_dims, output_dim, keys[i])
    
    return StructuralPGM(structure, mechanisms)


def make_chain_pgm(
    n_nodes: int,
    mechanism_type: MechanismType,
    key: jax.Array,
    node_dim: int = 1,
    **mechanism_kwargs
) -> StructuralPGM:
    """
    Create a chain PGM.
    
    Args:
        n_nodes: Number of nodes in the chain
        mechanism_type: Type of mechanism to use
        key: Random key
        node_dim: Dimension of each node
        **mechanism_kwargs: Additional mechanism arguments
        
    Returns:
        StructuralPGM with chain structure
    """
    structure = make_chain_structure(n_nodes, node_dim=node_dim)
    return make_structural_pgm(structure, mechanism_type, key, **mechanism_kwargs)


def make_fork_pgm(
    n_children: int,
    mechanism_type: MechanismType,
    key: jax.Array,
    root_dim: int = 1,
    child_dim: int = 1,
    **mechanism_kwargs
) -> StructuralPGM:
    """
    Create a fork PGM (z -> x1, z -> x2, ...).
    
    Args:
        n_children: Number of child nodes
        mechanism_type: Type of mechanism to use
        key: Random key
        root_dim: Dimension of root node
        child_dim: Dimension of each child
        **mechanism_kwargs: Additional mechanism arguments
        
    Returns:
        StructuralPGM with fork structure
    """
    structure = make_fork_structure(n_children, root_dim, child_dim)
    return make_structural_pgm(structure, mechanism_type, key, **mechanism_kwargs)


def make_collider_pgm(
    n_causes: int,
    mechanism_type: MechanismType,
    key: jax.Array,
    cause_dim: int = 1,
    effect_dim: int = 1,
    **mechanism_kwargs
) -> StructuralPGM:
    """
    Create a collider PGM (x1 -> y, x2 -> y, ...).
    
    Args:
        n_causes: Number of cause nodes
        mechanism_type: Type of mechanism to use
        key: Random key
        cause_dim: Dimension of each cause
        effect_dim: Dimension of effect
        **mechanism_kwargs: Additional mechanism arguments
        
    Returns:
        StructuralPGM with collider structure
    """
    structure = make_collider_structure(n_causes, cause_dim, effect_dim)
    return make_structural_pgm(structure, mechanism_type, key, **mechanism_kwargs)


def make_random_pgm(
    n_nodes: int,
    mechanism_type: MechanismType,
    key: jax.Array,
    sparsity: float = 0.3,
    node_dim: int = 1,
    **mechanism_kwargs
) -> StructuralPGM:
    """
    Create a random sparse PGM.
    
    Args:
        n_nodes: Number of nodes
        mechanism_type: Type of mechanism to use
        key: Random key
        sparsity: Fraction of edges to remove (0 = dense)
        node_dim: Dimension of each node
        **mechanism_kwargs: Additional mechanism arguments
        
    Returns:
        StructuralPGM with random DAG structure
    """
    k1, k2 = random.split(key)
    structure = make_random_dag_structure(n_nodes, sparsity, k1, node_dim=node_dim)
    return make_structural_pgm(structure, mechanism_type, k2, **mechanism_kwargs)


def make_dense_pgm(
    n_nodes: int,
    mechanism_type: MechanismType,
    key: jax.Array,
    node_dim: int = 1,
    **mechanism_kwargs
) -> StructuralPGM:
    """
    Create a fully connected (dense) PGM.
    
    Args:
        n_nodes: Number of nodes
        mechanism_type: Type of mechanism to use
        key: Random key
        node_dim: Dimension of each node
        **mechanism_kwargs: Additional mechanism arguments
        
    Returns:
        StructuralPGM with dense DAG structure
    """
    structure = make_dense_structure(n_nodes, node_dim=node_dim)
    return make_structural_pgm(structure, mechanism_type, key, **mechanism_kwargs)


# --- Convenience aliases for backward compatibility ---

def make_linear_chain_pgm(n_nodes: int, key: jax.Array, node_dim: int = 1) -> StructuralPGM:
    """Create a linear Gaussian chain PGM."""
    return make_chain_pgm(n_nodes, "linear", key, node_dim=node_dim)


def make_nonlinear_chain_pgm(n_nodes: int, key: jax.Array, node_dim: int = 1, hidden_width: int = 32) -> StructuralPGM:
    """Create a nonlinear (parent-dependent mean/var) chain PGM."""
    return make_chain_pgm(n_nodes, "nonlinear_parent", key, node_dim=node_dim, hidden_width=hidden_width)


def make_universal_chain_pgm(n_nodes: int, key: jax.Array, node_dim: int = 1, num_bins: int = 8) -> StructuralPGM:
    """Create a fully nonlinear chain PGM using splines."""
    return make_chain_pgm(n_nodes, "universal", key, node_dim=node_dim, num_bins=num_bins)


def make_linear_random_pgm(n_nodes: int, key: jax.Array, sparsity: float = 0.3) -> StructuralPGM:
    """Create a linear Gaussian random PGM."""
    return make_random_pgm(n_nodes, "linear", key, sparsity=sparsity)


def make_nonlinear_random_pgm(n_nodes: int, key: jax.Array, sparsity: float = 0.3, hidden_width: int = 32) -> StructuralPGM:
    """Create a nonlinear random PGM."""
    return make_random_pgm(n_nodes, "nonlinear_parent", key, sparsity=sparsity, hidden_width=hidden_width)


def make_universal_random_pgm(n_nodes: int, key: jax.Array, sparsity: float = 0.3, num_bins: int = 8) -> StructuralPGM:
    """Create a fully nonlinear random PGM."""
    return make_random_pgm(n_nodes, "universal", key, sparsity=sparsity, num_bins=num_bins)
