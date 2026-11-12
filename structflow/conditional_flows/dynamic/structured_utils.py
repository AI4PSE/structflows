import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union
from structflow.pgms.structure import Structure

def get_block_offsets(block_dims: Dict[str, int], block_order: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Computes start and end indices for each block in a concatenated vector.
    
    Args:
        block_dims: Dictionary mapping block name to its dimension.
        block_order: List of block names in the order they appear in the vector.
        
    Returns:
        Dictionary mapping block name to (start_index, end_index).
    """
    offsets = {}
    current = 0
    for block in block_order:
        dim = block_dims[block]
        offsets[block] = (current, current + dim)
        current += dim
    return offsets

def build_structure_mask(
    structure: Structure,
    in_block_dims: Dict[str, int],
    out_block_dims: Dict[str, int],
    in_block_order: Optional[List[str]] = None,
    out_block_order: Optional[List[str]] = None,
    extra_dependencies: Optional[Dict[str, List[str]]] = None,
    include_self: bool = True
) -> jnp.ndarray:
    """
    Creates a binary mask for block-triangular dependencies.
    
    Args:
        structure: The dependency structure defining parents.
        in_block_dims: Dimensions of input blocks.
        out_block_dims: Dimensions of output blocks.
        in_block_order: Order of blocks in the input vector. Defaults to structure.block_order.
        out_block_order: Order of blocks in the output vector. Defaults to structure.block_order.
        extra_dependencies: Additional dependencies not in the structure (e.g. {'A': ['_TIME']}).
        include_self: Whether each block depends on itself (diagonal blocks).
        
    Returns:
        A JAX array of shape (sum(out_block_dims), sum(in_block_dims)).
    """
    if in_block_order is None:
        in_block_order = structure.block_order
    if out_block_order is None:
        out_block_order = structure.block_order
        
    n_in = sum(in_block_dims.values())
    n_out = sum(out_block_dims.values())
    
    mask = np.zeros((n_out, n_in))
    
    in_offsets = get_block_offsets(in_block_dims, in_block_order)
    out_offsets = get_block_offsets(out_block_dims, out_block_order)
    
    for out_block in out_block_order:
        if out_block not in out_offsets: continue
        
        # Determine which input blocks this output block depends on
        parents = list(structure.dependencies.get(out_block, []))
        if extra_dependencies and out_block in extra_dependencies:
            parents.extend(extra_dependencies[out_block])
            
        allowed_inputs = set(parents)
        if include_self and out_block in in_block_dims:
            allowed_inputs.add(out_block)
            
        o_start, o_end = out_offsets[out_block]
        
        for in_block in in_block_order:
            if in_block in allowed_inputs:
                i_start, i_end = in_offsets[in_block]
                mask[o_start:o_end, i_start:i_end] = 1.0
                
    return jnp.array(mask)

def get_sparse_indices(mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (row_indices, col_indices) for non-zero elements in mask."""
    return jnp.where(mask > 0)
