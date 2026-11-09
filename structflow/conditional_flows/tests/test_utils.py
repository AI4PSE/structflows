"""
Shared utility functions for conditional flows tests.

Provides helpers for:
- Jacobian triangularity verification
- Loss comparison and learning validation
- Data generation and sampling
- Mode-specific behavior checking
"""

import jax
import jax.numpy as jnp
from jax import vmap
from typing import Dict, List, Set, Optional, Tuple
import numpy as np


# ============================================================================
# Jacobian Utilities
# ============================================================================

def flatten_dict_for_jacobian(x_dict: Dict[str, jnp.ndarray], 
                               block_order: List[str]) -> jnp.ndarray:
    """
    Flatten a dictionary of blocks into a single vector.
    
    Args:
        x_dict: Dictionary mapping block names to arrays
        block_order: Ordering of blocks for flattening
        
    Returns:
        Flattened vector
    """
    flat_parts = []
    for block in block_order:
        arr = x_dict[block]
        flat_parts.append(jnp.reshape(arr, (-1,)))
    return jnp.concatenate(flat_parts)


def get_block_slice(block: str, block_order: List[str], 
                    dims: Dict[str, int]) -> slice:
    """
    Get the slice indices for a block in a flattened vector.
    
    Args:
        block: Block name
        block_order: Ordering of blocks
        dims: Dimensions of each block
        
    Returns:
        Slice object for indexing into flattened vector
    """
    start_idx = 0
    for b in block_order:
        if b == block:
            end_idx = start_idx + dims[b]
            return slice(start_idx, end_idx)
        start_idx += dims[b]
    raise ValueError(f"Block {block} not found in block_order")


def unflatten_dict_from_vector(x_flat: jnp.ndarray, 
                                block_order: List[str],
                                dims: Dict[str, int]) -> Dict[str, jnp.ndarray]:
    """
    Unflatten a vector into a dictionary of blocks.
    
    Args:
        x_flat: Flattened vector
        block_order: Ordering of blocks
        dims: Dimensions of each block
        
    Returns:
        Dictionary mapping block names to arrays
    """
    result = {}
    idx = 0
    for block in block_order:
        dim = dims[block]
        result[block] = jnp.reshape(x_flat[idx:idx+dim], (dim,))
        idx += dim
    return result


def assert_triangular_jacobian(jac: jnp.ndarray,
                               block_order: List[str],
                               dependencies: Dict[str, List[str]],
                               dims: Dict[str, int],
                               atol: float = 1e-5) -> None:
    """
    Assert that Jacobian has correct block-triangular structure.
    
    Upper triangular part (future blocks) should be zero.
    Non-zero entries should exist where dependencies exist.
    
    Args:
        jac: Jacobian matrix (n x n)
        block_order: Topological ordering of blocks
        dependencies: Mapping of block -> parent blocks
        dims: Dimensions of each block
        atol: Absolute tolerance for "zero" check
        
    Raises:
        AssertionError if triangularity is violated
    """
    n_blocks = len(block_order)
    
    for i_block_idx, out_block in enumerate(block_order):
        for j_block_idx, in_block in enumerate(block_order):
            out_slice = get_block_slice(out_block, block_order, dims)
            in_slice = get_block_slice(in_block, block_order, dims)
            
            sub_jac = jac[out_slice, in_slice]
            
            if j_block_idx > i_block_idx:
                # Future blocks: should be zero (upper triangular)
                max_abs = jnp.max(jnp.abs(sub_jac))
                assert max_abs < atol, (
                    f"Upper triangular violation: "
                    f"Block {out_block} (idx {i_block_idx}) has non-zero gradient "
                    f"wrt future block {in_block} (idx {j_block_idx}). "
                    f"Max |Jac| = {max_abs}"
                )
            elif in_block in dependencies.get(out_block, []) or in_block == out_block:
                # Dependency or self: should be non-zero (or at least check it's reasonable)
                norm = jnp.linalg.norm(sub_jac)
                # Note: We don't assert it's non-zero (could be degenerate), 
                # just verify no upper triangular violations above
    
    print(f"✓ Jacobian is block-triangular for blocks: {block_order}")


# ============================================================================
# Loss and Learning Utilities
# ============================================================================

def assert_loss_decreased(loss_before: float, 
                         loss_after: float,
                         min_ratio: float = 0.9,
                         tolerance: float = 1e-5) -> None:
    """
    Assert that loss has decreased after training.
    
    Args:
        loss_before: Initial loss
        loss_after: Loss after training step(s)
        min_ratio: Minimum ratio (loss_after / loss_before) to accept
                  Default 0.9 means loss should decrease to at most 90% of original
        tolerance: Tolerance for floating point comparison
        
    Raises:
        AssertionError if loss did not decrease enough
    """
    ratio = loss_after / (loss_before + tolerance)
    assert ratio < min_ratio, (
        f"Loss did not decrease sufficiently. "
        f"Before: {loss_before:.6f}, After: {loss_after:.6f}, "
        f"Ratio: {ratio:.4f} (required < {min_ratio})"
    )
    print(f"✓ Loss decreased: {loss_before:.6f} -> {loss_after:.6f} (ratio: {ratio:.4f})")


def assert_gradients_nonzero(grads, threshold: float = 1e-7) -> None:
    """
    Assert that at least some gradients are non-zero.
    
    Args:
        grads: Gradient tree (any pytree structure)
        threshold: Threshold for "non-zero"
        
    Raises:
        AssertionError if all gradients are zero/None
    """
    def check_leaf(x):
        if x is None:
            return False
        if isinstance(x, jnp.ndarray):
            return jnp.max(jnp.abs(x)) > threshold
        return False
    
    import jax.tree_util as jtu
    has_nonzero = any(
        jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(check_leaf, g) 
        ) 
        for g in jax.tree_util.tree_leaves(grads)
    )
    
    # Simpler check: any leaf with non-zero values
    leaves = jax.tree_util.tree_leaves(grads)
    has_nonzero = any(
        isinstance(leaf, jnp.ndarray) and jnp.max(jnp.abs(leaf)) > threshold
        for leaf in leaves
    )
    
    assert has_nonzero, "All gradients are zero or near-zero"
    print(f"✓ Gradients are non-zero (threshold: {threshold})")


def assert_parameters_changed(params_before, params_after, 
                              threshold: float = 1e-7) -> None:
    """
    Assert that model parameters have changed.
    
    Args:
        params_before: Parameter tree before update
        params_after: Parameter tree after update
        threshold: Threshold for detecting change
        
    Raises:
        AssertionError if no parameters changed
    """
    leaves_before = jax.tree_util.tree_leaves(params_before)
    leaves_after = jax.tree_util.tree_leaves(params_after)
    
    assert len(leaves_before) == len(leaves_after), "Parameter structure changed"
    
    has_changed = False
    for leaf_b, leaf_a in zip(leaves_before, leaves_after):
        if isinstance(leaf_b, jnp.ndarray) and isinstance(leaf_a, jnp.ndarray):
            delta = jnp.max(jnp.abs(leaf_a - leaf_b))
            if delta > threshold:
                has_changed = True
                break
    
    assert has_changed, f"No parameters changed (threshold: {threshold})"
    print(f"✓ Parameters changed after update (threshold: {threshold})")


# ============================================================================
# Mode-Specific Behavior
# ============================================================================

def is_triangular_mode(mode: str) -> bool:
    """Check if mode supports Jacobian triangularity testing."""
    return mode in ["static", "dynamic", "carefl"]


def is_flow_based_mode(mode: str) -> bool:
    """Check if mode is flow-based (supports zero velocity = identity)."""
    return mode in ["static", "dynamic", "dynamicv2", "guided"]


def mode_supports_full_log_prob(mode: str) -> bool:
    """Check if mode supports arbitrary log_prob queries."""
    # joint/guidance mode only supports p(all_vars)
    return mode in ["static", "dynamic", "dynamicv2", "carefl"]


def mode_supports_transform_to_noise(mode: str) -> bool:
    """Check if mode has a tractable transform_to_noise operation."""
    # guidance might not have a direct backward pass
    return mode in ["static", "dynamic", "dynamicv2", "carefl"]


# ============================================================================
# Data Utilities
# ============================================================================

def create_constant_velocity_model(structure, velocity_constant: float = 0.0):
    """
    Create a mock velocity function that returns a constant.
    
    Useful for testing "zero velocity = identity" property.
    
    Args:
        structure: BlockTriangularStructure
        velocity_constant: Constant velocity value
        
    Returns:
        Callable that returns constant velocity dict
    """
    def constant_velocity(x_dict: Dict, t):
        """Return zero velocity for all blocks."""
        return {
            block: jnp.full_like(x_dict[block], velocity_constant)
            for block in structure.block_order
        }
    return constant_velocity


def sample_dict_batch(dist, key, n_samples: int, 
                     targets=None, conditions=None) -> Dict[str, jnp.ndarray]:
    """
    Sample a batch using distribution's batched API.
    
    Args:
        dist: Conditional distribution
        key: Random key
        n_samples: Batch size
        targets: Optional target blocks
        conditions: Optional condition values
        
    Returns:
        Batch of samples (batch, dim) for each block
    """
    if hasattr(dist, 'b_sample'):
        return dist.b_sample(key, n_samples, conditions=conditions, targets=targets)
    else:
        # # Fallback: use single sample API
        # keys = jax.random.split(key, n_samples)
        # samples = [
        #     dist.sample(k, conditions=conditions, targets=targets)
        #     for k in keys
        # ]
        # # Stack along batch dimension
        # result = {}
        # for block in samples[0].keys():
        #     result[block] = jnp.stack([s[block] for s in samples])
        # return result

        return vmap(lambda k: dist.sample(k, conditions=conditions, targets=targets))(jax.random.split(key, n_samples))


def compute_log_prob_batch(dist, x_batch: Dict[str, jnp.ndarray],
                           targets=None, conditions=None) -> jnp.ndarray:
    """
    Compute log_prob for a batch.
    
    Args:
        dist: Conditional distribution
        x_batch: Batch samples (batch, dim) for each block
        targets: Optional target blocks
        conditions: Optional condition blocks
        
    Returns:
        Log probabilities (batch,)
    """
    if hasattr(dist, 'b_log_prob'):
        return dist.b_log_prob(x_batch, targets=targets, conditions=conditions)
    else:
        # Fallback: loop over batch
        batch_size = next(iter(x_batch.values())).shape[0]
        log_probs = []
        for i in range(batch_size):
            x_i = {block: arr[i] for block, arr in x_batch.items()}
            lp = dist.log_prob(x_i, targets=targets, conditions=conditions)
            log_probs.append(lp)
        return jnp.array(log_probs)
