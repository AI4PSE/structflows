"""
Utilities for numerical tests using exact Gaussian velocity functions.

Provides adapters that wrap exact velocity functions from sinterp.exact_si_gaussian
to work with different distribution modes (static, dynamic, guided).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Callable, Optional, List
from structflow.pgms.structure import BlockTriangularStructure
from sinterp.stochasticfield import OneSidedField_eta1
from sinterp.interpolants import LinearDeterministic

__all__ = [
    'ExactGaussianStaticVelocity',
    'ExactGaussianParallelVelocity',
    'ExactGaussianGuidedVelocity',
    'ExactBlockVelocity',
    'create_exact_velocity_functions',
    'analytical_gaussian_log_prob',
    'create_guided_field_from_velocities',
]


# ============================================================================
# Block Velocity Wrapper (for static mode)
# ============================================================================

class ExactBlockVelocity(eqx.Module):
    """
    Wraps a scalar velocity function to match the signature expected by
    StaticConditionalDistribution: (t, x_block, *parents) -> velocity_block
    
    The underlying velocity function has signature: (x, t) -> velocity
    """
    v_fn: Callable
    
    def __call__(self, t, x_block, *parents):
        """
        Args:
            t: Time step (scalar or array)
            x_block: Block value at time t
            *parents: Parent block values (unused for independent case)
        
        Returns:
            Velocity for this block
        """
        return self.v_fn(x_block, t)


# ============================================================================
# Static Mode Adapter
# ============================================================================

class ExactGaussianStaticVelocity(eqx.Module):
    """
    Adapter wrapping exact Gaussian velocity functions for static mode.
    
    Provides the interface expected by StaticConditionalDistribution:
    - .structure property
    - .blocks property (dict of velocity functions per block)
    - __call__(x_dict, t, active_blocks=None) method
    """
    structure: BlockTriangularStructure
    v_fns: Dict[str, Callable]
    
    def __init__(self, structure: BlockTriangularStructure, 
                 v_fns: Dict[str, Callable]):
        """
        Args:
            structure: BlockTriangularStructure defining block ordering
            v_fns: Dict mapping block names to velocity functions (x, t) -> v
        """
        self.structure = structure
        self.v_fns = v_fns
    
    @property
    def blocks(self):
        """Return velocity functions for each block with correct signature."""
        return {
            name: ExactBlockVelocity(self.v_fns[name])
            for name in self.structure.block_order
        }
    
    def __call__(self, x_dict: Dict, t, active_blocks=None):
        """
        Compute velocity for all or active blocks.
        
        Args:
            x_dict: Dictionary mapping block names to values
            t: Time step
            active_blocks: Optional list of blocks to compute velocity for
        
        Returns:
            Dictionary of velocities for each block
        """
        blocks_to_compute = active_blocks or self.structure.block_order
        result = {}
        
        for name in blocks_to_compute:
            if name in self.v_fns:
                x_block = x_dict[name]
                result[name] = self.v_fns[name](x_block, t)
        
        return result


# ============================================================================
# Parallel Mode Adapter
# ============================================================================

class ExactGaussianParallelVelocity(eqx.Module):
    """
    Adapter wrapping exact Gaussian velocity functions for parallel/dynamic mode.
    
    Provides the interface expected by ParallelBlockTriangularDistribution:
    - .structure property
    - .block_offsets property
    - __call__(x_vec, t) method that takes flat vectors
    """
    structure: BlockTriangularStructure
    v_fns: Dict[str, Callable]
    block_offsets: Dict[str, tuple]  # Maps block name to (start, end) indices
    
    def __init__(self, structure: BlockTriangularStructure, 
                 v_fns: Dict[str, Callable]):
        """
        Args:
            structure: BlockTriangularStructure defining block ordering
            v_fns: Dict mapping block names to velocity functions (x, t) -> v
        """
        self.structure = structure
        self.v_fns = v_fns
        
        # Compute block offsets
        offsets = {}
        idx = 0
        for name in structure.block_order:
            dim = structure.dims[name]
            offsets[name] = (idx, idx + dim)
            idx += dim
        self.block_offsets = offsets
    
    def __call__(self, x_vec: jnp.ndarray, t):
        """
        Compute velocity for flat vector representation.
        
        Args:
            x_vec: Flat vector with blocks concatenated in order
            t: Time step
        
        Returns:
            Flat velocity vector
        """
        v_parts = []
        
        for name in self.structure.block_order:
            start, end = self.block_offsets[name]
            x_block = x_vec[start:end]
            
            if name in self.v_fns:
                v_block = self.v_fns[name](x_block, t)
            else:
                v_block = jnp.zeros_like(x_block)
            
            v_parts.append(v_block)
        
        return jnp.concatenate(v_parts)


# ============================================================================
# Guided Mode Adapter
# ============================================================================

class ExactGaussianGuidedVelocity(eqx.Module):
    """
    Adapter wrapping exact Gaussian velocity functions for guided mode.
    
    GuidanceBasedDistribution expects a StochasticField-like object with
    a get_velocity_b(x, t) method.
    """
    v_fns: Dict[str, Callable]
    key_order: List[str]
    block_dims: Dict[str, int]  # For flat vector slicing
    
    def __init__(self, v_fns: Dict[str, Callable], key_order: List[str],
                 block_dims: Optional[Dict[str, int]] = None):
        """
        Args:
            v_fns: Dict mapping variable names to velocity functions (x, t) -> v
            key_order: Ordering of keys for flattening/unflattening
            block_dims: Dict mapping names to dimensions for flat vector handling
        """
        self.v_fns = v_fns
        self.key_order = key_order
        self.block_dims = block_dims or {name: 1 for name in key_order}
    
    def get_velocity_b(self, x, t):
        """
        Compute velocity for dict or flat vector representation.
        
        Args:
            x: Either Dict[str, array] or flat vector
            t: Time step
        
        Returns:
            Same structure as input (dict or flat vector)
        """
        if isinstance(x, dict):
            result = {}
            for name in self.key_order:
                if name in self.v_fns and name in x:
                    x_val = x[name]
                    result[name] = self.v_fns[name](x_val, t)
            return result
        else:
            # Handle flat vector: split, compute, concatenate
            v_parts = []
            idx = 0
            for name in self.key_order:
                dim = self.block_dims[name]
                x_block = x[idx:idx+dim]
                if name in self.v_fns:
                    v_block = self.v_fns[name](x_block, t)
                else:
                    v_block = jnp.zeros_like(x_block)
                v_parts.append(v_block)
                idx += dim
            return jnp.concatenate(v_parts)


# ============================================================================
# Helper Functions
# ============================================================================

def create_exact_velocity_functions(
    structure: BlockTriangularStructure,
    target_params: Dict[str, Dict[str, jnp.ndarray]]
) -> Dict[str, Callable]:
    """
    Create exact Gaussian velocity functions for each block.
    
    Args:
        structure: BlockTriangularStructure
        target_params: Dict mapping block names to {'mean': m2, 'var': c2}
                      The base distribution is assumed to be N(0, I)
    
    Returns:
        Dict mapping block names to velocity functions v(x, t)
    """
    from sinterp.exact_si_gaussian import construct_p_b_s
    from sinterp.interpolants import LinearDeterministic
    
    interp = LinearDeterministic()
    v_fns = {}
    
    for block in structure.block_order:
        if block not in target_params:
            raise ValueError(f"Missing target parameters for block {block}")
        
        params = target_params[block]
        m2 = params['mean']
        var = params['var']
        
        # Base distribution: N(0, I)
        dim = structure.dims[block]
        m1 = jnp.zeros((dim,))
        c1 = jnp.eye(dim)
        
        # Target distribution: N(m2, diag(var))
        c2 = jnp.diag(var)
        
        # Get exact velocity function
        _, v, _ = construct_p_b_s(m1, m2, c1, c2, interp)
        v_fns[block] = v
    
    return v_fns


def analytical_gaussian_log_prob(x: jnp.ndarray,
                                  mean: jnp.ndarray,
                                  var: jnp.ndarray) -> jnp.ndarray:
    """
    Compute analytical log probability of x under N(mean, diag(var)).
    
    Args:
        x: Value (can be scalar or array)
        mean: Mean of Gaussian
        var: Variance (diagonal only)
    
    Returns:
        Log probability
    """
    x = jnp.atleast_1d(x)
    mean = jnp.atleast_1d(mean)
    var = jnp.atleast_1d(var)
    
    diff = x - mean
    log_det = jnp.sum(jnp.log(var))
    quad_form = jnp.sum(diff**2 / var)
    
    return -0.5 * (quad_form + log_det + x.shape[0] * jnp.log(2 * jnp.pi))


def create_guided_field_from_velocities(
    v_fns: Dict[str, Callable],
    key_order: List[str],
    dims: Optional[Dict[str, int]] = None
) -> OneSidedField_eta1:
    """
    Wrap exact velocity functions in a flat-vector adapter and then
    in OneSidedField_eta1 for guided mode.
    
    Args:
        v_fns: Dict mapping variable names to velocity functions (x, t) -> v
        key_order: Ordering of variables
        dims: Dict mapping variable names to dimensions (default: all 1D)
    
    Returns:
        OneSidedField_eta1 instance that wraps the velocities
    """
    if dims is None:
        dims = {name: 1 for name in key_order}
    
    # Compute block offsets for flat vector slicing
    offsets = {}
    idx = 0
    for name in key_order:
        dim = dims[name]
        offsets[name] = (idx, idx + dim)
        idx += dim
    
    # Create a flat-vector-based velocity wrapper
    class FlatVelocityNet(eqx.Module):
        v_fns: Dict[str, Callable]
        key_order: List[str]
        offsets: Dict[str, tuple]
        
        def __call__(self, x_flat, t):
            """Accept flat vector, return flat vector."""
            v_parts = []
            for name in self.key_order:
                start, end = self.offsets[name]
                x_block = x_flat[start:end]
                if name in self.v_fns:
                    v_block = self.v_fns[name](x_block, t)
                else:
                    v_block = jnp.zeros_like(x_block)
                v_parts.append(v_block)
            return jnp.concatenate(v_parts)
    
    net = FlatVelocityNet(v_fns, key_order, offsets)
    interpolator = LinearDeterministic()
    
    return OneSidedField_eta1(interpolator, net)
