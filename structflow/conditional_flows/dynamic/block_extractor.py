"""
Utilities to extract block-based velocity functions from ParallelVelocityFn.

This allows clean separation of concerns:
- ParallelVelocityFn: trained on full states with masked matrices
- BlockBasedVelocityFn: extracts per-block velocities for sampling/likelihood

The extracted weights are shared (no copying), so training dynamics are preserved.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Callable, Optional


class BlockBasedVelocityFn(eqx.Module):
    """
    Provides per-block velocity functions extracted from ParallelVelocityFn.
    
    Usage:
        block_net = BlockBasedVelocityFn.from_parallel(parallel_vfn)
        v_target = block_net.get_block_velocity(target_block)(x_target, t, parents_dict)
    """
    
    parallel_vfn: 'ParallelVelocityFn'  # Reference to original network (shared weights)
    block_offsets: Dict[str, tuple]     # (start, end) indices for each block
    structure: 'Structure'               # Block structure
    block_order: List[str]               # Order of blocks
    
    def __init__(self, parallel_vfn):
        # Use duck typing instead of isinstance check to handle wrapped/deserialized modules
        required_attrs = ['block_offsets', 'structure']
        for attr in required_attrs:
            if not hasattr(parallel_vfn, attr):
                raise TypeError(f"parallel_vfn must have attributes {required_attrs}, missing '{attr}'")
        
        self.parallel_vfn = parallel_vfn
        self.block_offsets = parallel_vfn.block_offsets
        self.structure = parallel_vfn.structure
        self.block_order = parallel_vfn.structure.block_order
    
    @classmethod
    def from_parallel(cls, parallel_vfn):
        """Create a block-based extractor from a parallel velocity function."""
        return cls(parallel_vfn)
    
    def get_block_velocity(self, block_name: str) -> Callable:
        """
        Returns a velocity function for a specific block.
        
        The returned function has signature:
            v_block = f(x_block, t, x_full_context)
        
        where:
            x_block: value of the block at time t
            t: time
            x_full_context: dict with all block values for dependency resolution
        """
        start, end = self.block_offsets[block_name]
        
        def block_velocity_fn(x_block: jnp.ndarray, t: float, x_full_context: Dict[str, jnp.ndarray]):
            """
            Compute velocity for a specific block given full context.
            
            Args:
                x_block: Block value (shape depends on block dimension)
                t: Time
                x_full_context: Dict with all block values for dependency resolution
            
            Returns:
                Velocity for this block
            """
            # IMPORTANT: Use x_block for this block's value in the full state.
            # This ensures that when computing the Jacobian (divergence) via JVP,
            # the function output actually depends on x_block, not just the
            # captured context values.
            context_with_current = {**x_full_context, block_name: x_block}
            x_full = self._reconstruct_full_state(context_with_current)
            
            # Get full velocity from parallel network
            v_full = self.parallel_vfn(x_full, t)
            
            # Extract just this block's velocity
            return v_full[start:end]
        
        return block_velocity_fn
    
    def _reconstruct_full_state(self, x_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Reconstruct full flat state vector from block dict.
        
        Missing blocks are filled with zeros. This is safe for block-triangular
        structures because the velocity function's masked matrices ensure that
        missing blocks don't affect the computation.
        """
        flat_list = []
        for block in self.block_order:
            if block in x_dict:
                flat_list.append(x_dict[block].ravel())
            else:
                # Fill missing blocks with zeros
                dim = self.structure.dims[block]
                flat_list.append(jnp.zeros((dim,)))
        return jnp.concatenate(flat_list)
    
    def get_multiblock_velocity(self, block_names: List[str]) -> Callable:
        """
        Returns a velocity function for multiple blocks simultaneously.
        
        The returned function has signature:
            v_blocks = f(x_blocks_dict, t, x_full_context)
        
        Returns velocities for specified blocks in a dict.
        """
        starts = [self.block_offsets[b][0] for b in block_names]
        ends = [self.block_offsets[b][1] for b in block_names]
        
        def multiblock_velocity_fn(
            x_blocks_dict: Dict[str, jnp.ndarray],
            t: float, 
            x_full_context: Dict[str, jnp.ndarray]
        ) -> Dict[str, jnp.ndarray]:
            """
            Compute velocities for multiple blocks given full context.
            
            Args:
                x_blocks_dict: Dict with values for active blocks
                t: Time
                x_full_context: Dict with all block values for dependency resolution
            
            Returns:
                Dict with velocities for active blocks
            """
            # Reconstruct full flat vector from context
            # BUG? what happens if both x_full and x_blocks contain values for the same key?
            # which one overwrites the other?
            x_full = self._reconstruct_full_state({**x_full_context, **x_blocks_dict})
            
            # Get full velocity from parallel network
            v_full = self.parallel_vfn(x_full, t)
            
            # Extract velocities for requested blocks
            result = {}
            for block_name, start, end in zip(block_names, starts, ends):
                result[block_name] = v_full[start:end]
            
            return result
        
        return multiblock_velocity_fn

def clip_by_norm(x, max_norm, axis=-1):
    """Clips the norm of a tensor x to max_norm."""
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    # Avoid division by zero
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return x * factor