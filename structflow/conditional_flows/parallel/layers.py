import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
from structflow.pgms.structure import Structure
from structflow.conditional_flows.parallel.structured_utils import build_structure_mask

# Re-export from structured_utils for backward compatibility if needed, 
# but we prefer using build_structure_mask now.
def create_dependency_mask(
    structure: Structure,
    in_block_dims: Dict[str, int],
    out_block_dims: Dict[str, int],
    **kwargs
) -> jnp.ndarray:
    """Legacy wrapper for build_structure_mask."""
    return build_structure_mask(structure, in_block_dims, out_block_dims, **kwargs)

class DenseMaskedLinear(eqx.Module):
    """
    Linear layer with a binary mask applied to weights.
    Uses dense matrix multiplication.
    """
    linear: eqx.nn.Linear
    mask: jax.Array
    
    def __init__(self, in_features: int, out_features: int, mask: jax.Array, key: jax.random.PRNGKey, use_bias: bool = True):
        self.linear = eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=key)
        
        if mask.shape != (out_features, in_features):
             raise ValueError(f"Mask shape {mask.shape} mismatch with layer shape {(out_features, in_features)}")
             
        self.mask = mask
        
    def __call__(self, x: jax.Array) -> jax.Array:
        # The mask is an array leaf, so eqx.filter(model, eqx.is_array) hands it to the
        # optimiser; stop_gradient keeps it exactly 0/1 (an optimiser with weight decay
        # would still shrink it).
        w_masked = self.linear.weight * jax.lax.stop_gradient(self.mask)
        out = w_masked @ x
        
        if self.linear.bias is not None:
            out = out + self.linear.bias
            
        return out

class SparseMaskedLinear(eqx.Module):
    """
    Linear layer with a binary mask applied to weights.
    Uses JAX BCOO sparse matrix multiplication.
    """
    weight_values: jax.Array
    indices: jax.Array
    bias: Optional[jax.Array]
    shape: Tuple[int, int] = eqx.field(static=True)
    
    def __init__(self, in_features: int, out_features: int, mask: jax.Array, key: jax.random.PRNGKey, use_bias: bool = True):
        self.shape = (out_features, in_features)
        
        # Extract sparse indices
        indices = jnp.stack(jnp.where(mask > 0), axis=-1)
        self.indices = indices
        num_nonzero = indices.shape[0]
        
        w_key, b_key = jax.random.split(key)
        # Use Kaiming uniform init or similar
        limit = jnp.sqrt(6 / (in_features + out_features))
        self.weight_values = jax.random.uniform(w_key, (num_nonzero,), minval=-limit, maxval=limit)
        
        if use_bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None
            
    def __call__(self, x: jax.Array) -> jax.Array:
        from jax.experimental import sparse
        # Construct BCOO on the fly. Since indices are static, this is efficient in JIT.
        w_sparse = sparse.BCOO((self.weight_values, self.indices), shape=self.shape)
        out = w_sparse @ x
        
        if self.bias is not None:
            out = out + self.bias
            
        return out

# Alias for backward compatibility, defaults to Dense
MaskedLinear = DenseMaskedLinear

class BlockLayerNorm(eqx.Module):
    """
    Applies LayerNorm independently to each block's section of the state vector.
    Preserves autoregressive structure by preventing mixing across blocks during normalization.
    Vectorized implementation for efficiency.
    """
    weight: jnp.ndarray
    bias: jnp.ndarray
    segment_ids: Tuple[int, ...] = eqx.field(static=True)
    block_sizes: Tuple[int, ...] = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    
    def __init__(self, block_dims: Dict[str, int], block_order: List[str], eps: float = 1e-5):
        """
        Args:
            block_dims: Mapping from block name to dimension.
            block_order: Order of blocks in the vector.
            eps: Epsilon for numerical stability.
        """
        total_dim = sum(block_dims[b] for b in block_order)
        self.weight = jnp.ones((total_dim,))
        self.bias = jnp.zeros((total_dim,))
        self.eps = eps
        
        segment_ids_list = []
        block_sizes_list = []
        for i, block in enumerate(block_order):
            dim = block_dims[block]
            segment_ids_list.extend([i] * dim)
            block_sizes_list.append(dim)
            
        self.segment_ids = tuple(segment_ids_list)
        self.block_sizes = tuple(block_sizes_list)
        self.num_segments = len(block_order)
            
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Compute block-wise means
        # means[i] = mean of block i
        # Convert tuples to arrays for JAX operations
        segment_ids = jnp.array(self.segment_ids)
        block_sizes = jnp.array(self.block_sizes)

        # 1. Compute block-wise means
        # means[i] = mean of block i
        sum_x = jax.ops.segment_sum(x, segment_ids, num_segments=self.num_segments)
        means = sum_x / block_sizes
        
        # 2. Compute block-wise variance
        # Var[X] = E[X^2] - E[X]^2
        sum_x2 = jax.ops.segment_sum(jnp.square(x), segment_ids, num_segments=self.num_segments)
        means2 = sum_x2 / block_sizes
        vars = jnp.maximum(0.0, means2 - jnp.square(means)) # Ensure non-negative
        
        # 3. Spread means and vars back to the original shape
        x_mean = means[segment_ids]
        x_var = vars[segment_ids]
        
        # 4. Normalize
        x_hat = (x - x_mean) / jnp.sqrt(x_var + self.eps)
        
        return x_hat * self.weight + self.bias

class MaskedResidualLayer(eqx.Module):
    """
    Residual layer with masking and block-wise LayerNorm.
    Structure: x + activation(linear(norm(x)))
    """
    linear: Union[DenseMaskedLinear, SparseMaskedLinear]
    norm: BlockLayerNorm
    activation: Callable = eqx.field(static=True)
    
    def __init__(self, 
                 dim: int, 
                 mask: jax.Array, 
                 block_dims: Dict[str, int], 
                 block_order: List[str],
                 activation: Callable = jax.nn.silu,
                 key: jax.random.PRNGKey = None,
                 use_sparse: bool = False):
        
        self.norm = BlockLayerNorm(block_dims, block_order)
        
        if use_sparse:
            self.linear = SparseMaskedLinear(dim, dim, mask, key=key)
        else:
            self.linear = DenseMaskedLinear(dim, dim, mask, key=key)
            
        self.activation = activation
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.norm(x)
        h = self.linear(h)
        h = self.activation(h)
        return x + h
