import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Callable, Optional, Union, Tuple, Set
from structflow.pgms.structure import Structure
from structflow.conditional_flows.parallel.layers import (
    DenseMaskedLinear, 
    SparseMaskedLinear, 
    MaskedResidualLayer
)
from structflow.conditional_flows.parallel.structured_utils import (
    build_structure_mask, 
    get_block_offsets
)
from flow.net import FourierFeatures


def clip_by_norm(x, max_norm, axis=-1):
    """Clips the norm of a tensor x to max_norm."""
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    # Avoid division by zero
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return x * factor

class ParallelStructuredFn(eqx.Module):
    """
    A vectorized function f(x, t) that respects block-triangular dependencies.
    """
    time_embed: FourierFeatures
    layers: List[eqx.Module]
    structure: Structure
    block_dims: Dict[str, int]
    hidden_block_dims: Dict[str, int]
    output_dim: int
    block_offsets: Dict[str, Tuple[int, int]] = eqx.field(static=True)
    
    def __init__(
        self,
        structure: Structure,
        hidden_dim: int,
        depth: int,
        key: jax.random.PRNGKey,
        activation: Callable = jax.nn.silu,
        time_embed_dim: int = 16,
        use_sparse: bool = False,
        output_dim: Optional[int] = None
    ):
        self.structure = structure
        self.block_dims = structure.dims
        self.block_offsets = get_block_offsets(self.block_dims, structure.block_order)
        
        # Calculate Input Dims
        state_dim = sum(structure.dims[b] for b in structure.block_order)
        input_dim = state_dim + time_embed_dim
        if output_dim is not None:
            self.output_dim = output_dim
        else:
            self.output_dim = state_dim
        
        # Keys
        keys = jax.random.split(key, depth + 2)
        t_key, in_key, out_key = keys[0], keys[1], keys[-1]
        res_keys = keys[2:-1]
        
        # 1. Component: Time Embedder
        self.time_embed = FourierFeatures(
            output_features=time_embed_dim,
            key=t_key
        )
        
        # 2. Hidden Unit Assignment
        # Divide hidden units among blocks
        n_blocks = len(structure.block_order)
        base_count = hidden_dim // n_blocks
        remainder = hidden_dim % n_blocks
        
        self.hidden_block_dims = {}
        for i, block in enumerate(structure.block_order):
            self.hidden_block_dims[block] = base_count + (1 if i < remainder else 0)
        hidden_block_dims = self.hidden_block_dims
            
        self.layers = []
        
        # 3. Input Projection (Mixing)
        # Input -> Hidden
        # Hidden(b) depends on Input(parents(b) U {b}) AND Input(_TIME)
        in_block_dims = dict(structure.dims)
        in_block_dims['_TIME'] = time_embed_dim 
        
        in_block_order = structure.block_order + ['_TIME']
        extra_deps = {b: ['_TIME'] for b in structure.block_order}
        
        mask_in = build_structure_mask(
            structure, 
            in_block_dims, 
            hidden_block_dims,
            in_block_order=in_block_order,
            extra_dependencies=extra_deps
        )
        
        linear_cls = SparseMaskedLinear if use_sparse else DenseMaskedLinear
        self.layers.append(linear_cls(input_dim, hidden_dim, mask_in, key=in_key))
        self.layers.append(activation)
        
        # 4. Residual Stack
        # Hidden -> Hidden (MaskedResidualLayer)
        mask_h = build_structure_mask(structure, hidden_block_dims, hidden_block_dims)
        
        for i in range(depth - 2):
            self.layers.append(
                MaskedResidualLayer(
                    dim=hidden_dim, 
                    mask=mask_h, 
                    block_dims=hidden_block_dims,
                    block_order=structure.block_order,
                    activation=activation,
                    key=res_keys[i],
                    use_sparse=use_sparse
                )
            )
            
        # 5. Output Projection
        # Hidden -> Output
        # Direct Map Mask: Output(b) depends only on Hidden(b) (Block Diagonal)
        # We can use build_structure_mask with an empty dependency map to get block-diagonal.
        class EmptyStruct:
            block_order = structure.block_order
            dependencies = {b: [] for b in block_order}
            
        mask_out = build_structure_mask(
            EmptyStruct, 
            hidden_block_dims, 
            structure.dims,
            include_self=True
        )
                    
        self.layers.append(linear_cls(hidden_dim, state_dim, mask_out, key=out_key))

    def get_mask(self, active_blocks: Union[List[str], Set[str]]) -> jnp.ndarray:
        """
        Creates a binary mask for the given active blocks.
        """
        mask = jnp.zeros((self.output_dim,))
        for block in active_blocks:
            start, end = self.block_offsets[block]
            mask = mask.at[start:end].set(1.0)
        return mask

    def __call__(self, x_state: jnp.ndarray, t: float):
        """
        Args:
            x_state: Flattened state vector of all blocks.
            t: Scalar time.
        """
        t_arr = jnp.array([t])
        t_embed = self.time_embed(t_arr).squeeze() # FourierFeatures returns (dim,)
        
        x_in = jnp.concatenate([x_state, t_embed])
        
        h = x_in
        for layer in self.layers:
            h = layer(h)
            
        h = jnp.clip(h, -10.0, 10.0)
        return h