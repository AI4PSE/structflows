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

class ParallelStructuredFn_wEta1(eqx.Module):
    """
    Function f(x_t, eta1, t) for Eta1 Prediction mode.
    
    Inputs are concatenated: [x_t, eta1, t_embed].
    
    Structure:
    - x_t part: Enforces Identity dependency (v^i depends on x_t^i).
    - eta1 part: Enforces Parent dependency (v^i depends on eta1^{parents(i)}).
    """
    time_embed: FourierFeatures
    layers: List[eqx.Module]
    structure: Structure
    block_dims: Dict[str, int]
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
        use_sparse: bool = False
    ):
        self.structure = structure
        self.block_dims = structure.dims
        self.block_offsets = get_block_offsets(self.block_dims, structure.block_order)
        
        # Calculate Dimensions
        state_dim = sum(structure.dims[b] for b in structure.block_order)
        # Input is x_t + eta1 + time
        input_dim = 2 * state_dim + time_embed_dim
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
        n_blocks = len(structure.block_order)
        base_count = hidden_dim // n_blocks
        remainder = hidden_dim % n_blocks
        
        hidden_block_dims = {}
        for i, block in enumerate(structure.block_order):
            hidden_block_dims[block] = base_count + (1 if i < remainder else 0)
            
        self.layers = []
        
        # 3. Input Projection (Mixing)
        # Input -> Hidden
        # Need special mask for [x_t, eta1, time]
        in_block_dims = dict(structure.dims)
        
        mask_in = self._build_split_input_mask(
            structure,
            in_block_dims,
            hidden_block_dims,
            time_embed_dim
        )
        
        linear_cls = SparseMaskedLinear if use_sparse else DenseMaskedLinear
        self.layers.append(linear_cls(input_dim, hidden_dim, mask_in, key=in_key))
        self.layers.append(activation)
        
        # 4. Residual Stack
        # Hidden -> Hidden (Standard triangular mask)
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
        # Hidden -> Output (Block Diagonal)
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

    def _build_split_input_mask(self, structure, in_block_dims, hidden_block_dims, time_dim):
        """
        Builds mask for concatenated input [x_t, eta1, time].
        x_t: Diagonal dependency (Identity)
        eta1: Parent dependency
        time: All
        """
        # 1. Identity Mask for x_t
        # Use dummy structure with no parents, but include_self=True
        class IdentityStruct:
            block_order = structure.block_order
            dependencies = {b: [] for b in structure.block_order}
            
        mask_xt = build_structure_mask(
            IdentityStruct,
            in_block_dims,
            hidden_block_dims,
            include_self=True
        )
        
        # 2. Parent Mask for eta1
        # Use real structure, but include_self=False (parents only)
        mask_eta = build_structure_mask(
            structure,
            in_block_dims,
            hidden_block_dims,
            include_self=False
        )
        
        # 3. Time Mask
        n_hidden = sum(hidden_block_dims.values())
        mask_time = jnp.ones((n_hidden, time_dim))
        
        return jnp.concatenate([mask_xt, mask_eta, mask_time], axis=1)

    def __call__(self, x_concat: jnp.ndarray, t: float):
        """
        Args:
            x_concat: Concatenated vector [x_t, eta1]
            t: Scalar time
        """
        t_arr = jnp.array([t])
        t_embed = self.time_embed(t_arr).squeeze()
        
        x_in = jnp.concatenate([x_concat, t_embed])
        
        h = x_in
        for layer in self.layers:
            h = layer(h)
            
        h = jnp.clip(h, -10.0, 10.0)
        return h
