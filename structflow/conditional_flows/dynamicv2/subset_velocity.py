"""
Weight clipping and trajectory interpolation for subset velocity functions.

Mode A implementation: clip the actual weight matrices of ParallelStructuredFn
to remove blocks that are not needed for the query. This works because the
block-triangular structure guarantees the clipped weights are already zero.

Provides:
- ClippedParallelFn: An equinox Module holding clipped layers
- clip_tail: Slice weight matrices to remove unneeded blocks
- SubsetVelocityFn: Wraps clipped velocity fn with condition trajectory interpolation
- interp_trajectory: Linear interpolation along trajectory time grid
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Optional, Tuple, Union, Set, Callable
from structflow.pgms.structure import Structure, BlockTriangularStructure
from structflow.conditional_flows.parallel.layers import (
    DenseMaskedLinear,
    SparseMaskedLinear,
    MaskedResidualLayer,
    BlockLayerNorm,
)
from structflow.conditional_flows.parallel.structured_utils import get_block_offsets


# ============================================================================
# Trajectory Interpolation
# ============================================================================

def interp_trajectory(traj: jnp.ndarray, ts: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    Linear interpolation of a trajectory at time t.

    Assumes ts is uniformly spaced from 0 to 1 (i.e. ``jnp.linspace(0, 1, n)``).

    Args:
        traj: Trajectory array of shape (n_steps+1, dim)
        ts: Time grid array of shape (n_steps+1,). Must be uniform in [0, 1].
        t: Time value to interpolate at

    Returns:
        Interpolated value of shape (dim,)
    """
    t = jnp.clip(t, 0.0, 1.0)

    n = ts.shape[0]
    idx_float = t * (n - 1)
    idx_low = jnp.floor(idx_float).astype(jnp.int32)
    idx_high = jnp.minimum(idx_low + 1, n - 1)

    alpha = idx_float - idx_low

    val_low = traj[idx_low]
    val_high = traj[idx_high]
    return (1.0 - alpha) * val_low + alpha * val_high


# ============================================================================
# Clipped Parallel Function (equinox Module)
# ============================================================================

class ClippedParallelFn(eqx.Module):
    """
    A clipped version of ParallelStructuredFn that operates on a subset of blocks.

    Created by :func:`clip_tail`. Has the same ``__call__`` signature as
    ``ParallelStructuredFn``: ``(x_state_flat, t) -> v_flat``, but the state
    and output vectors only cover the kept blocks.
    """
    time_embed: eqx.Module
    layers: list
    structure: Structure
    block_dims: Dict[str, int]
    hidden_block_dims: Dict[str, int]
    output_dim: int
    block_offsets: Dict[str, Tuple[int, int]] = eqx.field(static=True)

    def __call__(self, x_state: jnp.ndarray, t: float) -> jnp.ndarray:
        """Forward pass: identical to ParallelStructuredFn.__call__."""
        t_arr = jnp.array([t])
        t_embed = self.time_embed(t_arr).squeeze()

        x_in = jnp.concatenate([x_state, t_embed])

        h = x_in
        for layer in self.layers:
            h = layer(h)

        h = jnp.clip(h, -10.0, 10.0)
        return h

    def get_mask(self, active_blocks: Union[List[str], Set[str]]) -> jnp.ndarray:
        """Creates a binary mask for the given active blocks."""
        mask = jnp.zeros((self.output_dim,))
        for block in active_blocks:
            start, end = self.block_offsets[block]
            mask = mask.at[start:end].set(1.0)
        return mask


# ============================================================================
# Weight Clipping
# ============================================================================

def clip_tail(velocity_fn, keep_blocks: List[str]) -> ClippedParallelFn:
    """
    Clip a ParallelStructuredFn to keep only the specified blocks.

    Removes blocks not in *keep_blocks* by slicing weight matrices.  This is
    exact (not an approximation) because the block-triangular structure
    guarantees that the removed blocks' weights are already zero for the kept
    blocks' rows.

    **Requirement:** every parent of a kept block must also be kept.  In other
    words, you can only remove blocks that have no kept descendants.

    Args:
        velocity_fn: Original ``ParallelStructuredFn``
        keep_blocks: Block names to keep (order is preserved from block_order)

    Returns:
        A :class:`ClippedParallelFn` with clipped weights

    Raises:
        ValueError: If a removed block is a parent of a kept block, or
            keep_blocks is empty / contains unknown names.
    """
    structure = velocity_fn.structure
    block_order = structure.block_order

    # ---- Validate --------------------------------------------------------
    keep_set = set(keep_blocks)
    full_set = set(block_order)

    if not keep_set:
        raise ValueError("keep_blocks cannot be empty")
    if not keep_set.issubset(full_set):
        raise ValueError(
            f"keep_blocks contains unknown blocks: {keep_set - full_set}"
        )

    remove_set = full_set - keep_set
    for b in keep_set:
        for parent in structure.dependencies.get(b, []):
            if parent in remove_set:
                raise ValueError(
                    f"Cannot remove block '{parent}': it is a parent of "
                    f"kept block '{b}'"
                )

    # ---- Compute kept indices in the full vectors ------------------------
    full_state_dim = velocity_fn.output_dim
    t_dim = velocity_fn.time_embed.output_features
    full_hidden_block_dims = velocity_fn.hidden_block_dims

    # Hidden-vector offsets (full network)
    hidden_offset = 0
    full_hidden_offsets: Dict[str, Tuple[int, int]] = {}
    for b in block_order:
        dim = full_hidden_block_dims[b]
        full_hidden_offsets[b] = (hidden_offset, hidden_offset + dim)
        hidden_offset += dim

    # Indices to keep in state vector
    kept_state_indices_list: List[int] = []
    for b in block_order:
        if b in keep_set:
            s, e = velocity_fn.block_offsets[b]
            kept_state_indices_list.extend(range(s, e))
    kept_state_idx = jnp.array(kept_state_indices_list, dtype=jnp.int32)

    # Indices to keep in hidden vector
    kept_hidden_indices_list: List[int] = []
    for b in block_order:
        if b in keep_set:
            s, e = full_hidden_offsets[b]
            kept_hidden_indices_list.extend(range(s, e))
    kept_hidden_idx = jnp.array(kept_hidden_indices_list, dtype=jnp.int32)

    # Time-embed column indices in the input vector
    time_col_idx = jnp.arange(full_state_dim, full_state_dim + t_dim,
                              dtype=jnp.int32)

    # Input-layer column indices: kept state cols + time cols
    input_col_idx = jnp.concatenate([kept_state_idx, time_col_idx])

    # ---- Derived quantities for clipped network --------------------------
    kept_block_order = [b for b in block_order if b in keep_set]
    kept_block_dims = {b: structure.dims[b] for b in kept_block_order}
    kept_hidden_block_dims = {b: full_hidden_block_dims[b]
                              for b in kept_block_order}
    kept_state_dim = int(kept_state_idx.shape[0])

    clipped_structure = BlockTriangularStructure(
        block_order=kept_block_order,
        dependencies={
            b: [p for p in structure.dependencies.get(b, []) if p in keep_set]
            for b in kept_block_order
        },
        dims=kept_block_dims,
    )

    kept_offsets = get_block_offsets(kept_block_dims, kept_block_order)

    # ---- Clip each layer -------------------------------------------------
    clipped_layers: list = []
    is_first_linear = True

    for layer in velocity_fn.layers:
        if isinstance(layer, DenseMaskedLinear):
            if is_first_linear:
                # Input projection: (hidden_dim, state_dim + t_dim)
                clipped_layers.append(
                    _clip_dense(layer, kept_hidden_idx, input_col_idx)
                )
                is_first_linear = False
            else:
                # Output projection: (state_dim, hidden_dim)
                clipped_layers.append(
                    _clip_dense(layer, kept_state_idx, kept_hidden_idx)
                )
        elif isinstance(layer, SparseMaskedLinear):
            raise NotImplementedError(
                "clip_tail does not yet support SparseMaskedLinear layers. "
                "Use use_sparse=False."
            )
        elif isinstance(layer, MaskedResidualLayer):
            clipped_layers.append(
                _clip_residual(layer, kept_hidden_idx,
                               kept_hidden_block_dims, kept_block_order)
            )
        elif callable(layer):
            # Activation function – keep as-is
            clipped_layers.append(layer)

    # ---- Assemble --------------------------------------------------------
    return ClippedParallelFn(
        time_embed=velocity_fn.time_embed,
        layers=clipped_layers,
        structure=clipped_structure,
        block_dims=kept_block_dims,
        hidden_block_dims=kept_hidden_block_dims,
        output_dim=kept_state_dim,
        block_offsets=kept_offsets,
    )


# ---- Layer clipping helpers ------------------------------------------------

def _clip_dense(
    layer: DenseMaskedLinear,
    row_idx: jnp.ndarray,
    col_idx: jnp.ndarray,
) -> DenseMaskedLinear:
    """Clip a :class:`DenseMaskedLinear` to *row_idx* × *col_idx*."""
    new_weight = layer.linear.weight[row_idx][:, col_idx]
    new_mask = layer.mask[row_idx][:, col_idx]
    has_bias = layer.linear.bias is not None
    new_bias = layer.linear.bias[row_idx] if has_bias else None

    out_features, in_features = new_weight.shape

    # Create a fresh layer with the right shape (random init)
    new_layer = DenseMaskedLinear(
        in_features, out_features, new_mask,
        key=jax.random.PRNGKey(0), use_bias=has_bias,
    )
    # Transplant the actual clipped weights via eqx.tree_at
    new_layer = eqx.tree_at(lambda l: l.linear.weight, new_layer, new_weight)
    if has_bias:
        new_layer = eqx.tree_at(lambda l: l.linear.bias, new_layer, new_bias)

    return new_layer


def _clip_residual(
    layer: MaskedResidualLayer,
    kept_hidden_idx: jnp.ndarray,
    kept_hidden_block_dims: Dict[str, int],
    kept_block_order: List[str],
) -> MaskedResidualLayer:
    """Clip a :class:`MaskedResidualLayer` to the kept hidden indices."""
    kept_hidden_dim = int(kept_hidden_idx.shape[0])

    # Slice internal linear weight / mask
    new_linear_weight = layer.linear.linear.weight[kept_hidden_idx][:, kept_hidden_idx]
    new_linear_mask = layer.linear.mask[kept_hidden_idx][:, kept_hidden_idx]
    has_bias = layer.linear.linear.bias is not None
    new_linear_bias = (layer.linear.linear.bias[kept_hidden_idx]
                       if has_bias else None)

    # Slice norm parameters
    new_norm_weight = layer.norm.weight[kept_hidden_idx]
    new_norm_bias = layer.norm.bias[kept_hidden_idx]

    # Build a fresh residual layer with the right dimensions
    new_res = MaskedResidualLayer(
        dim=kept_hidden_dim,
        mask=new_linear_mask,
        block_dims=kept_hidden_block_dims,
        block_order=kept_block_order,
        activation=layer.activation,
        key=jax.random.PRNGKey(0),
        use_sparse=False,
    )

    # Transplant clipped weights
    new_res = eqx.tree_at(
        lambda r: r.linear.linear.weight, new_res, new_linear_weight)
    if has_bias:
        new_res = eqx.tree_at(
            lambda r: r.linear.linear.bias, new_res, new_linear_bias)
    new_res = eqx.tree_at(lambda r: r.norm.weight, new_res, new_norm_weight)
    new_res = eqx.tree_at(lambda r: r.norm.bias, new_res, new_norm_bias)

    return new_res


# ============================================================================
# Subset Velocity Function
# ============================================================================

class SubsetVelocityFn(eqx.Module):
    """
    Velocity function for a subset of variables with condition trajectory
    interpolation.

    Wraps a (possibly clipped) ``ParallelStructuredFn`` /
    :class:`ClippedParallelFn`.  At each call:

    1. Interpolates condition values at time *t* from stored trajectories
    2. Builds full state for the (clipped) network: ``[cond_vals | active_vals]``
    3. Calls the network
    4. Extracts only active-block velocities from the output

    Has signature ``(x_active_flat, t) -> v_active_flat``, compatible with
    :class:`~flow.flow.Flow` / integrators.
    """
    clipped_vfn: eqx.Module                    # ClippedParallelFn or ParallelStructuredFn
    condition_trajs: Dict[str, jnp.ndarray]    # forward trajs (n_steps+1, dim)
    traj_ts: jnp.ndarray                       # time grid for interpolation
    cond_block_names: List[str]                 # condition blocks in order
    active_block_names: List[str]              # active blocks in order
    cond_dim: int                              # total condition dimensions
    active_dim: int                            # total active dimensions

    def __init__(
        self,
        clipped_vfn: eqx.Module,
        condition_trajs: Dict[str, jnp.ndarray],
        traj_ts: jnp.ndarray,
        cond_block_names: List[str],
        active_block_names: List[str],
    ):
        """
        Args:
            clipped_vfn: Velocity function for all kept blocks (cond + active).
                Must expose ``.block_dims`` and ``.block_offsets``.
            condition_trajs: Maps condition block names to forward trajectories
                of shape ``(n_steps+1, block_dim)``.
            traj_ts: Time grid for trajectories, shape ``(n_steps+1,)``.
            cond_block_names: Condition block names (prefix of clipped_vfn
                blocks).
            active_block_names: Active (target) block names (suffix of
                clipped_vfn blocks).
        """
        self.clipped_vfn = clipped_vfn
        self.condition_trajs = condition_trajs
        self.traj_ts = traj_ts
        self.cond_block_names = list(cond_block_names)
        self.active_block_names = list(active_block_names)

        self.cond_dim = sum(
            clipped_vfn.block_dims.get(b, 0) for b in cond_block_names
        )
        self.active_dim = sum(
            clipped_vfn.block_dims.get(b, 0) for b in active_block_names
        )

    def __call__(self, x_active: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Compute velocity for active blocks.

        Args:
            x_active: State of active blocks only, shape ``(active_dim,)``
            t: Scalar time

        Returns:
            Velocity for active blocks, shape ``(active_dim,)``
        """
        # 1. Interpolate conditions at time t
        cond_vals = {}
        for block in self.cond_block_names:
            traj = self.condition_trajs[block]
            cond_vals[block] = interp_trajectory(traj, self.traj_ts, t)

        # 2. Slice active block values out of x_active (which is ordered by
        #    active_block_names).  We need per-block arrays to reconstruct
        #    x_full in the clipped network's block_order.
        active_vals = {}
        active_offset = 0
        for block in self.active_block_names:
            dim = self.clipped_vfn.block_dims[block]
            active_vals[block] = x_active[active_offset : active_offset + dim]
            active_offset += dim

        # 3. Build full state in the clipped network's block_order so that
        #    weight slices align correctly.  This avoids the wrong [cond|active]
        #    ordering when cond and active blocks are interleaved in block_order.
        full_parts = []
        for block in self.clipped_vfn.structure.block_order:
            if block in cond_vals:
                full_parts.append(cond_vals[block])
            else:
                full_parts.append(active_vals[block])

        x_full = jnp.concatenate(full_parts) if full_parts else x_active

        # 3. Call (clipped) velocity function
        v_full = self.clipped_vfn(x_full, t)

        # 4. Extract active velocities using block_offsets from the clipped_vfn
        active_parts = []
        for block in self.active_block_names:
            start, end = self.clipped_vfn.block_offsets[block]
            active_parts.append(v_full[start:end])

        return jnp.concatenate(active_parts) if active_parts else jnp.array([])
