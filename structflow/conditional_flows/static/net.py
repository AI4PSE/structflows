import inspect
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Dict, Callable, Any, Tuple, Optional
from jax import random
from flow.net import FourierFeatures
from structflow.pgms.structure import Structure


def _call_block_constructor(block_constructor, block_name, **kwargs):
    """Call a block constructor, passing name= only if it accepts it.

    Keeps backward compatibility with constructors of signature
    (input_dim, context_dims, key) while letting shape-aware factories
    dispatch on the block's name (e.g. conv blocks for image variables).
    """
    try:
        params = inspect.signature(block_constructor).parameters
        accepts_name = 'name' in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        accepts_name = False
    if accepts_name:
        kwargs['name'] = block_name
    return block_constructor(**kwargs)

class Block(eqx.Module):
    input_dim: int = eqx.field(static=True)
    context_dims: List[int] = eqx.field(static=True)

    def __init__(self, input_dim: int, 
                 context_dims: List[int], 
                 *args, **kwargs):
        # Use object.__setattr__ to bypass frozen dataclass check
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "context_dims", context_dims)

        # Validate inputs
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(context_dims, list):
            raise ValueError(f"context_dims must be a list, got {type(context_dims)}")
        for i, dim in enumerate(context_dims):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"context_dims[{i}] must be a positive integer, got {dim}")

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, *parents: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def validate_input(self, t: jnp.ndarray, x: jnp.ndarray, *parents: jnp.ndarray):
        # Validate shapes
        if t.ndim != 1:
            raise ValueError(f"Time t must be 1-dimensional, got shape {t.shape}")
        if x.ndim != 1:
            raise ValueError(f"State x must be 1-dimensional, got shape {x.shape}")
        
        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"State x has wrong dimension: expected {self.input_dim}, got {x.shape[0]}"
            )
        
        expected_n_parents = len(self.context_dims)
        if len(parents) != expected_n_parents:
            raise ValueError(
                f"Expected {expected_n_parents} parent(s), got {len(parents)}"
            )
        
        for i, parent in enumerate(parents):
            if parent.ndim != 1:
                raise ValueError(f"Parent {i} must be 1-dimensional, got shape {parent.shape}")
            expected_dim = self.context_dims[i]
            if parent.shape[0] != expected_dim:
                raise ValueError(
                    f"Parent {i} has wrong dimension: expected {expected_dim}, got {parent.shape[0]}"
                )

class ZeroBlock(Block):
    """
    A block that always outputs zero.
    Should allow us to build identity flows.

    Args:
        input_dim: Dimensionality of the state variable.
        context_dims: List of parent variable dimensions (in order).
    """
    def __init__(self, input_dim: int, context_dims: List[int], *args, **kwargs):
        super().__init__(input_dim, context_dims)

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, *parents: jnp.ndarray) -> jnp.ndarray:
        self.validate_input(t, x, *parents)
        return jnp.zeros_like(x)  # + 1e-6

class ResidualLayer(eqx.Module):
    """
    A layer that does: x + activation(linear(norm(x)))
    """
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    activation: Callable = eqx.field(static=True)

    def __init__(self, key, dim, activation):
        self.linear = eqx.nn.Linear(dim, dim, key=key)
        self.norm = eqx.nn.LayerNorm(dim)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.norm(x)
        h = self.linear(h)
        h = self.activation(h)
        return x + h

class ConditionalBlock(Block):
    """
    Neural network block for computing conditional velocities in a flow.
    Architecture: Linear -> Activation -> (depth-2) Residual Layers -> Linear
    """
    time_embed: FourierFeatures
    input_projection: eqx.nn.Linear
    residual_layers: List[ResidualLayer]
    output_projection: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(self, 
                 input_dim: int, 
                 context_dims: List[int], 
                 key: jax.Array,
                 hidden_dim: int = 128,
                 depth: int = 8,
                 time_embed_dim: int = 16,
                 activation: Callable = jax.nn.silu,
                 **kwargs):
        # Allow 'width' as an alias for hidden_dim for backward compatibility with some factories
        if 'width' in kwargs:
            hidden_dim = kwargs.pop('width')
            
        super().__init__(input_dim, context_dims)

        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(depth, int) or depth <= 0:
            raise ValueError(f"depth must be a positive integer, got {depth}")
        
        keys = jax.random.split(key, depth + 1)
        t_key, in_key, out_key = keys[0], keys[1], keys[-1]
        res_keys = keys[2:-1]

        self.time_embed = FourierFeatures(
            output_features=time_embed_dim,
            key=t_key
        )
        self.activation = activation
        
        # input consists of [x, t_embed, *parents]
        total_input_dim = input_dim + time_embed_dim + sum(context_dims)
        
        self.input_projection = eqx.nn.Linear(total_input_dim, hidden_dim, key=in_key)
        
        self.residual_layers = [
            ResidualLayer(
                key=res_keys[i],
                dim=hidden_dim,
                activation=activation
            )
            for i in range(max(0, depth - 2))
        ]
        
        self.output_projection = eqx.nn.Linear(hidden_dim, input_dim, key=out_key)

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, *parents: jnp.ndarray) -> jnp.ndarray:
        self.validate_input(t, x, *parents)
        
        # Standardize time to be (time_embed_dim,)
        t_embed = self.time_embed(t).squeeze()
        
        # Flat concatenation of [x, t_embed, parents]
        # Matches the ordering logic in ParallelVelocityFn/FlatVelocityFn (with parents added)
        flat_inputs = [x, t_embed] + list(parents)
        model_input = jnp.concatenate(flat_inputs, axis=-1)
        
        h = self.activation(self.input_projection(model_input))
        for layer in self.residual_layers:
            h = layer(h)
            
        return self.output_projection(h)


class StructuredVelocityFn(eqx.Module):
    """
    A velocity function structured according to a block dependency graph.
    """
    structure: Structure = eqx.field(static=True)
    # signature is (input_dim, Tuple[context_dims])
    _groups: List[Tuple[Any, List[str], eqx.Module]]

    def __init__(self, structure: Structure, key: jax.Array, block_constructor: Callable):
        if not isinstance(structure, Structure):
            raise TypeError(
                f"structure must be a Structure instance, got {type(structure)}"
            )
        
        object.__setattr__(self, "structure", structure)
        
        blocks = {}
        keys = jax.random.split(key, len(structure.block_order))
        
        for i, block_name in enumerate(structure.block_order):
            dim = structure.dims[block_name]
            parent_names = structure.dependencies.get(block_name, [])
            parent_dims = [structure.dims[p] for p in parent_names]

            blocks[block_name] = _call_block_constructor(
                block_constructor,
                block_name,
                input_dim=dim,
                context_dims=parent_dims,
                key=keys[i]
            )

        # Group by (block type, input_dim, context_dims): parameter stacking via
        # tree_map requires identical pytree structure, so blocks of different
        # types (e.g. ConvBlock vs ConditionalBlock) must never share a group.
        groups_map = {}
        for block_name in structure.block_order:
            blk = blocks[block_name]
            sig = (type(blk).__name__, blk.input_dim, tuple(blk.context_dims))
            if sig not in groups_map:
                groups_map[sig] = []
            groups_map[sig].append(block_name)

        _groups = []
        sorted_sigs = sorted(groups_map.keys(), key=lambda s: (s[0], s[1], len(s[2]), s[2]))
        
        for sig in sorted_sigs:
            group_names = groups_map[sig]
            group_instances = [blocks[name] for name in group_names]
            stacked_block = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *group_instances)
            _groups.append((sig, group_names, stacked_block))
            
        object.__setattr__(self, "_groups", _groups)


    @property
    def blocks(self) -> Dict[str, eqx.Module]:
        """
        Reconstruct the blocks dictionary from grouped stacked blocks.
        This ensures the blocks are always in sync with the trained parameters in _groups.
        """
        blocks = {}
        for _, group_names, stacked_block in self._groups:
            for i, name in enumerate(group_names):
                # Unstack the block
                blocks[name] = jax.tree_util.tree_map(lambda x: x[i], stacked_block)
        return blocks

    def __call__(self, x_all: Dict[str, jnp.ndarray], t: float, active_blocks: List[str] = None) -> Dict[str, jnp.ndarray]:
        if not isinstance(x_all, dict):
            raise TypeError(f"x_all must be a dict, got {type(x_all)}")
            
        if active_blocks is None:
            compute_blocks = self.structure.block_order
        else:
            compute_blocks = active_blocks
            
        provided_keys = set(x_all.keys())
        unknown_keys = provided_keys - set(self.structure.block_order)
        if unknown_keys:
            raise ValueError(f"Unknown blocks in x_all: {unknown_keys}")
            
        required_keys = set(compute_blocks)
        for block in compute_blocks:
            required_keys.update(self.structure.dependencies.get(block, []))
            
        missing_keys = required_keys - provided_keys
        if missing_keys:
            raise ValueError(f"Missing required blocks in x_all: {missing_keys}")
            
        t_arr = jnp.atleast_1d(jnp.asarray(t))
        if t_arr.shape != (1,):
            raise ValueError(f"Time t must be scalar or shape (1,), got shape {t_arr.shape}")

        def evaluate_group(stacked_block, t, xs_batch, *parents_batch_list):
            n_parents = len(parents_batch_list)
            in_axes = (0, None, 0) + (0,) * n_parents
            
            return eqx.filter_vmap(
                lambda b, t, x, *p: b(t, x, *p),
                in_axes=in_axes
            )(stacked_block, t, xs_batch, *parents_batch_list)

        v_all_flat = {}
        for sig, group_names, stacked_block in self._groups:
            if active_blocks is not None:
                relevant_indices = [i for i, name in enumerate(group_names) if name in compute_blocks]
                if not relevant_indices:
                    continue
                
                idx_arr = jnp.array(relevant_indices)
                sub_block = jax.tree_util.tree_map(lambda x: x[idx_arr], stacked_block)
                target_names = [group_names[i] for i in relevant_indices]
            else:
                target_names = group_names
                sub_block = stacked_block
            
            _block_type, input_dim, context_dims = sig

            try:
                xs_list = [x_all[name] for name in target_names]
                xs_batch = jnp.stack(xs_list)
            except KeyError as e:
                 raise ValueError(f"Missing state for block '{e.args[0]}'")

            parents_batch_list = []
            for i in range(len(context_dims)):
                p_list = []
                for name in target_names:
                    p_name = self.structure.dependencies[name][i]
                    p_list.append(x_all[p_name])
                
                parents_batch_list.append(jnp.stack(p_list))
                
            velocities = evaluate_group(sub_block, t_arr, xs_batch, *parents_batch_list)
            
            for i, name in enumerate(target_names):
                v_all_flat[name] = velocities[i]

        return v_all_flat


class SimpleStructuredVelocityFn(eqx.Module):
    """
    A simple, sequential implementation of StructuredVelocityFn.
    Useful for verification and as a baseline.
    """
    structure: Structure = eqx.field(static=True)
    blocks: Dict[str, eqx.Module]

    def __init__(self, structure: Structure, key: jax.Array, block_constructor: Callable):
        if not isinstance(structure, Structure):
            raise TypeError(
                f"structure must be a Structure instance, got {type(structure)}"
            )
        
        self.structure = structure
        
        blocks = {}
        keys = jax.random.split(key, len(structure.block_order))
        
        for i, block_name in enumerate(structure.block_order):
            dim = structure.dims[block_name]
            parent_names = structure.dependencies.get(block_name, [])
            parent_dims = [structure.dims[p] for p in parent_names]

            blocks[block_name] = _call_block_constructor(
                block_constructor,
                block_name,
                input_dim=dim,
                context_dims=parent_dims,
                key=keys[i]
            )

        self.blocks = blocks

    def __call__(self, x_all: Dict[str, jnp.ndarray], t: float, active_blocks: List[str] = None) -> Dict[str, jnp.ndarray]:
        if not isinstance(x_all, dict):
            raise TypeError(f"x_all must be a dict, got {type(x_all)}")
            
        if active_blocks is None:
            compute_blocks = self.structure.block_order
        else:
            compute_blocks = active_blocks
            
        t_arr = jnp.atleast_1d(jnp.asarray(t))
        
        v_all = {}
        for block_name in compute_blocks:
            block = self.blocks[block_name]
            parent_names = self.structure.dependencies.get(block_name, [])
            parents = [x_all[p] for p in parent_names]
            
            v_all[block_name] = block(t_arr, x_all[block_name], *parents)
            
        return v_all
