"""Factory functions for creating flow models.

This module provides separate factory functions for each flow mode (joint, static, dynamic, carefl).
Each function takes prepared parameters (no config parsing) and returns (model, step_fn).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import vmap, random
from typing import Dict, Any, Callable, Optional, Tuple, Type, Union

from structflow.bayes.distribution import IsotropicLaplace
from structflow.conditional_flows.static.net import StructuredVelocityFn, ConditionalBlock
from structflow.conditional_flows.static.flow import PartialVelocityFn
from structflow.conditional_flows.dist import StructuredPGMDistribution
from structflow.conditional_flows.carefl.net import ResNetConditioner
from structflow.pgms.structural_pgm import StructuralPGM
from structflow.pgms.mechanisms import (
    CAREFLMechanism,
    ConditionerMechanism,
    UniversalMechanism,
    make_universal_mechanism,
    UniversalConditionerMechanism
)
from sinterp.interpolants import OneSidedLinear
from sinterp.stochasticfield import OneSidedField_eta1
from sinterp.couplings import SinkhornCoupling, C2OTCoupling


# Adaptive-weight target for use_coupling='c2ot' (the value used in the paper's experiments;
# sinterp's own C2OTCoupling default, 0.01, is the C2OT paper's).
C2OT_TARGET_RATIO = 0.05

COUPLING_DOC = """use_coupling: How noise is paired with data in each training batch.
            False: independent. True / 'joint': Sinkhorn OT over the joint vector with a
            barycentric projection (legacy; the paired noise is an average of noise
            rows, so its variance shrinks, and for conditional models it depends on the
            condition). 'hungarian': exact OT permutation over the joint vector (the noise
            stays exactly N(0, I)). 'c2ot': one permutation per block, matching the block's
            values with its parents as the C2OT condition; with a fixed block order this is a
            minibatch Knothe-Rosenblatt coupling (target ratio C2OT_TARGET_RATIO). Pass a
            C2OTCoupling instance to set its options.
            Defaults, one per mode, are the couplings that keep each mode consistent: static
            False (each block's network sees its conditions as clean inputs, so independent
            pairing is exact), dynamic 'c2ot' (independent or joint pairing biases the
            block-triangular optimum), guided 'hungarian' (models the joint, which exact OT
            preserves)."""


def _resolve_coupling(use_coupling):
    """Map the ``use_coupling`` option to (kind, coupling); see COUPLING_DOC."""
    if isinstance(use_coupling, C2OTCoupling):
        return 'c2ot', use_coupling
    if use_coupling is False or use_coupling is None:
        return 'none', None
    if use_coupling is True or use_coupling == 'joint':
        return 'joint', SinkhornCoupling()
    if use_coupling == 'hungarian':
        return 'hungarian', C2OTCoupling(w=0.0)
    if use_coupling == 'c2ot':
        return 'c2ot', C2OTCoupling(target_ratio=C2OT_TARGET_RATIO)
    raise ValueError(f"use_coupling must be True, False, 'joint', 'hungarian', 'c2ot' "
                     f"or a C2OTCoupling, got {use_coupling!r}")


def _couple(kind, coupling, key, x_dict, e_dict, structure):
    """Apply the coupling resolved by _resolve_coupling to one batch."""
    if kind == 'none':
        return x_dict, e_dict
    if kind == 'c2ot':
        return x_dict, coupling.couple_blocks(key, x_dict, e_dict, structure.dependencies)
    x_dict, e_dict, _, _, _ = coupling(key, x_dict, e_dict)
    return x_dict, e_dict


def create_joint_flow(
    structure: Any,
    depth: int,
    width: int,
    model_key: Any,
    optimizer: Any,
    use_coupling: Union[bool, str] = 'hungarian',
    normaliser: Optional[Any] = None,
) -> Tuple[Any, Callable]:
    """Create a joint (flat) flow model.
    
    Args:
        structure: BlockTriangularStructure with dims and block_order
        depth: Depth of the neural network
        width: Hidden dimension of the network
        model_key: JAX random key for model initialization
        optimizer: Optax optimizer
        use_coupling: True/'joint', 'hungarian', 'c2ot', False or a C2OTCoupling;
            see COUPLING_DOC above.
        normaliser: Optional DictNormaliser for normalization
        
    Returns:
        Tuple of (model, step_fn)
    """
    from structflow.conditional_flows.guidance.net import FlatVelocityFn
    
    total_dim = sum(structure.dims.values())
    interpolator = OneSidedLinear()
    coupling_kind, coupling = _resolve_coupling(use_coupling)
    
    net = FlatVelocityFn(
        input_dim=total_dim,
        hidden_dim=width,
        depth=depth,
        key=model_key,
        time_embed_dim=16
    )
    
    model = OneSidedField_eta1(interpolator, net)
    
    @eqx.filter_jit
    def step_fn(model, opt_state, normaliser, x_dict_batch, k):
        if normaliser is not None:
            normaliser = normaliser.update(x_dict_batch)
            x_dict_batch = normaliser.normalise(x_dict_batch)
        noise_key, time_key, coupling_key = random.split(k, 3)

        e_dict_batch = {
            name: random.normal(random.fold_in(noise_key, i), x_dict_batch[name].shape)
            for i, name in enumerate(structure.block_order)
        }
        x0_dict_batch = jax.tree_util.tree_map(jnp.zeros_like, x_dict_batch)
        batch_size = x_dict_batch[structure.block_order[0]].shape[0]
        t_batch = random.uniform(time_key, (batch_size, 1))

        x_dict_batch, e_dict_batch = _couple(coupling_kind, coupling, coupling_key,
                                             x_dict_batch, e_dict_batch, structure)

        # Helper to flatten batch dict -> (batch, total_dim)
        def flatten_batch(d):
            return jnp.concatenate([d[b] for b in structure.block_order], axis=1)

        def loss_fn(m):
            x0_flat = flatten_batch(x0_dict_batch)
            x1_flat = flatten_batch(x_dict_batch)
            e_flat = flatten_batch(e_dict_batch)
            
            losses = vmap(m.total_loss)(x0_flat, x1_flat, e_flat, t_batch)
            return jnp.mean(losses)
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, normaliser, loss
    
    return model, step_fn


def create_static_flow(
    structure: Any,
    depth: int,
    width: int,
    model_key: Any,
    optimizer: Any,
    use_coupling: Union[bool, str] = False,
    normaliser: Optional[Any] = None,
    block_shapes: Optional[Dict[str, Any]] = None,
    conv_channels: int = 64,
) -> Tuple[Any, Callable]:
    """Create a static (structured) flow model.

    Args:
        structure: BlockTriangularStructure with dependencies, block_order, and dims
        depth: Depth of the neural network
        width: Hidden dimension of each block's network
        model_key: JAX random key for model initialization
        optimizer: Optax optimizer
        use_coupling: True/'joint', 'hungarian', 'c2ot', False or a C2OTCoupling;
            see COUPLING_DOC above.
        normaliser: Optional DictNormaliser for normalization
        block_shapes: Optional {block_name: (H, W, C)} — blocks listed here get a
            convolutional U-Net (ConvBlock) instead of the MLP ConditionalBlock.
        conv_channels: Base channel count for ConvBlocks.

    Returns:
        Tuple of (model, step_fn)
    """
    from structflow.conditional_flows.conv import ConvBlock

    interpolator = OneSidedLinear()
    coupling_kind, coupling = _resolve_coupling(use_coupling)

    def block_constructor(input_dim, context_dims, key, name=None):
        if block_shapes and name in block_shapes:
            return ConvBlock(input_dim, context_dims, key,
                             shape=tuple(block_shapes[name]),
                             base_channels=conv_channels)
        return ConditionalBlock(input_dim, context_dims, key,
                                width=width,
                                depth=depth)

    structured_net = StructuredVelocityFn(structure, model_key, block_constructor)
    model = OneSidedField_eta1(interpolator, structured_net)
    
    @eqx.filter_jit
    def step_fn(model, opt_state, normaliser, x_dict_batch, k, mask_dict=None):
        """One flow-matching step.

        ``mask_dict`` optionally supplies a per-block boolean mask of shape
        (batch,), marking the rows from which that block's loss term may be
        estimated. The flow-matching loss is already a sum of block-local
        terms -- block ``v``'s term reads only ``x1_all[v]`` and
        ``x1_all[p]`` for ``p in parents(v)`` -- so a row that is missing some
        variable can still train every block whose own value and parents it
        does observe. With ``mask_dict=None`` the original behaviour is
        reproduced exactly, including the accumulation order of the block sum.

        Because a block's term touches only its own value and its parents, and
        a block is masked in only where those are present, placeholder values
        standing in for unobserved entries cannot reach any term that is
        counted. The per-block mean is taken over that block's own admissible
        rows rather than over the batch, so blocks with fewer admissible rows
        are not silently down-weighted relative to the unmasked objective.
        """
        if normaliser is not None:
            normaliser = normaliser.update(x_dict_batch)
            x_dict_batch = normaliser.normalise(x_dict_batch)
        noise_key, time_key, coupling_key = random.split(k, 3)
        
        e_dict_batch = {
            name: random.normal(random.fold_in(noise_key, i), x_dict_batch[name].shape)
            for i, name in enumerate(structure.block_order)
        }
        
        t_batch = random.uniform(time_key, (x_dict_batch[structure.block_order[0]].shape[0], 1))

        x_dict_batch, e_dict_batch = _couple(coupling_kind, coupling, coupling_key,
                                             x_dict_batch, e_dict_batch, structure)

        def loss_fn(m):
            def block_fm_loss(x1_all, e1_all, t, block):
                parent_names = structure.dependencies.get(block, [])
                ctx = {p: x1_all[p] for p in parent_names}

                v = PartialVelocityFn(m.net, ctx, [block])
                block_model = OneSidedField_eta1(interpolator, v)

                return block_model.total_loss(jnp.zeros_like(x1_all[block]), x1_all[block], e1_all[block], t)

            if mask_dict is None:
                def aggregate_fm_loss(x1_all, e1_all, t):
                    loss = 0.0
                    for block in structure.block_order:
                        loss += block_fm_loss(x1_all, e1_all, t, block)
                    return loss

                losses = vmap(aggregate_fm_loss)(x_dict_batch, e_dict_batch, t_batch)
                return jnp.mean(losses)

            def per_block_fm_loss(x1_all, e1_all, t):
                return jnp.stack([block_fm_loss(x1_all, e1_all, t, block)
                                  for block in structure.block_order])

            losses = vmap(per_block_fm_loss)(x_dict_batch, e_dict_batch, t_batch)
            w = jnp.stack([mask_dict[block].astype(losses.dtype)
                           for block in structure.block_order], axis=1)
            counts = jnp.maximum(w.sum(axis=0), 1.0)
            return ((losses * w).sum(axis=0) / counts).sum()
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, normaliser, loss
    
    return model, step_fn


def create_dynamic_flow(
    structure: Any,
    depth: int,
    width: int,
    model_key: Any,
    optimizer: Any,
    use_coupling: Union[bool, str] = 'c2ot',
    use_sparse: bool = False,
    normaliser: Optional[Any] = None,
) -> Tuple[Any, Callable]:
    """Create a dynamic (parallel) flow model.
    
    Args:
        structure: BlockTriangularStructure with dependencies, block_order, and dims
        depth: Depth of the neural network
        width: Hidden dimension per block
        model_key: JAX random key for model initialization
        optimizer: Optax optimizer
        use_coupling: True/'joint', 'hungarian', 'c2ot', False or a C2OTCoupling;
            see COUPLING_DOC above.
        use_sparse: Whether to use sparse interactions
        normaliser: Optional DictNormaliser for normalization
        
    Returns:
        Tuple of (model, step_fn)
    """
    from structflow.conditional_flows.dynamic.net import ParallelStructuredFn
    
    interpolator = OneSidedLinear()
    coupling_kind, coupling = _resolve_coupling(use_coupling)
    
    n_blocks = len(structure.block_order)
    total_hidden_dim = width * n_blocks
    
    net = ParallelStructuredFn(
        structure=structure,
        hidden_dim=total_hidden_dim,
        depth=depth,
        key=model_key,
        time_embed_dim=16,
        use_sparse=use_sparse
    )
    
    model = OneSidedField_eta1(interpolator, net)
    
    @eqx.filter_jit
    def step_fn(model, opt_state, normaliser, x_dict_batch, k):
        if normaliser is not None:
            normaliser = normaliser.update(x_dict_batch)
            x_dict_batch = normaliser.normalise(x_dict_batch)
        noise_key, time_key, coupling_key = random.split(k, 3)

        e_dict_batch = {
            name: random.normal(random.fold_in(noise_key, i), x_dict_batch[name].shape)
            for i, name in enumerate(structure.block_order)
        }
        
        batch_size = x_dict_batch[structure.block_order[0]].shape[0]
        t_batch = random.uniform(time_key, (batch_size, 1))
        
        x0_dict_batch = jax.tree_util.tree_map(jnp.zeros_like, x_dict_batch)

        x_dict_batch, e_dict_batch = _couple(coupling_kind, coupling, coupling_key,
                                             x_dict_batch, e_dict_batch, structure)

        def flatten_batch(d):
            return jnp.concatenate([d[b] for b in structure.block_order], axis=1)
            
        def loss_fn(m):
            x0_flat = flatten_batch(x0_dict_batch)
            x1_flat = flatten_batch(x_dict_batch)
            e_flat = flatten_batch(e_dict_batch)
            
            losses = vmap(m.total_loss)(x0_flat, x1_flat, e_flat, t_batch)
            return jnp.mean(losses)
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, normaliser, loss
    
    return model, step_fn


def create_carefl_flow(
    structure: Any,
    depth: int,
    width: int,
    model_key: Any,
    optimizer: Any,
    normaliser: Optional[Any] = None,
) -> Tuple[Any, Callable]:
    """Create a CAREFL (identifiable) flow model.
    
    Args:
        structure: BlockTriangularStructure with dependencies, block_order, and dims
        depth: Depth of the neural network
        width: Hidden dimension for each block's conditioner
        model_key: JAX random key for model initialization
        optimizer: Optax optimizer
        normaliser: Optional DictNormaliser for normalization
        
    Returns:
        Tuple of (model, step_fn)
    """
    conditioners = {}
    base_distributions = {}
    
    # 1. Build Conditioners & Base Dists per Block
    mechanisms = {}
    for i, block in enumerate(structure.block_order):
        parent_blocks = structure.dependencies.get(block, [])
        
        input_dim = sum(structure.dims[p] for p in parent_blocks)
        output_dim = structure.dims[block]
        
        model_key, c_key = random.split(model_key)
        
        conditioners[block] = ResNetConditioner(
            input_dim=input_dim,
            output_dim=2 * output_dim, # Affine parameters (s, t)
            key=c_key,
            hidden_dim=width,
            depth=depth
        )
        
        # Wrap in the new Mechanism framework
        mechanisms[block] = ConditionerMechanism(
            parent_names=parent_blocks,
            input_dim=input_dim,
            output_dim=output_dim,
            conditioner=conditioners[block]
        )
        
        base_distributions[block] = IsotropicLaplace(dim=output_dim)

    # 2. Build Model
    pgm = StructuralPGM(
        structure=structure,
        mechanisms=mechanisms,
        base_distributions=base_distributions
    )
    
    # Use the lightweight StructuredPGMDistribution
    # This replaces the heavier CAREFLDistribution wrapper while keeping the same logic
    model = StructuredPGMDistribution(pgm)
    
    # 3. Step Function
    @eqx.filter_jit
    def step_fn(model, opt_state, normaliser, x_dict_batch, k):
        if normaliser is not None:
            normaliser = normaliser.update(x_dict_batch)
            x_dict_batch = normaliser.normalise(x_dict_batch)
        
        def loss_fn(m):
            log_probs = m.b_log_prob(x_dict_batch)
            return -jnp.mean(log_probs)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g), grads)
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1e3, 1e3), grads)

        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, normaliser, loss
    
    return model, step_fn


def create_dcm_flow(
    structure: Any,
    depth: int,
    width: int,
    model_key: Any,
    optimizer: Any,
    normaliser: Optional[Any] = None,
) -> Tuple[Any, Callable]:
    """Create a DCM (per-node diffusion) model: one conditional ε-prediction
    network per block, wired by the same BlockTriangularStructure as the
    static (CNF) backbone.

    The step_fn is the standard ε-MSE denoising loss over uniformly sampled
    continuous t — structurally parallel to create_static_flow's
    aggregate_fm_loss, summing a per-block loss over structure.block_order
    with each block conditioned on its *clean* parents.

    Note: **no OT coupling** (SinkhornCoupling) — the denoising objective
    marginalizes over its own forward corruption and has no analogue of the
    coupling's data/noise pairing. Record this asymmetry when reporting the
    backbone ablation.

    Args:
        structure: BlockTriangularStructure with dependencies, block_order, and dims
        depth: Depth of each block's ε-network
        width: Hidden dimension of each block's ε-network
        model_key: JAX random key for model initialization
        optimizer: Optax optimizer
        normaliser: Optional DictNormaliser for normalization

    Returns:
        Tuple of (model, step_fn); model is the StructuredVelocityFn container
        of DiffusionBlocks (interpret outputs as ε̂).
    """
    from structflow.conditional_flows.dcm.net import build_structured_eps_fn
    from structflow.conditional_flows.dcm.schedule import alpha_bar

    model = build_structured_eps_fn(structure, model_key, width=width, depth=depth)

    @eqx.filter_jit
    def step_fn(model, opt_state, normaliser, x_dict_batch, k):
        if normaliser is not None:
            normaliser = normaliser.update(x_dict_batch)
            x_dict_batch = normaliser.normalise(x_dict_batch)
        noise_key, time_key = random.split(k, 2)

        e_dict_batch = {
            name: random.normal(random.fold_in(noise_key, i), x_dict_batch[name].shape)
            for i, name in enumerate(structure.block_order)
        }

        batch_size = x_dict_batch[structure.block_order[0]].shape[0]
        t_batch = random.uniform(time_key, (batch_size, 1))

        def loss_fn(m):
            def aggregate_eps_loss(x1_all, e_all, t):
                # t has shape (1,); alpha_bar broadcasts.
                ab = alpha_bar(t)
                loss = 0.0
                for block in structure.block_order:
                    parent_names = structure.dependencies.get(block, [])
                    # Corrupt the block's own value; parents stay clean.
                    x_t = jnp.sqrt(ab) * x1_all[block] + jnp.sqrt(1.0 - ab) * e_all[block]
                    ctx = {p: x1_all[p] for p in parent_names}
                    eps_hat = m({block: x_t, **ctx}, t, active_blocks=[block])[block]
                    loss += jnp.mean((eps_hat - e_all[block]) ** 2)
                return loss

            losses = vmap(aggregate_eps_loss)(x_dict_batch, e_dict_batch, t_batch)
            return jnp.mean(losses)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, normaliser, loss

    return model, step_fn


def create_universal_flow(
    structure: Any,
    depth: int,
    width: int,
    model_key: Any,
    optimizer: Any,
    num_bins: int = 8,
    normaliser: Optional[Any] = None,
) -> Tuple[Any, Callable]:
    """Create a fully nonlinear (Universal) flow model using Spline mechanisms.
    
    Uses ResNet conditioners to parameterize the splines.
    
    Args:
        structure: BlockTriangularStructure with dependencies, block_order, and dims
        depth: Depth of the parameter networks
        width: Hidden dimension for each block's parameter network
        model_key: JAX random key for model initialization
        optimizer: Optax optimizer
        num_bins: Number of spline bins
        normaliser: Optional DictNormaliser for normalization
        
    Returns:
        Tuple of (model, step_fn)
    """
    mechanisms = {}
    
    for block in structure.block_order:
        parent_blocks = structure.dependencies.get(block, [])
        parent_dims = [structure.dims[p] for p in parent_blocks]
        input_dim = sum(parent_dims)
        output_dim = structure.dims[block]
        
        model_key, m_key, c_key = random.split(model_key, 3)
        
        # Universal mechanism params size: 3 * num_bins + 1 (widths, heights, derivatives)
        param_dim = 3 * num_bins + 1
        total_param_dim = output_dim * param_dim
        
        conditioner = ResNetConditioner(
            input_dim=input_dim,
            output_dim=total_param_dim,
            key=c_key,
            hidden_dim=width,
            depth=depth
        )
        
        mechanisms[block] = UniversalConditionerMechanism(
            parent_names=parent_blocks,
            input_dim=input_dim,
            output_dim=output_dim,
            conditioner=conditioner,
            key=m_key,
            num_bins=num_bins
        )
    
    # Build Model
    pgm = StructuralPGM(structure, mechanisms)
    model = StructuredPGMDistribution(pgm)
    
    # Step Function
    @eqx.filter_jit
    def step_fn(model, opt_state, normaliser, x_dict_batch, k):
        if normaliser is not None:
            normaliser = normaliser.update(x_dict_batch)
            x_dict_batch = normaliser.normalise(x_dict_batch)
        
        def loss_fn(m):
            # StructuredPGMDistribution supports b_log_prob via the base class
            log_probs = m.b_log_prob(x_dict_batch)
            return -jnp.mean(log_probs)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

        # Robustness for training splines
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g), grads)
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1e3, 1e3), grads)

        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, normaliser, loss
    
    return model, step_fn
