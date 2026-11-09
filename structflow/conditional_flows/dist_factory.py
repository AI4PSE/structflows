"""
Factory functions for creating ConditionalDistribution instances.

This module provides core distribution creation logic, organized by distribution type.
It does NOT handle config parsing - that's delegated to callers.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any, Optional

from sinterp.stochasticfield import StochasticField, OneSidedField_eta1
from structflow.bayes.distribution import Gaussian
from structflow.conditional_flows.static.dist import StaticConditionalDistribution
from structflow.conditional_flows.static.flow import StaticConditionalFlow
from structflow.conditional_flows.guidance.dist import GuidanceBasedDistribution
from structflow.conditional_flows.guidance.flow import GuidedFlow
from structflow.conditional_flows.dist import NormalisedDistribution
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.parallel.dist import ParallelBlockTriangularDistribution
from structflow.conditional_flows.parallel.flow import ParallelFlow


# ============================================================================
# Adapter Classes
# ============================================================================

class BlockModelAdapter(eqx.Module):
    """
    Swaps arguments to make a ConditionalBlock compatible with StochasticField.
    Field expects: model(x, t, *args)
    Block expects: net(t, x, *args)
    """
    net: eqx.Module
    
    def __call__(self, x, t, *args):
        return self.net(t, x, *args)


class BlockFieldAdapter(eqx.Module):
    """
    Wraps a single block's network in a mini-OneSidedField to compute velocity.
    Exposes the signature expected by StaticConditionalFlow: (t, x, *parents).
    """
    field: OneSidedField_eta1

    def __call__(self, t, x, *parents):
        # Delegate to the field's logic!
        # This reuses the exact singularity fix/clamping defined in your Field class.
        return self.field.get_velocity_b(x, t, *parents)


class StructuredFieldAdapter(eqx.Module):
    """
    Wraps the main OneSidedField_eta1.
    1. Exposes .structure from the internal model.
    2. Implements __call__ via field.get_velocity_b.
    3. Implements .blocks by creating BlockFieldAdapters on the fly.
    """
    field: OneSidedField_eta1

    @property
    def structure(self):
        # Delegate to the internal StructuredVelocityFn
        return self.field.net.structure

    @property
    def blocks(self):
        # Create adapters for each block so they act like velocity functions
        # This allows StaticConditionalFlow to work without knowing about 'eta1'
        adapters = {}
        for name, block_net in self.field.net.blocks.items():
            # 1. Adapt the block net to accept (x, t) order
            net_adapter = BlockModelAdapter(block_net)
            
            # 2. Create a mini-field for this block using the SAME interpolator
            # This ensures consistent gamma/time definition
            block_field = OneSidedField_eta1(self.field.interpolator, net_adapter)
            
            # 3. Wrap it to accept (t, x) order required by PartialVelocityFn
            adapters[name] = BlockFieldAdapter(block_field)
            
        return adapters

    def __call__(self, x_all, t, active_blocks=None):
        if active_blocks is None:
            # Directly use the field's velocity method for the full joint state.
            return self.field.get_velocity_b(x_all, t)
            
        # If we have active blocks, we compute velocity for just those blocks.
        # We manually perform the OneSidedField solve logic for the subset to
        # avoid tree_map mismatch errors in self.field.get_velocity_b if x_all
        # contains extra context keys.
        
        # 1. Get predicted x1 for active blocks using FULL context
        eta_1_val = self.field.get_denoiser_eta1(x_all, t, active_blocks)
        
        # 2. Subset the state to match predicted blocks for the solve logic
        x_active = {k: x_all[k] for k in active_blocks}
        
        # 3. Solve for velocity: v = gamma_dot * eta_z + beta_dot * eta_1
        # where eta_z = (x_t - beta * eta_1) / gamma
        t_safe = jnp.minimum(t, 0.99)
        gamma_t = self.field.interpolator.gamma(t_safe)
        beta_t = self.field.interpolator.beta(t_safe)
        gamma_dot = self.field.interpolator.dgammadt(t_safe)
        beta_dot = self.field.interpolator.dbetadt(t_safe)
        safe_gamma = jnp.maximum(gamma_t, 1e-3)
        
        # Solve for eta_z (only for active blocks)
        eta_z_val = jax.tree_util.tree_map(
            lambda _e1, _xt: (_xt - beta_t * _e1) / safe_gamma,
            eta_1_val, x_active
        )
        
        # Combine into velocity
        return jax.tree_util.tree_map(
            lambda _ez, _e1: gamma_dot * _ez + beta_dot * _e1,
            eta_z_val, eta_1_val
        )


class ParallelFieldAdapter(eqx.Module):
    """
    Wraps a Parallel StochasticField to expose the signature expected by ParallelFlow.
    Signature: (x, t, active_blocks=None)
    """
    field: StochasticField

    @property
    def structure(self):
        return self.field.net.structure

    @property
    def block_offsets(self):
        return self.field.net.block_offsets

    @property
    def output_dim(self):
        return self.field.net.output_dim

    @property
    def time_embed(self):
        return self.field.net.time_embed

    @property
    def hidden_block_dims(self):
        return self.field.net.hidden_block_dims

    @property
    def layers(self):
        return self.field.net.layers

    def __call__(self, x, t, active_blocks=None):
        return self.field.get_velocity_b(x, t)


# ============================================================================
# Distribution Factory Functions
# ============================================================================

def _create_base_distributions(structure: BlockTriangularStructure) -> Dict[str, Any]:
    """
    Create Gaussian base distributions for each block.
    """
    return {
        block: Gaussian(mean=jnp.zeros(dim), cov=jnp.eye(dim))
        for block, dim in structure.dims.items()
    }


def create_static_distribution(
    model: Any, 
    structure: BlockTriangularStructure,
    n_steps: int = 50,
    integrator_name: str = 'diffrax'
) -> StaticConditionalDistribution:
    """
    Create a StaticConditionalDistribution.
    
    Args:
        model: Either an OneSidedField_eta1 or a model with a .net attribute (velocity function).
        structure: BlockTriangularStructure defining the dependency graph.
        n_steps: Number of integration steps.
        integrator_name: Name of the integrator to use.
        
    Returns:
        StaticConditionalDistribution instance.
    """
    if isinstance(model, OneSidedField_eta1):
        velocity_fn = StructuredFieldAdapter(model)
    else:
        velocity_fn = model.net if hasattr(model, 'net') else model
        
    flow = StaticConditionalFlow(velocity_fn, n_steps=n_steps, integrator_name=integrator_name)
    base_distributions = _create_base_distributions(structure)
    
    return StaticConditionalDistribution(flow, base_distributions)


def create_guided_distribution(
    model: Any,
    structure: BlockTriangularStructure,
    n_steps: int = 50,
    integrator_name: str = 'diffrax',
    guidance_scale: float = 7.0
) -> GuidanceBasedDistribution:
    """
    Create a GuidanceBasedDistribution (joint mode with guided flows).
    
    Args:
        model: A StochasticField instance.
        structure: BlockTriangularStructure defining the dependency graph.
        n_steps: Number of integration steps.
        integrator_name: Name of the integrator to use.
        guidance_scale: Guidance scale parameter.
        
    Returns:
        GuidanceBasedDistribution instance.
    """
    from structflow.bayes.distribution import DictDistribution
    
    guided_flow = GuidedFlow(
        model=model,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        integrator_name=integrator_name
    )
    base_distributions = _create_base_distributions(structure)
    
    return GuidanceBasedDistribution(
        flow=guided_flow,
        base_distribution=DictDistribution(base_distributions),
        key_order=structure.block_order
    )


def create_dcm_distribution(
    model: Any,
    structure: BlockTriangularStructure,
    n_steps: int = 50,
) -> Any:
    """
    Create a DCMDistribution (per-node diffusion backbone).

    Args:
        model: A StructuredVelocityFn container of DiffusionBlocks
            (as returned by create_dcm_flow).
        structure: BlockTriangularStructure defining the dependency graph.
        n_steps: Number of DDIM steps (NFE) for sampling and inversion.

    Returns:
        DCMDistribution instance.
    """
    from structflow.conditional_flows.dcm.dist import DCMDistribution

    base_distributions = _create_base_distributions(structure)
    return DCMDistribution(model, base_distributions, n_steps=n_steps)


def create_parallel_distribution(
    model: OneSidedField_eta1,
    structure: BlockTriangularStructure,
    n_steps: int = 50,
    integrator_name: str = 'diffrax'
) -> ParallelBlockTriangularDistribution:
    """
    Create a ParallelBlockTriangularDistribution (dynamic mode).
    
    Args:
        model: Either an OneSidedField_eta1 or a model with a .net attribute (velocity function).
        structure: BlockTriangularStructure defining the dependency graph.
        n_steps: Number of integration steps.
        integrator_name: Name of the integrator to use.
        
    Returns:
        ParallelBlockTriangularDistribution instance.
    """
    velocity_fn = ParallelFieldAdapter(model)
    flow = ParallelFlow(velocity_fn, n_steps=n_steps, integrator_name=integrator_name)
    base_distributions = _create_base_distributions(structure)
    
    return ParallelBlockTriangularDistribution(flow, base_distributions)


def create_dynamicv2_distribution(
    model: Any,
    structure: BlockTriangularStructure,
    n_steps: int = 50,
    integrator_name: str = 'diffrax'
) -> Any:
    """
    Create a DynamicV2Distribution (weight clipping + trajectory interpolation).

    Args:
        model: Either an OneSidedField_eta1 or a model with a .net attribute
            (velocity function).
        structure: BlockTriangularStructure defining the dependency graph.
        n_steps: Number of integration steps.
        integrator_name: Name of the integrator to use.

    Returns:
        DynamicV2Distribution instance.
    """
    from structflow.conditional_flows.dynamicv2 import DynamicV2Flow, DynamicV2Distribution

    velocity_fn = ParallelFieldAdapter(model)
    parallel_flow = ParallelFlow(velocity_fn, n_steps=n_steps,
                                  integrator_name=integrator_name)
    dynamicv2_flow = DynamicV2Flow(parallel_flow)
    base_distributions = _create_base_distributions(structure)

    return DynamicV2Distribution(dynamicv2_flow, base_distributions)


# ============================================================================
# Main Factory Function
# ============================================================================

def create_distribution(
    mode: str,
    structure_dict: Dict[str, Any],
    model: Any,
    normaliser: Optional[Any] = None,
    n_steps: int = 50,
    integrator_name: str = 'diffrax',
    guidance_scale: float = 7.0
) -> Any:
    """
    Create a ConditionalDistribution from model parameters.
    
    This function dispatches to specific factory functions based on the mode.
    It does NOT parse config dicts - callers should extract and pass individual parameters.
    
    Args:
        mode: Distribution type: 'static', 'joint' (guided), 'dynamic' (parallel), 'dcm', or 'carefl'.
        structure_dict: Dictionary with keys: 'dependencies', 'block_order', 'dims'.
        model: The trained model (specific type depends on mode).
        normaliser: Optional normaliser to wrap the distribution.
        n_steps: Number of integration steps.
        integrator_name: Name of the integrator.
        guidance_scale: Guidance scale (only for 'joint' mode).
        
    Returns:
        A ConditionalDistribution instance.
    """
    structure = BlockTriangularStructure(
        dependencies=structure_dict['dependencies'],
        block_order=structure_dict['block_order'],
        dims=structure_dict['dims']
    )
    
    if mode == 'static':
        dist = create_static_distribution(
            model=model,
            structure=structure,
            n_steps=n_steps,
            integrator_name=integrator_name
        )
    elif mode == 'guided':
        dist = create_guided_distribution(
            model=model,
            structure=structure,
            n_steps=n_steps,
            integrator_name=integrator_name,
            guidance_scale=guidance_scale
        )
    # elif mode == 'dynamic':
    #     dist = create_parallel_distribution(
    #         model=model,
    #         structure=structure,
    #         n_steps=n_steps,
    #         integrator_name=integrator_name
    #     )
    elif mode in ['dynamic', 'dynamicv2']:
        dist = create_dynamicv2_distribution(
            model=model,
            structure=structure,
            n_steps=n_steps,
            integrator_name=integrator_name
        )
    elif mode == 'dcm':
        dist = create_dcm_distribution(
            model=model,
            structure=structure,
            n_steps=n_steps
        )
    elif mode in ['carefl', 'universal']:
        dist = model
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if normaliser is not None:
        return NormalisedDistribution(dist=dist, normaliser=normaliser)
    
    return dist
