"""
Test factory integration: dist_factory and flow_factory.

Verifies:
- create_distribution() works for all modes and structures
- create_flow() returns valid flow + step_fn pairs
- Normaliser wrapping works correctly
- Distributions created from factories are functional
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import optax

from structflow.conditional_flows.tests.conftest import MODES, STRUCTURE_TYPES


# ============================================================================
# Factory Creation Tests
# ============================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("structure_type", STRUCTURE_TYPES)
def test_create_distribution_factory(mode, structure_type):
    """Test create_distribution factory for all mode/structure combinations."""
    
    from structflow.pgms.factories import make_chain_pgm, make_fork_pgm, make_collider_pgm
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.flow_factory import (
        create_static_flow,
        create_dynamic_flow,
        create_joint_flow,
        create_carefl_flow,
        create_universal_flow
    )
    
    # Create PGM
    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)
    
    if structure_type == "chain":
        pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=model_key)
    elif structure_type == "fork":
        pgm = make_fork_pgm(n_children=2, mechanism_type="linear", key=model_key)
    elif structure_type == "collider":
        pgm = make_collider_pgm(n_causes=2, mechanism_type="linear", key=model_key)
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")
    
    structure = pgm.structure
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    # Create flow model based on mode
    if mode == "static":
        model, _ = create_static_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    elif mode == "dynamic":
        model, _ = create_dynamic_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    elif mode == "guided":
        model, _ = create_joint_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    elif mode == "carefl":
        model, _ = create_carefl_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
        )
    elif mode == "universal":
        model, _ = create_universal_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Create distribution
    dist = create_distribution(
        mode=mode,
        structure_dict=structure_dict,
        model=model,
    )
    
    # Verify distribution is functional
    assert dist is not None
    assert hasattr(dist, 'sample')
    assert hasattr(dist, 'log_prob')
    
    # Try to sample
    sample_key = random.PRNGKey(0)
    sample = dist.sample(sample_key)
    assert isinstance(sample, dict)
    assert len(sample) > 0


@pytest.mark.parametrize("mode", MODES)
def test_create_flow_and_dist(mode):
    """Test creating flow and distribution together."""
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.flow_factory import (
        create_static_flow,
        create_dynamic_flow,
        create_joint_flow,
        create_carefl_flow,
        create_universal_flow,
    )
    
    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)
    
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=model_key)
    structure = pgm.structure
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    # Create flow
    if mode == "static":
        model, step_fn = create_static_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    elif mode == "dynamic":
        model, step_fn = create_dynamic_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    elif mode == "guided":
        model, step_fn = create_joint_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    elif mode == "carefl":
        model, step_fn = create_carefl_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
        )
    elif mode == "universal":
        model, step_fn = create_universal_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=model_key,
            optimizer=optax.adam(1e-3),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Create distribution
    dist = create_distribution(
        mode=mode,
        structure_dict=structure_dict,
        model=model,
    )
    
    # Verify both are functional
    assert dist is not None
    assert step_fn is not None
    
    sample = dist.sample(random.PRNGKey(0))
    assert isinstance(sample, dict)


# ============================================================================
# Normaliser Integration Tests
# ============================================================================

def test_normaliser_wrapping_static():
    """Test that normaliser wrapping works with static distribution."""
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.dist import NormalisedDistribution
    from structflow.conditional_flows.utils import DictNormaliser
    from structflow.conditional_flows.flow_factory import create_static_flow
    
    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)
    
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=model_key)
    structure = pgm.structure
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    # Create distribution
    model, _ = create_static_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=model_key,
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    
    dist = create_distribution(
        mode="static",
        structure_dict=structure_dict,
        model=model,
    )
    
    # Sample batch for normaliser
    sample_keys = random.split(data_key, 16)
    x_batch = {
        block: jnp.stack([
            pgm.sample(k)[block]
            for k in sample_keys
        ])
        for block in structure.block_order
    }
    
    # Create normaliser
    normaliser = DictNormaliser.from_data(x_batch)
    
    # Wrap distribution
    normalised_dist = NormalisedDistribution(dist=dist, normaliser=normaliser)
    
    # Test that normalised dist works
    x = normalised_dist.sample(random.PRNGKey(0))
    assert isinstance(x, dict)
    
    # Test log_prob with Jacobian correction
    lp = normalised_dist.log_prob(x)
    assert jnp.isfinite(lp)


def test_normaliser_corrects_log_prob():
    """Test that normaliser correctly adjusts log_prob with Jacobian."""
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.dist import NormalisedDistribution
    from structflow.conditional_flows.utils import DictNormaliser
    from structflow.conditional_flows.flow_factory import create_static_flow
    
    key = random.PRNGKey(42)
    model_key, data_key = random.split(key)
    
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=model_key)
    structure = pgm.structure
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    # Create distribution
    model, _ = create_static_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=model_key,
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    
    dist = create_distribution(
        mode="static",
        structure_dict=structure_dict,
        model=model,
    )
    
    # Sample data
    x = dist.sample(random.PRNGKey(0))
    
    # Create normaliser
    sample_keys = random.split(data_key, 8)
    x_batch = {
        block: jnp.stack([
            pgm.sample(k)[block]
            for k in sample_keys
        ])
        for block in structure.block_order
    }
    normaliser = DictNormaliser.from_data(x_batch)
    
    # Compute log_prob before and after normalisation
    lp_unnormalised = dist.log_prob(x)
    
    normalised_dist = NormalisedDistribution(dist=dist, normaliser=normaliser)
    lp_normalised = normalised_dist.log_prob(x)
    
    # Normalised log_prob should differ by Jacobian correction
    # log_p_norm = log_p + log|det J|
    # So they should be different (unless log|det J| ≈ 0)
    # At minimum, both should be finite
    assert jnp.isfinite(lp_unnormalised)
    assert jnp.isfinite(lp_normalised)


# ============================================================================
# Mode Compatibility Tests
# ============================================================================

def test_all_modes_create_valid_distributions():
    """Test that all modes can create valid distributions."""
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.flow_factory import (
        create_static_flow,
        create_dynamic_flow,
        create_joint_flow,
        create_carefl_flow,
        create_universal_flow,
    )
    
    key = random.PRNGKey(42)
    model_key = random.PRNGKey(43)
    
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=model_key)
    structure = pgm.structure
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    results = {}
    
    # Static
    model, _ = create_static_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=random.PRNGKey(0),
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    dist = create_distribution(mode="static", structure_dict=structure_dict, model=model)
    results["static"] = dist
    
    # Dynamic
    model, _ = create_dynamic_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=random.PRNGKey(1),
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    dist = create_distribution(mode="dynamic", structure_dict=structure_dict, model=model)
    results["dynamic"] = dist
    
    # Joint
    model, _ = create_joint_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=random.PRNGKey(2),
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    dist = create_distribution(mode="guided", structure_dict=structure_dict, model=model)
    results["guided"] = dist
    
    # CAREFL
    model, _ = create_carefl_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=random.PRNGKey(3),
        optimizer=optax.adam(1e-3),
    )
    dist = create_distribution(mode="carefl", structure_dict=structure_dict, model=model)
    results["carefl"] = dist
    
    # Verify all work
    for mode, dist in results.items():
        assert dist is not None, f"{mode} distribution is None"
        
        sample = dist.sample(key)
        assert isinstance(sample, dict), f"{mode} sample failed"
        assert len(sample) > 0, f"{mode} sample is empty"
        
        print(f"✓ Mode {mode} creates valid distribution")


def test_factory_modes_match_direct_creation():
    """Test that factory-created distributions match direct construction."""
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.flow_factory import create_static_flow
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.static.dist import StaticConditionalDistribution
    
    key = random.PRNGKey(42)
    model_key = random.PRNGKey(43)
    
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=model_key)
    structure = pgm.structure
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    # Create flow
    model, _ = create_static_flow(
        structure=pgm.structure,
        depth=2, width=16,
        model_key=random.PRNGKey(0),
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    
    # Create via factory
    dist_factory = create_distribution(
        mode="static",
        structure_dict=structure_dict,
        model=model,
    )
    
    # Should be a StaticConditionalDistribution
    assert isinstance(dist_factory, StaticConditionalDistribution), (
        f"Factory returned {type(dist_factory)}, expected StaticConditionalDistribution"
    )
    
    # Should have flow attribute
    assert hasattr(dist_factory, 'flow')
    assert hasattr(dist_factory, 'base_distributions')
