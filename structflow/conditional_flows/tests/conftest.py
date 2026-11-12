"""
Shared pytest fixtures and parametrization for conditional flows tests.

Provides:
- Mode parametrization (static, dynamic, joint, carefl)
- Structure fixtures (chain, fork, collider)
- PGM and distribution creation fixtures
- Mode-specific behavior handling
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

from structflow.pgms.factories import make_chain_pgm, make_fork_pgm, make_collider_pgm
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.dist_factory import create_distribution
from structflow.conditional_flows.flow_factory import (
    create_static_flow,
    create_dynamic_flow,
    create_joint_flow,
    create_carefl_flow,
    create_universal_flow,
)
from structflow.bayes.distribution import Gaussian

N_STEPS=10

# ============================================================================
# Constants and Parametrization
# ============================================================================

MODES = ["static", "dynamic", "guided", "carefl", "universal"]
# MODES = ["carefl", "universal", "dynamic"]
STRUCTURE_TYPES = ["chain", "fork", "collider"]

# Modes that support Jacobian triangularity testing
TRIANGULAR_MODES = ["static", "dynamic", "carefl", "universal"]

# Modes that are flow-based (support zero velocity test)
FLOW_BASED_MODES = ["static", "dynamic", "guided"]


# ============================================================================
# Parametrize Fixtures
# ============================================================================

@pytest.fixture(params=MODES)
def mode(request):
    """Parametrize all tests across distribution modes."""
    return request.param


@pytest.fixture(params=STRUCTURE_TYPES)
def structure_type(request):
    """Parametrize across different DAG structures."""
    return request.param


# ============================================================================
# Structure Fixtures
# ============================================================================

@pytest.fixture
def chain_structure():
    """Simple 2-node chain: x0 -> x1"""
    return BlockTriangularStructure(
        block_order=['x0', 'x1'],
        dependencies={'x0': [], 'x1': ['x0']},
        dims={'x0': 1, 'x1': 1}
    )


@pytest.fixture
def fork_structure():
    """Fork structure: z -> x1, z -> x2"""
    return BlockTriangularStructure(
        block_order=['z', 'x1', 'x2'],
        dependencies={'z': [], 'x1': ['z'], 'x2': ['z']},
        dims={'z': 1, 'x1': 1, 'x2': 1}
    )


@pytest.fixture
def collider_structure():
    """Collider structure: x1 -> y, x2 -> y"""
    return BlockTriangularStructure(
        block_order=['x1', 'x2', 'y'],
        dependencies={'x1': [], 'x2': [], 'y': ['x1', 'x2']},
        dims={'x1': 1, 'x2': 1, 'y': 1}
    )


@pytest.fixture
def structure(structure_type):
    """Return appropriate structure based on parametrization."""
    if structure_type == "chain":
        return BlockTriangularStructure(
            block_order=['x0', 'x1'],
            dependencies={'x0': [], 'x1': ['x0']},
            dims={'x0': 1, 'x1': 1}
        )
    elif structure_type == "fork":
        return BlockTriangularStructure(
            block_order=['z', 'x1', 'x2'],
            dependencies={'z': [], 'x1': ['z'], 'x2': ['z']},
            dims={'z': 1, 'x1': 1, 'x2': 1}
        )
    elif structure_type == "collider":
        return BlockTriangularStructure(
            block_order=['x1', 'x2', 'y'],
            dependencies={'x1': [], 'x2': [], 'y': ['x1', 'x2']},
            dims={'x1': 1, 'x2': 1, 'y': 1}
        )
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")


# ============================================================================
# PGM Fixtures
# ============================================================================

@pytest.fixture
def pgm_key():
    """Random key for PGM creation."""
    return random.PRNGKey(42)


@pytest.fixture
def simple_pgm(pgm_key):
    """Create a simple 2-node linear chain PGM for learning tests."""
    return make_chain_pgm(
        n_nodes=2,
        mechanism_type="linear",
        key=pgm_key,
        node_dim=1
    )


# ============================================================================
# Distribution/Flow Creation Fixtures
# ============================================================================

@pytest.fixture
def dist(mode, simple_pgm, pgm_key):
    """
    Create appropriate distribution for the given mode.
    
    Uses simple 2-node chain PGM.
    """
    pgm = simple_pgm
    structure = pgm.structure
    
    # Convert structure to dict for dist_factory
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims
    }
    
    if mode == "static":
        model, _ = create_static_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        return create_distribution(
            mode="static",
            structure_dict=structure_dict,
            model=model,
            n_steps=N_STEPS,
        )
    
    elif mode == "dynamic":
        model, _ = create_dynamic_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        return create_distribution(
            mode="dynamic",
            structure_dict=structure_dict,
            model=model,
            n_steps=N_STEPS,
        )
    
    elif mode == "guided":
        model, _ = create_joint_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        return create_distribution(
            mode="guided",
            structure_dict=structure_dict,
            model=model,
            n_steps=N_STEPS,
        )
    
    elif mode == "carefl":
        model, _ = create_carefl_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
        )
        return create_distribution(
            mode="carefl",
            structure_dict=structure_dict,
            model=model,
            n_steps=N_STEPS,
        )
    
    elif mode == "universal":
        model, _ = create_universal_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
        )
        return create_distribution(
            mode="universal",
            structure_dict=structure_dict,
            model=model,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


@pytest.fixture
def flow_and_step_fn(mode, simple_pgm, pgm_key):
    """
    Create flow model and training step function for the given mode.
    
    Returns: (model, step_fn, structure)
    """
    pgm = simple_pgm
    
    if mode == "static":
        model, step_fn = create_static_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    
    elif mode == "dynamic":
        model, step_fn = create_dynamic_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    
    elif mode == "guided":
        model, step_fn = create_joint_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
    
    elif mode == "carefl":
        model, step_fn = create_carefl_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
        )
    
    elif mode == "universal":
        model, step_fn = create_universal_flow(
            structure=pgm.structure,
            depth=2,
            width=16,
            model_key=pgm_key,
            optimizer=optax.adam(1e-3),
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return model, step_fn, pgm.structure


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_data_small(simple_pgm, pgm_key):
    """Sample a small batch (4 samples) from the PGM."""
    n_samples = 4
    samples = simple_pgm.b_sample(pgm_key, n_samples)
    return {k: v for k, v in samples.items()}


@pytest.fixture
def sample_data_batch(simple_pgm, pgm_key):
    """Sample a larger batch (32 samples) from the PGM."""
    n_samples = 32
    samples = simple_pgm.b_sample(pgm_key, n_samples)
    return {k: v for k, v in samples.items()}
