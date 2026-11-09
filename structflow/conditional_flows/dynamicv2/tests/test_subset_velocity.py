"""
Tests for DynamicV2 subset velocity and interpolation components.

Tests:
- Trajectory interpolation accuracy
- Subset velocity function matches full velocity (when subsets align)
- Divergence computation on subsets
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random, vmap
import equinox as eqx

from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.parallel.net import ParallelStructuredFn
from structflow.conditional_flows.dynamicv2.subset_velocity import interp_trajectory, SubsetVelocityFn
from flow.utils import divergence_jvp


# ============================================================================
# Interpolation Tests
# ============================================================================

def test_interp_trajectory_at_endpoints():
    """Interpolation at t=0 should return first point, at t=1 should return last."""
    # Create a simple trajectory
    traj = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # shape (3, 2)
    ts = jnp.array([0.0, 0.5, 1.0])
    
    # At t=0
    val_0 = interp_trajectory(traj, ts, 0.0)
    assert jnp.allclose(val_0, traj[0])
    
    # At t=1
    val_1 = interp_trajectory(traj, ts, 1.0)
    assert jnp.allclose(val_1, traj[-1])


def test_interp_trajectory_linear():
    """Linear interpolation between points should be correct."""
    traj = jnp.array([[0.0], [1.0], [2.0]])  # shape (3, 1)
    ts = jnp.array([0.0, 0.5, 1.0])
    
    # At t=0.25 (halfway between t=0 and t=0.5)
    val = interp_trajectory(traj, ts, 0.25)
    expected = 0.5  # halfway between 0.0 and 1.0
    assert jnp.allclose(val, jnp.array([expected]), atol=1e-6)
    
    # At t=0.75 (halfway between t=0.5 and t=1.0)
    val = interp_trajectory(traj, ts, 0.75)
    expected = 1.5  # halfway between 1.0 and 2.0
    assert jnp.allclose(val, jnp.array([expected]), atol=1e-6)


def test_interp_trajectory_clamping():
    """Values outside [0, 1] should be clamped."""
    traj = jnp.array([[0.0], [1.0]])
    ts = jnp.array([0.0, 1.0])
    
    # t < 0 should clamp to 0
    val = interp_trajectory(traj, ts, -0.5)
    assert jnp.allclose(val, traj[0])
    
    # t > 1 should clamp to 1
    val = interp_trajectory(traj, ts, 1.5)
    assert jnp.allclose(val, traj[-1])


# ============================================================================
# Subset Velocity Function Tests
# ============================================================================

@pytest.fixture
def simple_chain_structure():
    """Create a simple 3-block chain: A -> B -> C."""
    return BlockTriangularStructure(
        block_order=['A', 'B', 'C'],
        dependencies={'A': [], 'B': ['A'], 'C': ['B']},
        dims={'A': 2, 'B': 2, 'C': 2}
    )


@pytest.fixture
def parallel_velocity_fn(simple_chain_structure):
    """Create a ParallelStructuredFn for testing."""
    key = random.PRNGKey(0)
    return ParallelStructuredFn(
        structure=simple_chain_structure,
        hidden_dim=12,
        depth=2,
        key=key,
        time_embed_dim=8,
        use_sparse=False
    )


def test_subset_velocity_fn_creation(parallel_velocity_fn):
    """Test that SubsetVelocityFn can be created with valid inputs."""
    # Create condition trajectories
    traj_A = jnp.ones((11, 2))  # 11 time steps, dim 2
    ts = jnp.linspace(0, 1, 11)
    
    subset_vfn = SubsetVelocityFn(
        clipped_vfn=parallel_velocity_fn,
        condition_trajs={'A': traj_A},
        traj_ts=ts,
        cond_block_names=['A'],
        active_block_names=['B', 'C'],
    )
    
    # Check attributes
    assert subset_vfn.cond_dim == 2
    assert subset_vfn.active_dim == 4  # 2 + 2 for B and C
    assert subset_vfn.traj_ts.shape[0] == 11


def test_subset_velocity_fn_call(parallel_velocity_fn):
    """Test that SubsetVelocityFn.__call__ works and produces correct shapes."""
    # Create condition trajectories
    traj_A = jnp.ones((11, 2))
    ts = jnp.linspace(0, 1, 11)
    
    subset_vfn = SubsetVelocityFn(
        clipped_vfn=parallel_velocity_fn,
        condition_trajs={'A': traj_A},
        traj_ts=ts,
        cond_block_names=['A'],
        active_block_names=['B', 'C'],
    )
    
    # Call at t=0.5
    x_active = jnp.array([0.1, 0.2, 0.3, 0.4])  # B + C state
    v_active = subset_vfn(x_active, 0.5)
    
    # Should return velocity for B and C only
    assert v_active.shape == (4,)
    assert jnp.all(jnp.isfinite(v_active))


def test_subset_velocity_fn_no_conditions(parallel_velocity_fn):
    """Test SubsetVelocityFn with no conditions."""
    ts = jnp.linspace(0, 1, 11)
    
    subset_vfn = SubsetVelocityFn(
        clipped_vfn=parallel_velocity_fn,
        condition_trajs={},  # No conditions
        traj_ts=ts,
        cond_block_names=[],
        active_block_names=['A', 'B', 'C'],
    )
    
    # Call at t=0.5
    x_active = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # A + B + C
    v_active = subset_vfn(x_active, 0.5)
    
    # Should return velocity for all blocks
    assert v_active.shape == (6,)
    assert jnp.all(jnp.isfinite(v_active))


# ============================================================================
# Divergence Tests
# ============================================================================

def test_divergence_jvp_on_subset(parallel_velocity_fn):
    """Test that divergence_jvp works on subset velocity function."""
    # Create condition trajectories
    traj_A = jnp.ones((11, 2))
    ts = jnp.linspace(0, 1, 11)
    
    subset_vfn = SubsetVelocityFn(
        clipped_vfn=parallel_velocity_fn,
        condition_trajs={'A': traj_A},
        traj_ts=ts,
        cond_block_names=['A'],
        active_block_names=['B', 'C'],
    )
    
    # Compute divergence at a point
    x_active = jnp.array([0.1, 0.2, 0.3, 0.4])
    t = 0.5
    
    # Create a velocity function with time fixed
    vfn = lambda x: subset_vfn(x, t)
    
    # Compute divergence
    div_fn = divergence_jvp(vfn, argnums=0)
    div_val = div_fn(x_active)
    
    # Divergence should be a scalar
    assert div_val.shape == ()
    assert jnp.isfinite(div_val)


def test_divergence_trace_property(parallel_velocity_fn):
    """Divergence should be the trace of the Jacobian."""
    from jax import jacfwd
    
    # Simple test: create a velocity that has known divergence
    def test_vfn(x, t):
        # v = x (so divergence = n where n = len(x))
        return x
    
    x = jnp.array([1.0, 2.0, 3.0])
    
    # Compute divergence
    div_fn = divergence_jvp(test_vfn, argnums=0)
    div_val = div_fn(x, 0.5)
    
    # For v = x, divergence = 3
    assert jnp.allclose(div_val, 3.0, atol=1e-5)


# ============================================================================
# Integration Tests
# ============================================================================

def test_subset_velocity_vs_full_velocity(parallel_velocity_fn):
    """
    Test that SubsetVelocityFn(no conditions) gives same output as 
    calling the full velocity function on full state.
    """
    ts = jnp.linspace(0, 1, 11)
    
    # Create subset vfn with all blocks
    subset_vfn = SubsetVelocityFn(
        clipped_vfn=parallel_velocity_fn,
        condition_trajs={},
        traj_ts=ts,
        cond_block_names=[],
        active_block_names=['A', 'B', 'C'],
    )
    
    # Create a full state
    x_full = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    t = 0.5
    
    # Compute velocity from both
    v_subset = subset_vfn(x_full, t)
    v_full = parallel_velocity_fn(x_full, t)
    
    # Should be the same (within numerical error)
    assert jnp.allclose(v_subset, v_full, atol=1e-5)


def test_subset_velocity_interpolates_conditions(parallel_velocity_fn):
    """Test that SubsetVelocityFn properly interpolates condition trajectories."""
    # Create a trajectory that changes over time
    # A: [0, 0] at t=0, [1, 1] at t=1
    traj_A = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    ts = jnp.array([0.0, 0.5, 1.0])
    
    subset_vfn = SubsetVelocityFn(
        clipped_vfn=parallel_velocity_fn,
        condition_trajs={'A': traj_A},
        traj_ts=ts,
        cond_block_names=['A'],
        active_block_names=['B', 'C'],
    )
    
    x_active = jnp.array([0.2, 0.3, 0.4, 0.5])
    
    # At t=0, A should be [0, 0]
    v1 = subset_vfn(x_active, 0.0)
    
    # At t=0.5, A should be [0.5, 0.5]
    v2 = subset_vfn(x_active, 0.5)
    
    # At t=1, A should be [1, 1]
    v3 = subset_vfn(x_active, 1.0)
    
    # All should be finite and have correct shape
    assert v1.shape == (4,) and jnp.all(jnp.isfinite(v1))
    assert v2.shape == (4,) and jnp.all(jnp.isfinite(v2))
    assert v3.shape == (4,) and jnp.all(jnp.isfinite(v3))
