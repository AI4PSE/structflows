"""
Test that distributions maintain block-triangular structure via Jacobian.

Verifies:
- Velocity Jacobian is block-triangular (upper block part is zero)
- Forward map is structurally consistent with dependencies
- Jacobian structure matches expected sparsity pattern
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from structflow.conditional_flows.tests.test_utils import (
    assert_triangular_jacobian,
    flatten_dict_for_jacobian,
    unflatten_dict_from_vector,
    is_triangular_mode,
)
from structflow.conditional_flows.tests.conftest import MODES, TRIANGULAR_MODES

# ============================================================================
# Skip configuration for non-triangular modes
# ============================================================================

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def skip_non_triangular_mode(mode):
    """Skip test if mode doesn't support triangularity check."""
    if not is_triangular_mode(mode):
        pytest.skip(f"Mode {mode} does not have explicit Jacobian triangularity")


# ============================================================================
# Jacobian Structure Tests
# ============================================================================

@pytest.mark.parametrize("mode", TRIANGULAR_MODES)
def test_velocity_jacobian_is_triangular(mode, structure_type, skip_non_triangular_mode):
    """
    Test that velocity function has block-triangular Jacobian.
    
    For each output block, it should only depend on itself and its parents,
    not on future blocks (upper triangular should be zero).
    """
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.dist_factory import create_distribution
    from structflow.conditional_flows.flow_factory import (
        create_static_flow, create_dynamic_flow, create_carefl_flow
    )
    import optax
    
    # Create simple PGM
    key = random.PRNGKey(42)
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=key)
    structure = pgm.structure
    
    # Create model based on mode
    if mode == "static":
        model, _ = create_static_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        # Extract velocity function
        velocity_fn = model.net
    elif mode == "dynamic":
        model, _ = create_dynamic_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        velocity_fn = model.net
    elif mode == "carefl":
        # CAREFL doesn't have a velocity function; it's affine
        # Skip this test for CAREFL
        pytest.skip("CAREFL is affine, not velocity-based")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Create test input
    x_test = {block: jnp.ones((structure.dims[block],))
              for block in structure.block_order}
    t_test = jnp.array(0.5)
    
    # Compute Jacobian using jacfwd
    if mode == "static":
        # Static mode: velocity_fn takes dict and accepts active_blocks
        def velocity_flat(x_flat):
            x_dict = unflatten_dict_from_vector(x_flat, structure.block_order, structure.dims)
            v_dict = velocity_fn(x_dict, t_test, active_blocks=structure.block_order)
            return flatten_dict_for_jacobian(v_dict, structure.block_order)
    else:  # dynamic
        # Dynamic mode: velocity_fn takes flat vector
        def velocity_flat(x_flat):
            return velocity_fn(x_flat, t_test)
    
    x_flat_test = flatten_dict_for_jacobian(x_test, structure.block_order)
    jac = jax.jacfwd(velocity_flat)(x_flat_test)
    
    # Verify triangularity
    assert_triangular_jacobian(
        jac, 
        structure.block_order,
        structure.dependencies,
        structure.dims,
        atol=1e-5
    )


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_jacobian_respects_dependencies(mode, structure_type):
    """
    Test that Jacobian zero entries match dependency structure.
    
    If block B has no dependency on block A, then Jacobian[B, A] should be zero.
    """
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.flow_factory import create_static_flow, create_dynamic_flow
    import optax
    
    key = random.PRNGKey(42)
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=key)
    structure = pgm.structure
    
    if mode == "static":
        model, _ = create_static_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        velocity_fn = model.net
    else:  # dynamic
        model, _ = create_dynamic_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        velocity_fn = model.net
    
    x_test = {block: jnp.ones((structure.dims[block],))
              for block in structure.block_order}
    t_test = jnp.array(0.5)
    
    if mode == "static":
        def velocity_flat(x_flat):
            x_dict = unflatten_dict_from_vector(x_flat, structure.block_order, structure.dims)
            v_dict = velocity_fn(x_dict, t_test, active_blocks=structure.block_order)
            return flatten_dict_for_jacobian(v_dict, structure.block_order)
    else:  # dynamic
        def velocity_flat(x_flat):
            return velocity_fn(x_flat, t_test)
    
    x_flat_test = flatten_dict_for_jacobian(x_test, structure.block_order)
    jac = jax.jacfwd(velocity_flat)(x_flat_test)
    
    # Check that non-dependencies are zero in Jacobian
    # If block_j is not in dependencies[block_i], then Jac[i, j] should be zero
    from structflow.conditional_flows.tests.test_utils import get_block_slice
    for i, block_i in enumerate(structure.block_order):
        for j, block_j in enumerate(structure.block_order):
            if block_j not in structure.dependencies.get(block_i, []) and block_i != block_j:
                # Should be zero (except diagonal and dependencies)
                slice_i = get_block_slice(block_i, structure.block_order, structure.dims)
                slice_j = get_block_slice(block_j, structure.block_order, structure.dims)
                sub_jac = jac[slice_i, slice_j]
                
                if j > i:  # Future blocks must be zero
                    assert jnp.allclose(sub_jac, 0, atol=1e-5), (
                        f"Expected zero Jacobian for {block_i} <- {block_j} "
                        f"(future block), but got max |Jac| = {jnp.max(jnp.abs(sub_jac))}"
                    )


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_chain_jacobian_chain_rule(mode):
    """
    Test that chain structure Jacobian follows the dependency chain.
    
    For chain x0 -> x1:
    - v1 should depend on v1 itself (diagonal)
    - v1 should depend on x0 (parent)
    - v0 should NOT depend on x1 (future)
    """
    
    from structflow.pgms.factories import make_chain_pgm
    from structflow.conditional_flows.flow_factory import create_static_flow, create_dynamic_flow
    import optax
    from structflow.conditional_flows.tests.test_utils import get_block_slice
    
    key = random.PRNGKey(42)
    pgm = make_chain_pgm(n_nodes=2, mechanism_type="linear", key=key)
    structure = pgm.structure
    
    if mode == "static":
        model, _ = create_static_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        velocity_fn = model.net
    else:  # dynamic
        model, _ = create_dynamic_flow(
            structure=pgm.structure,
            depth=2, width=16,
            model_key=key,
            optimizer=optax.adam(1e-3),
            use_coupling=False,
        )
        velocity_fn = model.net
    
    x_test = {block: jnp.ones((structure.dims[block],))
              for block in structure.block_order}
    t_test = jnp.array(0.5)
    
    if mode == "static":
        def velocity_flat(x_flat):
            x_dict = unflatten_dict_from_vector(x_flat, structure.block_order, structure.dims)
            v_dict = velocity_fn(x_dict, t_test, active_blocks=structure.block_order)
            return flatten_dict_for_jacobian(v_dict, structure.block_order)
    else:  # dynamic
        def velocity_flat(x_flat):
            return velocity_fn(x_flat, t_test)
    
    x_flat_test = flatten_dict_for_jacobian(x_test, structure.block_order)
    jac = jax.jacfwd(velocity_flat)(x_flat_test)
    
    # For chain: x0 -> x1
    # Jacobian[v_x1, x0] should be non-zero (dependency)
    slice_x0 = get_block_slice('x0', structure.block_order, structure.dims)
    slice_x1 = get_block_slice('x1', structure.block_order, structure.dims)
    
    # v_x1 depends on x0 (parent)
    j_v1_x0 = jac[slice_x1, slice_x0]
    assert not jnp.allclose(j_v1_x0, 0, atol=1e-5), (
        f"Expected non-zero Jacobian for v_x1 wrt x0 (parent dependency)"
    )
    
    # v_x0 should NOT depend on x1 (future)
    j_v0_x1 = jac[slice_x0, slice_x1]
    assert jnp.allclose(j_v0_x1, 0, atol=1e-5), (
        f"Expected zero Jacobian for v_x0 wrt x1 (future block), "
        f"but got max |Jac| = {jnp.max(jnp.abs(j_v0_x1))}"
    )
