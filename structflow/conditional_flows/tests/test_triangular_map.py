
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from functools import partial

from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.static.net import StructuredVelocityFn, ConditionalBlock
from sinterp.stochasticfield import OneSidedField_eta1
from sinterp.interpolants import OneSidedLinear
from structflow.conditional_flows.dist_factory import StructuredFieldAdapter

def test_velocity_is_block_triangular():
    """
    Verifies that the StructuredFieldAdapter (converting eta1-prediction to velocity)
    preserves the block-triangular structure.
    """
    # 1. Setup Structure: A -> B -> C
    # Dim: A=1, B=2, C=1. Total = 4.
    dims = {'A': 1, 'B': 2, 'C': 1}
    block_order = ['A', 'B', 'C']
    dependencies = {'A': [], 'B': ['A'], 'C': ['A', 'B']}
    structure = BlockTriangularStructure(block_order, dependencies, dims)
    
    # 2. Build Model matching the StructuredFieldAdapter requirements
    key = jax.random.PRNGKey(42)
    block_key, _ = jax.random.split(key)
    
    # Use a small network
    block_cons = partial(ConditionalBlock, width=16, depth=2, embedding_dim=8)
    
    velocity_net = StructuredVelocityFn(
        structure=structure,
        key=block_key,
        block_constructor=block_cons
    )
    
    interpolator = OneSidedLinear()
    field = OneSidedField_eta1(interpolator, velocity_net)
    adapter = StructuredFieldAdapter(field)
    
    # 3. Define flattened wrapper for Jacobian calculation
    indices = {}
    curr = 0
    for name in block_order:
        indices[name] = slice(curr, curr + dims[name])
        curr += dims[name]
    total_dim = curr

    def flattened_velocity(x_vec, t):
        # Unflatten vector to dict
        x_dict = {name: x_vec[indices[name]] for name in block_order}
        # Compute velocity (joint mode)
        v_dict = adapter(x_dict, t)
        # Flatten dict to vector - use ravel to handle potential (1, dim) or (dim,)
        return jnp.concatenate([v_dict[name].ravel() for name in block_order])

    # 4. Compute Jacobian at a random point
    t_test = jnp.array([0.5])
    jax_rng = jax.random.PRNGKey(0)
    x_test = jax.random.normal(jax_rng, (total_dim,))
    
    # Use jacfwd for the Jacobian
    jac_fn = jax.jacfwd(flattened_velocity, argnums=0)
    J = jac_fn(x_test, t_test)
    
    # 5. Verify Block-Triangular Structure
    # Map: block_i depends on block_j if j <= i and j in parents(i) or j == i.
    # Jacobian J[out_idx, in_idx] should be 0 if block(in_idx) > block(out_idx)
    
    for i_block_idx, out_name in enumerate(block_order):
        for j_block_idx, in_name in enumerate(block_order):
            sub_jac = J[indices[out_name], indices[in_name]]
            
            if j_block_idx > i_block_idx:
                # Strictly upper block-triangular part -> must be ZERO
                # (e.g., A depends on B? No.)
                assert jnp.allclose(sub_jac, 0.0, atol=1e-6), \
                    f"Violation: Block {out_name} (idx {i_block_idx}) has non-zero gradient " \
                    f"wrt distal Block {in_name} (idx {j_block_idx})."
            else:
                # Lower triangular or diagonal part.
                # Check if dependencies are alive if they exist.
                # A depends on [], B depends on [A], C depends on [A, B]
                is_diag = (i_block_idx == j_block_idx)
                is_dependency = (in_name in dependencies.get(out_name, []))
                
                if is_diag or is_dependency:
                    # Should generally be non-zero for a random network
                    norm = jnp.linalg.norm(sub_jac)
                    assert norm > 1e-7, \
                        f"Warning: Expected non-zero Jacobian for {out_name} wrt {in_name}, but got norm {norm}"

    print("Success: Velocity function is block-triangular!")

if __name__ == "__main__":
    test_velocity_is_block_triangular()
