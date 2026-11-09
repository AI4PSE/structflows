import jax
from jax import vmap
import jax.numpy as jnp
import ott.geometry.pointcloud
import ott.solvers.linear
# import arviz as az
import numpy as np
from structflow.bayes.distribution import Distribution
import equinox as eqx
from typing import Dict, Set, Union, List, Optional
from jax import random

@eqx.filter_jit
def compute_swd_jax(key, X, Y, num_projections=200):
    """
    Computes Sliced Wasserstein Distance (SWD) between two sets of samples.

    SWD projects multidimensional samples onto random directions and computes
    the 1D Wasserstein distance (L2 distance between sorted projections).

    Args:
        key: JAX PRNGKey.
        X: Array of shape (N, D), first sample set.
        Y: Array of shape (M, D), second sample set.
        num_projections: Number of random directions to project onto.

    Returns:
        The Sliced Wasserstein Distance (scalar), normalized by sample size and dimensionality.
    """

    # 1. Generate random directions on the sphere
    d = X.shape[1]
    N = X.shape[0]
    M = Y.shape[0]
    
    if d != 1:
        dirs = jax.random.normal(key, (d, num_projections))
        dirs = dirs / jnp.linalg.norm(dirs, axis=0) # Normalize to unit sphere
        
        # 2. Project samples onto directions
        # X_proj shape: (N, num_projections)
        X_proj = X @ dirs
        Y_proj = Y @ dirs
    else:
        X_proj = X
        Y_proj = Y

    num_projections = min(num_projections, X.shape[0], Y.shape[0])
    
    # 3. Sort projections (Wasserstein in 1D is simple L2 between sorted lists)
    X_sorted = jnp.sort(X_proj, axis=0)
    Y_sorted = jnp.sort(Y_proj, axis=0)
    
    # 4. Compute average squared distance (Wasserstein-2)
    swd = jnp.mean((X_sorted - Y_sorted)**2)
    swd = jnp.sqrt(swd)
    
    # 5. Normalize by sample size and dimensionality for comparability across different data sizes
    # Divide by sqrt(max(N,M)) to normalize for sample size
    # Divide by sqrt(D) to normalize for dimensionality
    normalization_factor = jnp.sqrt(jnp.maximum(N, M) * jnp.float32(d))
    return swd / normalization_factor

@eqx.filter_jit
def compute_sinkhorn_jax(X, Y, epsilon=0.1):
    """
    Computes the Entropic Optimal Transport (Sinkhorn) cost using ott-jax.

    Args:
        X: Array of shape (N, D), first sample set.
        Y: Array of shape (M, D), second sample set.
        epsilon: Entropic regularization parameter.

    Returns:
        The OT cost (scalar), normalized by sample size and dimensionality.
    """
    d = X.shape[1]
    N = X.shape[0]
    M = Y.shape[0]
    geom = ott.geometry.pointcloud.PointCloud(X, Y, epsilon=epsilon)
    out = ott.solvers.linear.solve(geom)
    
    # Standardize to Root Mean Squared Distance
    rms_dist = jnp.sqrt(out.primal_cost / N)
    
    # Apply same normalization as SWD
    normalization_factor = jnp.sqrt(jnp.maximum(N, M) * jnp.float32(d))
    return rms_dist / normalization_factor

@eqx.filter_jit
def compute_kl_jax(p: Distribution, q: Distribution, key, num_samples=2000):
    """
    Estimates the KL divergence KL(p || q) using Monte Carlo sampling.
    
    KL(p || q) = E_p [ log p(x) - log q(x) ]

    Args:
        p: Reference distribution.
        q: Approximate distribution.
        key: JAX PRNGKey.
        num_samples: Number of samples for Monte Carlo estimation.

    Returns:
        The estimated KL divergence (scalar).
    """
    samples = p.b_sample(key, num_samples)
    log_p = p.b_log_prob(samples)
    log_q = q.b_log_prob(samples)
    return jnp.mean(log_p - log_q)

@eqx.filter_jit
def _flatten_samples(samples):
    """
    Flattens a pytree of samples into a single (N, D) array.

    Args:
        samples: A pytree of arrays where the first dimension is the batch size N.

    Returns:
        An array of shape (N, D) where D is the total number of flattened elements per sample.
    """
    if isinstance(samples, jnp.ndarray):
        return samples.reshape(samples.shape[0], -1)
    
    flat_leaves, _ = jax.tree_util.tree_flatten(samples)
    reshaped_leaves = [leaf.reshape(leaf.shape[0], -1) for leaf in flat_leaves]
    return jnp.concatenate(reshaped_leaves, axis=1)

@eqx.filter_jit
def _compare_jit(p, q, k1, k2, k3, num_samples, num_projections):
    """JIT'd core of distribution comparison to minimize host-device syncs."""
    samples_p = p.b_sample(k1, num_samples)
    samples_q = q.b_sample(k2, num_samples)
    
    X = _flatten_samples(samples_p)
    Y = _flatten_samples(samples_q)
    
    swd = compute_swd_jax(k3, X, Y, num_projections=num_projections)
    
    try:
        log_p_on_p = p.b_log_prob(samples_p)
        log_q_on_p = q.b_log_prob(samples_p)
        log_p_on_q = p.b_log_prob(samples_q)
        log_q_on_q = q.b_log_prob(samples_q)
        
        test_nll = -jnp.mean(log_q_on_p)
        kl = jnp.mean(log_p_on_p - log_q_on_p)
        log_weights = log_p_on_q - log_q_on_q
        
        return swd, test_nll, kl, log_weights, False
    except NotImplementedError:
        return swd, jnp.inf, jnp.inf, jnp.zeros(num_samples), True

def compare_distributions(p: Distribution, q: Distribution, key, num_samples=1000, num_projections=100):
    """
    Compares two distributions using a variety of metrics.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    swd, test_nll, kl, log_weights, not_implemented = _compare_jit(
        p, q, k1, k2, k3, num_samples, num_projections
    )
    
    results = {
        "SWD": float(swd),
        "Test_NLL": float(test_nll),
        "KL": float(kl),
        # "k_hat": jnp.inf
    }

    # if not not_implemented:
    #     log_weights_np = np.asarray(log_weights).flatten()
    #     if not np.any(np.isnan(log_weights_np)):
    #         # az.psislw is usually the bottleneck for host-device sync if called frequently
    #         k_hat_val = az.psislw(log_weights_np[None, :])[0]
    #         results["k_hat"] = float(np.asarray(k_hat_val).flatten()[0])

    return results

def _compute_stiffness(velocity_fn, x, t, argnums=0):
    """
    Computes the Frobenius norm of the Jacobian of the velocity function at (x, t).
    
    Args:
        velocity_fn: The velocity function (StructuredVelocityFn).
        x: Current state (Dict[str, jnp.ndarray]).
        t: Current time (float).
        
    Returns:
        The Frobenius norm of the Jacobian (scalar).
    """
    # We want the Jacobian of the whole velocity vector with respect to the whole state vector.
    # Since x and v are dicts, we can flatten them to compute the Jacobian and then its norm.
    
    def flat_v(x_flat):
        # Unflatten x
        x_dict = jax.tree_util.tree_unflatten(treedef, x_flat)
        # Compute v
        v_dict = velocity_fn(x_dict, t)
        # Flatten v
        v_flat, _ = jax.tree_util.tree_flatten(v_dict)
        return jnp.concatenate([l.ravel() for l in v_flat])

    # Flatten x to get starting point
    x_leaves, treedef = jax.tree_util.tree_flatten(x)
    x_flat = jnp.concatenate([l.ravel() for l in x_leaves])
    
    # Compute Jacobian
    jac = jax.jacfwd(flat_v, argnums=argnums)(x_flat)
    
    # Frobenius norm is sqrt of sum of squares of all elements
    return jnp.linalg.norm(jac, ord='fro')

@eqx.filter_jit
def _analyze_flow_jit(q, key, num_samples):
    """JIT'd core of flow analysis."""
    # 1. Sample base noise x0
    # We assume q has base_distributions and flow attributes
    all_blocks = q.flow.velocity_fn.structure.block_order
    
    # Sample from all base distributions
    keys = jax.random.split(key, len(all_blocks) + 1)
    x0_keys = jax.random.split(keys[0], len(all_blocks))
    x0 = {}
    for i, block in enumerate(all_blocks):
        # Each block might have its own sample method
        x0[block] = vmap(q.base_distributions[block].sample)(jax.random.split(x0_keys[i], num_samples))
        
    # 2. Integrate Forward to get trajectories
    # We use a simple vmap over the trajector_forward call to be robust 
    # to different flow implementations (static vs dynamic).
    def get_traj(x0_sample):
        # We pass only x0 and active_blocks. 
        # For joint analysis, conditions/condition_trajs default to None.
        return q.flow.trajectory_forward(x0_sample, all_blocks)

    trajs = vmap(get_traj)(x0)
    # trajs is a dict where each value is (num_samples, n_steps + 1, dim)
    
    # 3. Compute velocities and stiffness over time
    n_steps = q.flow.n_steps
    ts = jnp.linspace(0, 1, n_steps + 1)
    
    def compute_metrics_at_t(i):
        t = ts[i]
        # state_at_t is (num_samples, PyTree)
        state_at_t = {k: trajs[k][:, i, ...] for k in all_blocks}
        
        # Velocity at t for all samples
        # v_at_t is (num_samples, PyTree of velocities)
        v_at_t = vmap(lambda x: q.flow.velocity_fn(x, t))(state_at_t)
        
        # Flatten velocities to compute magnitudes
        # We use a helper that handles PyTrees consistently
        def get_mag(v_dict):
            leaves, _ = jax.tree_util.tree_flatten(v_dict)
            flat = jnp.concatenate([l.ravel() for l in leaves])
            return jnp.linalg.norm(flat)

        v_mags = vmap(get_mag)(v_at_t)
        avg_vel = jnp.mean(v_mags)
        
        # Stiffness at t
        stiffness_vals = vmap(lambda x: _compute_stiffness(q.flow.velocity_fn, x, t))(state_at_t)
        avg_stiffness = jnp.mean(stiffness_vals)
        
        return avg_vel, avg_stiffness

    avg_vel_list, avg_stiffness_list = jax.lax.map(compute_metrics_at_t, jnp.arange(n_steps + 1))
    
    # 4. Path Straightness
    # (Distance between endpoints) / (Total path length)
    # Endpoints
    x0_leaves, _ = jax.tree_util.tree_flatten(x0)
    x0_flat = jnp.concatenate([l.reshape(num_samples, -1) for l in x0_leaves], axis=1)
    
    x1 = {k: trajs[k][:, -1, ...] for k in all_blocks}
    x1_leaves, _ = jax.tree_util.tree_flatten(x1)
    x1_flat = jnp.concatenate([l.reshape(num_samples, -1) for l in x1_leaves], axis=1)
    
    displacement = jnp.linalg.norm(x1_flat - x0_flat, axis=1)
    
    # Path length approximation: sum of step distances
    # trajs_flat: (num_samples, n_steps + 1, total_dim)
    trajs_leaves, _ = jax.tree_util.tree_flatten(trajs)
    trajs_flat = jnp.concatenate([l.reshape(num_samples, n_steps + 1, -1) for l in trajs_leaves], axis=2)
    
    step_diffs = jnp.diff(trajs_flat, axis=1)
    step_lengths = jnp.linalg.norm(step_diffs, axis=2)
    path_lengths = jnp.sum(step_lengths, axis=1)
    
    # Avoid div by zero
    straightness = displacement / (path_lengths + 1e-8)
    avg_straightness = jnp.mean(straightness)
    
    return {
        "avg_velocity": avg_vel_list,
        "avg_stiffness": avg_stiffness_list,
        "path_straightness": avg_straightness,
        "ts": ts
    }

def analyze_flow_dynamics(q: Distribution, key, num_samples=1000):
    """
    Computes diagnostics for a flow-based distribution.
    
    Args:
        q: A flow-based distribution (e.g. BlockTriangularDistribution).
        key: JAX PRNGKey.
        num_samples: Number of samples for estimation.
        
    Returns:
        Dictionary of metrics.
    """
    if not hasattr(q, "flow"):
        raise ValueError("analyze_flow_dynamics expects a distribution with a .flow attribute.")
    
    # Check if we should unwrap NormalisedDistribution
    if hasattr(q, "dist"):
         # Recursively handle nested distributions (like NormalisedDistribution)
         # However, we might want the metrics in the original space?
         # User's request is aimed at properties of q (the flow). 
         # The inner flow has the velocities.
         return analyze_flow_dynamics(q.dist, key, num_samples)

    return _analyze_flow_jit(q, key, num_samples)

@eqx.filter_jit
def compute_counterfactual(p: Distribution, q: Distribution, key, factual_data: Dict[str, jnp.ndarray], targets):
    """
    Computes MSE between ground truth (p) and model (q) counterfactuals.
    
    Args:
        p: Ground truth distribution (e.g., FrozenConditional).
        q: Approximate distribution (e.g., FrozenConditional).
        factual_data: Batched factual observations used for abduction.
        
    Returns:
        Dictionary containing the Counterfactual_MSE.
    """
    # 1. Compute Counterfactuals
    # Both p and q are assumed to be FrozenConditionals or similar,
    # where the interventions are already baked-in.
    try:
        gt_cf = _flatten_samples(p.b_counterfactual(factual_data))
        model_cf = _flatten_samples(q.b_counterfactual(factual_data))
        
        swd = compute_swd_jax(key, gt_cf, model_cf, num_projections=100)
        return {"Counterfactual_SWD": swd}
    except NotImplementedError:
        return {"Counterfactual_SWD": jnp.nan}
        
    # 2. Compute MSE across all target variables
    # mse = 0.0
    # for k in targets:
    #     mse += jnp.mean((gt_cf[k] - model_cf[k])**2)
    # return {"Counterfactual_MSE": mse}
