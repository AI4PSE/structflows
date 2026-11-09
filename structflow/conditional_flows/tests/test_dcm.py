"""Tests for the DCM (per-node diffusion) backbone.

Covers:
- DDIM round-trip (invert then forward) error decreasing at the expected
  first-order rate in n_steps — a *positive* test of the mechanism the
  backbone ablation claims (DDIM inversion carries an O(1/N) bias).
- ConditionalDistribution API contract: sampling, transform_to_noise,
  counterfactual through the shared CounterfactualMixin.
- log_prob raises NotImplementedError eagerly (capability gap must surface).
- A small end-to-end learning check of the ε-MSE objective.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random, vmap
import optax

from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.dcm.net import DiffusionBlock
from structflow.conditional_flows.dcm.sampler import ddim_forward, ddim_invert
from structflow.conditional_flows.dcm.schedule import alpha_bar
from structflow.conditional_flows.flow_factory import create_dcm_flow
from structflow.conditional_flows.dist_factory import create_distribution, create_dcm_distribution


def _chain_structure(dim=1):
    return BlockTriangularStructure(
        block_order=['x0', 'x1'],
        dependencies={'x0': [], 'x1': ['x0']},
        dims={'x0': dim, 'x1': dim},
    )


def _make_dcm_dist(structure, seed=0, n_steps=32, width=16, depth=2):
    model, _ = create_dcm_flow(
        structure=structure,
        depth=depth,
        width=width,
        model_key=random.PRNGKey(seed),
        optimizer=optax.adam(1e-3),
    )
    return create_dcm_distribution(model, structure, n_steps=n_steps)


# ============================================================================
# Schedule sanity
# ============================================================================

def test_schedule_endpoints():
    """t=1 (data) must be exact; t=0 (noise) close to pure noise but clipped
    away from zero so inversion stays numerically stable."""
    assert float(alpha_bar(jnp.array(1.0))) == pytest.approx(1.0)
    ab0 = float(alpha_bar(jnp.array(0.0)))
    assert 0.0 < ab0 <= 1e-3
    ts = jnp.linspace(0.0, 1.0, 101)
    abs_ = np.asarray(alpha_bar(ts))
    assert np.all(np.diff(abs_) >= -1e-7), "alpha_bar must be monotone in t"


# ============================================================================
# DDIM round-trip: first-order convergence in n_steps
# ============================================================================

def test_ddim_roundtrip_first_order_convergence():
    """‖forward(invert(x, N), N) − x‖ (the abduction reconstruction error)
    must decrease with N at the first-order rate: the published DDIM inversion
    re-uses ε̂ at the current point, giving an O(1/N) global bias."""
    key = random.PRNGKey(0)
    dim, parent_dim = 2, 2
    block = DiffusionBlock(dim, [parent_dim], key, width=32, depth=3)

    data_key, parent_key = random.split(random.PRNGKey(1))
    xs = random.normal(data_key, (16, dim))
    parents = random.normal(parent_key, (16, parent_dim))

    def roundtrip_err(n_steps):
        def one(x, pa):
            u = ddim_invert(block, x, (pa,), n_steps)
            x_rec = ddim_forward(block, u, (pa,), n_steps)
            return jnp.sqrt(jnp.sum((x_rec - x) ** 2))
        return float(jnp.mean(vmap(one)(xs, parents)))

    n_grid = [4, 8, 16, 32, 64, 128]
    errs = [roundtrip_err(n) for n in n_grid]

    # Errors must decrease monotonically (strictly, up to float noise).
    for a, b in zip(errs, errs[1:]):
        assert b < a * 1.05, f"round-trip error not decreasing: {errs}"

    # Convergence order: slope of log(err) vs log(N) over the resolved range.
    # First-order (O(1/N)) means slope ≈ -1.
    log_n = np.log(np.asarray(n_grid, dtype=float))
    log_e = np.log(np.asarray(errs))
    slope = np.polyfit(log_n, log_e, 1)[0]
    assert -1.6 < slope < -0.6, (
        f"expected ~first-order convergence (slope ≈ -1), got slope={slope:.3f}, "
        f"errors={errs}")


def test_ddim_invert_is_not_exact_inverse_at_low_nfe():
    """The inversion bias must be *visible* at low NFE (it is the object of
    study in the ablation): a coarse chain reconstructs measurably worse than
    a fine one."""
    key = random.PRNGKey(2)
    block = DiffusionBlock(2, [], key, width=32, depth=3)
    x = jnp.array([0.7, -1.2])

    def err(n):
        u = ddim_invert(block, x, (), n)
        return float(jnp.sqrt(jnp.sum((ddim_forward(block, u, (), n) - x) ** 2)))

    assert err(4) > 5.0 * err(128)


# ============================================================================
# ConditionalDistribution API
# ============================================================================

def test_dcm_sample_shapes_and_conditioning():
    structure = _chain_structure(dim=2)
    dist = _make_dcm_dist(structure, n_steps=8)
    key = random.PRNGKey(0)

    s = dist.sample(key)
    assert set(s.keys()) == {'x0', 'x1'}
    for v in s.values():
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(v))

    # Conditional (ancestral) query
    s1 = dist.sample(key, conditions={'x0': jnp.array([0.5, -0.5])}, targets={'x1'})
    assert set(s1.keys()) == {'x1'}
    assert jnp.all(jnp.isfinite(s1['x1']))

    # Batched
    sb = dist.b_sample(key, 5)
    assert sb['x0'].shape == (5, 2)
    assert sb['x1'].shape == (5, 2)


def test_dcm_non_ancestral_query_raises_eagerly():
    """p(x0 | x1) requires inversion; must raise NotImplementedError from
    plain Python (before tracing)."""
    structure = _chain_structure()
    dist = _make_dcm_dist(structure, n_steps=4)
    with pytest.raises(NotImplementedError):
        dist.sample(random.PRNGKey(0), conditions={'x1': jnp.array([0.1])},
                    targets={'x0'})


def test_dcm_log_prob_raises_eagerly():
    structure = _chain_structure()
    dist = _make_dcm_dist(structure, n_steps=4)
    x = {'x0': jnp.array([0.1]), 'x1': jnp.array([0.2])}
    with pytest.raises(NotImplementedError):
        dist.log_prob(x)


def test_dcm_transform_to_noise_matches_manual_inversion():
    """transform_to_noise must be exactly the per-block ddim_invert with
    *factual* parents, for every block."""
    structure = _chain_structure(dim=2)
    dist = _make_dcm_dist(structure, n_steps=32)
    x = {'x0': jnp.array([0.3, -0.4]), 'x1': jnp.array([0.5, 0.2])}

    z = dist.transform_to_noise(x)
    assert set(z.keys()) == {'x0', 'x1'}
    for v in z.values():
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(v))

    blocks = dist.eps_fn.blocks
    z0_manual = ddim_invert(blocks['x0'], x['x0'], (), dist.n_steps)
    z1_manual = ddim_invert(blocks['x1'], x['x1'], (x['x0'],), dist.n_steps)
    assert jnp.array_equal(z['x0'], z0_manual)
    assert jnp.array_equal(z['x1'], z1_manual)


def test_dcm_per_block_roundtrip_error_shrinks_with_nfe():
    """Per-block reconstruction error of invert-then-forward (the abduction
    round trip used by counterfactuals) must shrink roughly linearly in NFE."""
    structure = _chain_structure(dim=2)
    x = {'x0': jnp.array([0.3, -0.4]), 'x1': jnp.array([0.5, 0.2])}

    def block_err(n_steps):
        dist = _make_dcm_dist(structure, n_steps=n_steps)
        z = dist.transform_to_noise(x)
        rec = dist._simulate({'x1': z['x1']}, {'x0': x['x0']}, ['x1'])['x1']
        return float(jnp.linalg.norm(rec - x['x1']))

    e32, e128 = block_err(32), block_err(128)
    assert e128 < e32 / 2.0, f"expected ~4x reduction from 32->128 NFE, got {e32} -> {e128}"


# ============================================================================
# Counterfactuals through the shared mixin
# ============================================================================

def test_dcm_counterfactual_contract():
    """Counterfactual on the chain, do(x0): the mixin must abduct x1's noise
    with *factual* parents, then re-simulate with the *intervened* parent.
    Verify against a manual computation of exactly that."""
    structure = _chain_structure(dim=2)
    dist = _make_dcm_dist(structure, n_steps=16)
    key = random.PRNGKey(4)
    data = dist.sample(key)
    do_x0 = jnp.array([2.0, -2.0])

    cf, log_prob = dist.counterfactual(data, {'x0': do_x0})

    # Intervention respected exactly; log_prob is NaN (capability gap).
    assert jnp.array_equal(cf['x0'], do_x0)
    assert jnp.isnan(log_prob)

    # Manual abduction-action-prediction for x1:
    block_x1 = dist.eps_fn.blocks['x1']
    u1 = ddim_invert(block_x1, data['x1'], (data['x0'],), dist.n_steps)
    x1_manual = ddim_forward(block_x1, u1, (do_x0,), dist.n_steps)
    assert jnp.allclose(cf['x1'], x1_manual), (cf['x1'], x1_manual)


def test_dcm_counterfactual_samples_only_path():
    structure = _chain_structure()
    dist = _make_dcm_dist(structure, n_steps=8)
    data = dist.sample(random.PRNGKey(5))
    cf_full, lp_full = dist.counterfactual(data, {'x0': jnp.array([1.0])})
    cf_fast, lp_fast = dist.counterfactual(data, {'x0': jnp.array([1.0])},
                                           compute_log_prob=False)
    assert jnp.isnan(lp_full) and jnp.isnan(lp_fast)
    for k in cf_full:
        assert jnp.array_equal(cf_full[k], cf_fast[k])


def test_dcm_batched_counterfactual():
    structure = _chain_structure()
    dist = _make_dcm_dist(structure, n_steps=8)
    n = 4
    data = dist.b_sample(random.PRNGKey(6), n)
    interventions = {'x0': jnp.tile(jnp.array([1.0]), (n, 1))}
    cf, lp = dist.b_counterfactual(data, interventions)
    assert cf['x0'].shape == (n, 1)
    assert cf['x1'].shape == (n, 1)
    assert lp.shape == (n,)
    assert jnp.all(jnp.isnan(lp))


def test_dcm_dispatch_via_create_distribution():
    structure = _chain_structure()
    model, _ = create_dcm_flow(
        structure=structure, depth=2, width=16,
        model_key=random.PRNGKey(0), optimizer=optax.adam(1e-3))
    structure_dict = {
        'dependencies': structure.dependencies,
        'block_order': structure.block_order,
        'dims': structure.dims,
    }
    from structflow.conditional_flows.dcm.dist import DCMDistribution
    dist = create_distribution(mode='dcm', structure_dict=structure_dict,
                               model=model, n_steps=8)
    assert isinstance(dist, DCMDistribution)
    assert dist.n_steps == 8
    s = dist.sample(random.PRNGKey(1))
    assert set(s.keys()) == {'x0', 'x1'}


# ============================================================================
# Learning: ε-MSE objective actually fits a distribution
# ============================================================================

def test_dcm_learns_shifted_gaussian():
    """Train the DCM backbone on a 1D shifted/scaled Gaussian chain and check
    the learned conditional mean moves toward the truth."""
    structure = _chain_structure(dim=1)

    model, step_fn = create_dcm_flow(
        structure=structure, depth=3, width=32,
        model_key=random.PRNGKey(0), optimizer=optax.adam(3e-3))
    optimizer = optax.adam(3e-3)
    import equinox as eqx
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Ground truth: x0 ~ N(1.5, 0.5^2), x1 = 0.8 * x0 + N(0, 0.3^2)
    def sample_batch(k, n):
        k0, k1 = random.split(k)
        x0 = 1.5 + 0.5 * random.normal(k0, (n, 1))
        x1 = 0.8 * x0 + 0.3 * random.normal(k1, (n, 1))
        return {'x0': x0, 'x1': x1}

    key = random.PRNGKey(42)
    losses = []
    for i in range(400):
        key, dk, sk = random.split(key, 3)
        batch = sample_batch(dk, 128)
        model, opt_state, _, loss = step_fn(model, opt_state, None, batch, sk)
        losses.append(float(loss))

    assert np.mean(losses[-50:]) < np.mean(losses[:50]), "loss did not decrease"

    dist = create_dcm_distribution(model, structure, n_steps=64)
    samples = dist.b_sample(random.PRNGKey(7), 500)
    x0_mean = float(jnp.mean(samples['x0']))
    assert abs(x0_mean - 1.5) < 0.4, f"x0 mean {x0_mean}, expected ~1.5"
    # Conditional structure: E[x1 | do-nothing] ≈ 0.8 * E[x0]
    x1_mean = float(jnp.mean(samples['x1']))
    assert abs(x1_mean - 0.8 * 1.5) < 0.5, f"x1 mean {x1_mean}, expected ~1.2"
