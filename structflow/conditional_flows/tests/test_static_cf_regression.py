"""Regression test for static counterfactuals (abduction, action, prediction).

The counterfactual graph logic (ancestral closure, descendant sets, abduction block
selection) lives in ``conditional_flows.counterfactual.CounterfactualMixin`` and is shared
with the DCM diffusion backbone used in the paper's backbone ablation. This test rebuilds a
few small static distributions with fixed keys and data and compares their counterfactual
samples and log-probabilities with stored reference outputs in
``data/static_cf_reference.npz``.

The reference was captured on CPU with jax 0.6.2 (the version pinned by the paper
repository). On that version the outputs must match exactly; on other jax versions,
compiled float32 ODE solves differ slightly, so samples are compared with a tolerance.

Note on determinism: every case here has at most two simulation blocks, so the only
order-sensitive float accumulations (divergence and base log-prob sums over ``list(set)``)
involve at most two terms, and IEEE addition is commutative, making the outputs invariant
to Python's per-process set iteration order.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
import pytest

from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.flow_factory import create_static_flow
from structflow.conditional_flows.dist_factory import create_static_distribution
from structflow.conditional_flows.dist import NormalisedDistribution
from structflow.conditional_flows.utils import DictNormaliser

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "data",
                              "static_cf_reference.npz")
REFERENCE_JAX = "0.6.2"          # jax version the reference outputs were captured with (CPU)
# Tolerances for other jax versions: observed drift is <= 3e-3 on samples and <= 3e-5
# relative on log-probs (the untrained networks give |log p| ~ 1e5).
SAMPLE_TOL = dict(rtol=1e-2, atol=1e-2)
LOG_PROB_RTOL = 1e-4

N_STEPS = 10


def _make_dist(structure, seed):
    model, _ = create_static_flow(
        structure=structure,
        depth=2,
        width=16,
        model_key=random.PRNGKey(seed),
        optimizer=optax.adam(1e-3),
        use_coupling=False,
    )
    return create_static_distribution(model, structure, n_steps=N_STEPS)


def _chain():
    return BlockTriangularStructure(
        block_order=['x0', 'x1'],
        dependencies={'x0': [], 'x1': ['x0']},
        dims={'x0': 1, 'x1': 1},
    )


def _fork():
    return BlockTriangularStructure(
        block_order=['z', 'x1', 'x2'],
        dependencies={'z': [], 'x1': ['z'], 'x2': ['z']},
        dims={'z': 2, 'x1': 2, 'x2': 2},
    )


def _collider():
    return BlockTriangularStructure(
        block_order=['x1', 'x2', 'y'],
        dependencies={'x1': [], 'x2': [], 'y': ['x1', 'x2']},
        dims={'x1': 1, 'x2': 1, 'y': 1},
    )


def build_cases():
    """Deterministic (dist, data, interventions, targets, compute_log_prob)
    tuples keyed by case name. Fixed keys and fixed data arrays: no sampling."""
    chain_dist = _make_dist(_chain(), seed=0)
    fork_dist = _make_dist(_fork(), seed=1)
    collider_dist = _make_dist(_collider(), seed=2)

    chain_data = {'x0': jnp.array([0.3]), 'x1': jnp.array([-0.7])}
    fork_data = {
        'z': jnp.array([0.5, -0.2]),
        'x1': jnp.array([-1.1, 0.4]),
        'x2': jnp.array([0.9, 0.1]),
    }
    collider_data = {
        'x1': jnp.array([0.2]),
        'x2': jnp.array([-0.4]),
        'y': jnp.array([1.3]),
    }

    # A normalised wrapper case exercises the NormalisedDistribution ->
    # inner-counterfactual path through the mixin.
    norm_stats_data = {
        'x0': random.normal(random.PRNGKey(10), (256, 1)) * 2.0 + 1.0,
        'x1': random.normal(random.PRNGKey(11), (256, 1)) * 0.5 - 1.0,
    }
    normaliser = DictNormaliser.from_data(norm_stats_data)
    norm_dist = NormalisedDistribution(dist=_make_dist(_chain(), seed=3),
                                       normaliser=normaliser)

    return {
        'chain_do_x0': (
            chain_dist, chain_data, {'x0': jnp.array([2.0])}, None, True),
        'chain_do_x0_targets_x1_samples_only': (
            chain_dist, chain_data, {'x0': jnp.array([2.0])}, {'x1'}, False),
        'chain_no_intervention': (
            chain_dist, chain_data, {}, None, True),
        'fork_do_z': (
            fork_dist, fork_data, {'z': jnp.array([1.5, -1.5])}, None, True),
        'collider_do_x1': (
            collider_dist, collider_data, {'x1': jnp.array([-2.0])}, None, True),
        'normalised_chain_do_x0': (
            norm_dist, chain_data, {'x0': jnp.array([2.0])}, None, True),
    }


def run_case(case):
    dist, data, interventions, targets, compute_log_prob = case
    cf, log_prob = dist.counterfactual(
        data, interventions, targets=targets, compute_log_prob=compute_log_prob)
    out = {k: np.asarray(v) for k, v in cf.items()}
    out['__log_prob__'] = np.asarray(log_prob)
    return out


def capture(path=REFERENCE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {}
    for name, case in build_cases().items():
        for k, v in run_case(case).items():
            flat[f"{name}::{k}"] = v
    np.savez(path, **flat)
    print(f"Captured {len(flat)} reference arrays to {path}")


@pytest.mark.parametrize("case_name", sorted(build_cases().keys()))
def test_static_counterfactual_matches_reference(case_name):
    """Static counterfactual outputs match the stored reference: exactly on the reference
    jax version, within float tolerance on others (run on CPU: JAX_PLATFORMS=cpu)."""
    assert os.path.exists(REFERENCE_PATH), (
        f"Missing reference file {REFERENCE_PATH}; regenerate with "
        f"python -m conditional_flows.tests.test_static_cf_regression --capture "
        f"ONLY against a known-good implementation.")
    ref = np.load(REFERENCE_PATH)
    case = build_cases()[case_name]
    out = run_case(case)
    keys = [k.split('::', 1)[1] for k in ref.files if k.startswith(case_name + '::')]
    assert sorted(keys) == sorted(out.keys()), (
        f"Output keys changed: reference {sorted(keys)} vs now {sorted(out.keys())}")
    for k in keys:
        expected = ref[f"{case_name}::{k}"]
        got = out[k]
        if np.isnan(expected).all() and expected.shape == got.shape:
            assert np.isnan(got).all(), f"{case_name}::{k}: expected all-NaN"
            continue
        if jax.__version__ == REFERENCE_JAX:
            assert np.array_equal(expected, got), (
                f"{case_name}::{k} differs from the reference: expected {expected}, got {got}")
        elif k == "__log_prob__":
            np.testing.assert_allclose(got, expected, rtol=LOG_PROB_RTOL, err_msg=f"{case_name}::{k}")
        else:
            np.testing.assert_allclose(got, expected, **SAMPLE_TOL, err_msg=f"{case_name}::{k}")


def test_samples_only_path_matches_full_path_samples():
    """compute_log_prob=False must return exactly the same samples as the
    default path, with log_prob == NaN."""
    cases = build_cases()
    dist, data, interventions, targets, _ = cases['chain_do_x0']
    cf_full, lp_full = dist.counterfactual(data, interventions, targets=targets)
    cf_fast, lp_fast = dist.counterfactual(data, interventions, targets=targets,
                                           compute_log_prob=False)
    assert set(cf_full.keys()) == set(cf_fast.keys())
    for k in cf_full:
        assert np.array_equal(np.asarray(cf_full[k]), np.asarray(cf_fast[k]))
    assert np.isfinite(np.asarray(lp_full))
    assert np.isnan(np.asarray(lp_fast))


if __name__ == "__main__":
    import sys
    if "--capture" in sys.argv:
        capture()
    else:
        print("Run with --capture to (re)generate the reference file.")
