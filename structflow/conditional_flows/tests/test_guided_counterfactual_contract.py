"""Contract tests for GuidanceBasedDistribution.counterfactual.

The guidance-based (joint) flow computes the counterfactual density *jointly*
over all non-intervened variables — it cannot produce marginal densities. A
strict subset of the non-intervened variables therefore used to be accepted
silently while still returning the joint density; wrapped in a normaliser that
applied a Jacobian correction for only the requested subset, this produced
mismatched log-probabilities (observed as negative KL divergences).

These tests pin the intended contract: the full non-intervened set (or
``targets=None``) is accepted and agree; a strict subset, or targets overlapping
the intervention, raise ``NotImplementedError``.
"""

import jax.numpy as jnp
from jax import random
import optax
import pytest

from structflow.pgms.factories import make_chain_pgm
from structflow.conditional_flows.flow_factory import create_joint_flow
from structflow.conditional_flows.dist_factory import create_distribution


def _make_guided_dist(n_nodes=3, seed=0):
    key = random.PRNGKey(seed)
    pgm = make_chain_pgm(n_nodes=n_nodes, mechanism_type="linear", key=key)
    structure = pgm.structure
    structure_dict = {
        "dependencies": structure.dependencies,
        "block_order": structure.block_order,
        "dims": structure.dims,
    }
    model, _ = create_joint_flow(
        structure=structure, depth=2, width=16, model_key=key,
        optimizer=optax.adam(1e-3), use_coupling=False,
    )
    dist = create_distribution(mode="guided", structure_dict=structure_dict, model=model)
    return pgm, dist


def test_guided_counterfactual_full_set_and_none_agree():
    """The full non-intervened set and targets=None give the same log_prob."""
    pgm, dist = _make_guided_dist()
    data = pgm.sample(random.PRNGKey(1))
    interventions = {"x0": jnp.array([1.0])}  # non-intervened = {x1, x2}

    _, lp_full = dist.counterfactual(data, interventions, targets={"x1", "x2"})
    _, lp_none = dist.counterfactual(data, interventions, targets=None)

    assert jnp.isfinite(lp_full)
    assert jnp.allclose(lp_full, lp_none)


def test_guided_counterfactual_strict_subset_raises():
    """Requesting a strict subset of the non-intervened variables is unsupported."""
    pgm, dist = _make_guided_dist()
    data = pgm.sample(random.PRNGKey(1))
    interventions = {"x0": jnp.array([1.0])}  # non-intervened = {x1, x2}

    with pytest.raises(NotImplementedError):
        dist.counterfactual(data, interventions, targets={"x1"})


def test_guided_counterfactual_targets_overlapping_intervention_raises():
    """Targets that include an intervened variable are unsupported."""
    pgm, dist = _make_guided_dist()
    data = pgm.sample(random.PRNGKey(1))
    interventions = {"x0": jnp.array([1.0])}

    with pytest.raises(NotImplementedError):
        dist.counterfactual(data, interventions, targets={"x0", "x1", "x2"})
