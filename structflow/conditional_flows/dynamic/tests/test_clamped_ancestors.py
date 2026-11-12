"""Dynamic conditioning with a clamped block that has ancestors.

Fixture: a linear-Gaussian chain x0 -> x1 -> x2 with scalar blocks and the exact flow-matching
field under the per-block (Knothe-Rosenblatt) coupling. That coupling pairs noise e with data
x = L e (L the Cholesky factor of the joint covariance), so x_t = M_t e with
M_t = (1 - t) I + t L and the field is v(x, t) = (L - I) M_t^{-1} x, which is lower-triangular,
i.e. a valid dynamic (block-triangular) field. The flow is exactly x = L e.

Clamping x1 = a (a do() on x1, which is also the conditional query p(x2 | x1 = a) since x1 is
x2's only parent) must give x2 ~ N(w2 a, s2). In the dynamic flow x2 reads x1's *trajectory*,
which depends on x0's; the clamped block's trajectory must therefore be built from x0's actual
trajectory (sampled when sampling, abducted in a counterfactual). Building it from a fixed
placeholder for x0 adds a term proportional to x0's noise: the interventional variance is
inflated and the counterfactual is biased.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from structflow.bayes.distribution import Gaussian
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.dynamic.parallel_flow import ParallelFlow
from structflow.conditional_flows.dynamic.flow import DynamicFlow
from structflow.conditional_flows.dynamic.dist import DynamicDistribution

W1, W2 = 0.8, 0.9          # x1 = W1 x0 + e1,  x2 = W2 x1 + e2
V0, V1, V2 = 2.0, 0.5, 0.3  # var(x0), var(e1), var(e2)
BLOCKS = ["x0", "x1", "x2"]
STRUCT = BlockTriangularStructure(block_order=BLOCKS,
                                  dependencies={"x0": [], "x1": ["x0"], "x2": ["x1"]},
                                  dims={b: 1 for b in BLOCKS})


def _cov():
    c00 = V0
    c11 = W1**2 * V0 + V1
    c22 = W2**2 * c11 + V2
    c01, c12 = W1 * V0, W2 * c11
    c02 = W2 * c01
    return np.array([[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]])


class _KRChainVelocity(eqx.Module):
    structure: BlockTriangularStructure = eqx.field(static=True)
    block_offsets: dict = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    L: tuple = eqx.field(static=True)

    def __init__(self):
        self.structure = STRUCT
        self.block_offsets = {"x0": (0, 1), "x1": (1, 2), "x2": (2, 3)}
        self.output_dim = 3
        self.L = tuple(map(tuple, np.linalg.cholesky(_cov()).tolist()))

    def __call__(self, x, t):
        L = jnp.asarray(self.L)
        M = (1 - t) * jnp.eye(3) + t * L
        return (L - jnp.eye(3)) @ jnp.linalg.solve(M, x)


def _dist(n_steps=64):
    base = {b: Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1)) for b in BLOCKS}
    flow = DynamicFlow(ParallelFlow(_KRChainVelocity(), n_steps=n_steps, integrator_name="rk4"))
    return DynamicDistribution(flow, base)


def test_flow_fixture_reproduces_the_joint():
    """Positive control: the fixture field transports N(0, I) to the chain's joint."""
    d = _dist()
    x = jax.vmap(lambda k: d.sample(k))(random.split(random.PRNGKey(0), 4000))
    X = np.concatenate([np.asarray(x[b]) for b in BLOCKS], 1)
    assert np.abs(np.cov(X.T) - _cov()).max() < 0.15


def test_clamped_block_uses_sampled_ancestor_trajectories():
    """p(x2 | do(x1 = a)) = N(W2 a, V2). A placeholder trajectory for x0 inflates the
    variance to V2 + (W2 W1)^2 V0 (here 0.3 + 1.04)."""
    a = 1.5
    d = _dist()
    f = eqx.filter_jit(lambda dd, ks: jax.vmap(
        lambda k: dd.sample(k, conditions={"x1": jnp.array([a])}, targets={"x2"}))(ks))
    s = np.asarray(f(d, random.split(random.PRNGKey(1), 4000))["x2"]).ravel()
    assert abs(s.mean() - W2 * a) < 0.05, s.mean()
    assert abs(s.var() - V2) < 0.05, s.var()


def test_counterfactual_uses_abducted_ancestor_trajectories():
    """x2_cf = x2 + W2 (a - x1) exactly for every factual individual."""
    a = 1.5
    d = _dist()
    true = _dist()  # sample factuals from the same (exact) flow
    xs = jax.vmap(lambda k: true.sample(k))(random.split(random.PRNGKey(2), 64))
    cf = eqx.filter_jit(lambda dd, x: jax.vmap(lambda xi: dd.counterfactual(
        xi, {"x1": jnp.array([a])}, targets={"x2"})[0]["x2"])(x))(d, xs)
    expected = xs["x2"] + W2 * (a - xs["x1"])
    assert np.abs(np.asarray(cf) - np.asarray(expected)).max() < 1e-2
