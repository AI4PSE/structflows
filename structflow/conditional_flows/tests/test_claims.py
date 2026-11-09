"""
Claim-driven regression tests (see ~/.claude/skills/claim-driven-tests).

The rest of the suite checks the samplers on an *independent* two-block Gaussian, with oracle
velocities and with OT coupling switched off. Four real problems were invisible there:
dynamic conditioning is inconsistent for correlated blocks, the triangular masks were trainable,
joint OT coupling biases static conditionals, and the guidance weight is too small by a factor
of t. Every test here sits on a fixture where a wrong implementation gives a different number:

    y ~ N(0, 1),  theta = rho * y + sqrt(1 - rho^2) * eps,  rho = 0.9,  condition y = C

so p(theta | y = C) = N(rho C, 1 - rho^2) = N(1.35, 0.19). Interpolant throughout:
x_t = (1 - t) e + t x1, e ~ N(0, I) (OneSidedLinear).

Claims and the bug each test catches:
  C1  The exact conditional field gives the exact conditional (static path, correlated parent).
      Catches: parents not passed / passed in the wrong order.
  C2  Dynamic conditioning: at the population optimum of joint FM over block-triangular fields
      the joint is NOT recovered for correlated blocks (cov 0.571 instead of 0.9), but it is for
      independent ones. The library's dynamic sampler must reproduce that optimum's conditional.
      Catches: a regression in the backward-then-forward conditioning path, and anyone
      re-asserting the (false) consistency claim.
  C3  Masked (dynamic) networks stay exactly block-triangular after training.
      Catches: the mask being trained (a plain array leaf seen by the optimiser).
  C4  Training noise is independent of the condition (train/sample parity), for each coupling.
      Catches: joint OT coupling (known bias, xfail) and mis-wiring of use_coupling='c2ot'.
  C5  Guidance with the exact likelihood score reproduces the exact conditional velocity.
      Catches: a wrong guidance weight (known: code uses (1 - t) instead of (1 - t) / t, xfail).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import random
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm

from structflow.bayes.distribution import Gaussian
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.static.dist import StaticConditionalDistribution
from structflow.conditional_flows.static.flow import StaticConditionalFlow
from structflow.conditional_flows.parallel.dist import ParallelBlockTriangularDistribution
from structflow.conditional_flows.parallel.flow import ParallelFlow
from structflow.conditional_flows.parallel.layers import DenseMaskedLinear
from structflow.conditional_flows.flow_factory import create_static_flow, create_dynamic_flow
from structflow.conditional_flows.dist_factory import create_distribution
from structflow.conditional_flows.guidance.guidance_based import AbstractGuidedVelocity
from sinterp.interpolants import OneSidedLinear
from sinterp.stochasticfield import OneSidedField_eta1
from sinterp.couplings import C2OTCoupling

RHO, C = 0.9, 1.5
S1SQ = 1 - RHO**2
SIGMA = np.array([[1.0, RHO], [RHO, 1.0]])
STRUCT_DICT = {"dependencies": {"y": [], "theta": ["y"]}, "block_order": ["y", "theta"],
               "dims": {"y": 1, "theta": 1}}
STRUCT = BlockTriangularStructure(**STRUCT_DICT)
N_SAMPLES = 4000
TRUE_MEAN, TRUE_VAR = RHO * C, S1SQ


def _toy_batch(key, n=128, rho=RHO):
    k1, k2 = random.split(key)
    y = random.normal(k1, (n, 1))
    return {"y": y, "theta": rho * y + np.sqrt(1 - rho**2) * random.normal(k2, (n, 1))}


def _mc_tol(var, n=N_SAMPLES, k=5.0, disc=0.01):
    """k Monte Carlo standard errors of a mean, plus a stated discretisation allowance."""
    return k * np.sqrt(var / n) + disc


def _sample_theta(dist, c=C, n=N_SAMPLES, seed=7):
    f = eqx.filter_jit(lambda d, cv, ks: jax.vmap(
        lambda k: d.sample(k, conditions=cv, targets=frozenset({"theta"})))(ks))
    s = f(dist, {"y": jnp.array([c])}, random.split(random.PRNGKey(seed), n))["theta"]
    s = np.asarray(s).ravel()
    return s.mean(), s.var()


# ---------------------------------------------------------------------------------------------
# Closed-form Gaussian fields (independent derivation; nothing here calls library code)
#   v(x, t) = E[x1 - e | x_t = x] = A(t) x,  A = (t S - (1-t) I) (t^2 S + (1-t)^2 I)^-1
#   triangular optimum: y-row = E[ydot | y_t] (scalar regression), theta-row = true row
# ---------------------------------------------------------------------------------------------

def _A_full(t, S=SIGMA):
    M = t**2 * S + (1 - t)**2 * np.eye(len(S))
    return (t * S - (1 - t) * np.eye(len(S))) @ np.linalg.inv(M)


def _A_tri(t, S=SIGMA):
    A = _A_full(t, S).copy()
    A[0, 0] = (t * S[0, 0] - (1 - t)) / (t**2 * S[0, 0] + (1 - t)**2)
    A[0, 1:] = 0.0
    return A


def _push_cov(A_fn, S=SIGMA):
    """Covariance at t=1 of the flow of v = A(t) x started from N(0, I): P' = AP + PA^T."""
    d = len(S)
    rhs = lambda t, p: (lambda P, A: (A @ P + P @ A.T).ravel())(p.reshape(d, d), A_fn(t, S))
    sol = solve_ivp(rhs, (0.0, 1.0), np.eye(d).ravel(), method="DOP853", rtol=1e-11, atol=1e-12)
    return sol.y[:, -1].reshape(d, d)


def _conditional_from_cov(P, c=C):
    return P[0, 1] / P[0, 0] * c, P[1, 1] - P[0, 1]**2 / P[0, 0]


class _LinearParallelVelocity(eqx.Module):
    """Flat linear field v = A(t) x for the dynamic (parallel) sampler, target N(0, S)."""
    structure: BlockTriangularStructure = eqx.field(static=True)
    block_offsets: dict = eqx.field(static=True)
    triangular: bool = eqx.field(static=True)
    S: tuple = eqx.field(static=True)

    def __init__(self, S, triangular=True):
        self.structure, self.triangular = STRUCT, triangular
        self.block_offsets = {"y": (0, 1), "theta": (1, 2)}
        self.S = tuple(map(tuple, np.asarray(S).tolist()))

    def __call__(self, x, t):
        S = jnp.asarray(self.S)
        M = t**2 * S + (1 - t)**2 * jnp.eye(2)
        A = (t * S - (1 - t) * jnp.eye(2)) @ jnp.linalg.inv(M)
        if self.triangular:
            ayy = (t * S[0, 0] - (1 - t)) / (t**2 * S[0, 0] + (1 - t)**2)
            A = A.at[0, 0].set(ayy).at[0, 1].set(0.0)
        return A @ x


class _ExactConditionalStaticVelocity(eqx.Module):
    """Exact static fields: y ~ N(0,1) unconditionally; theta | y ~ N(rho y, 1-rho^2)."""
    structure: BlockTriangularStructure = eqx.field(static=True)

    def __init__(self):
        self.structure = STRUCT

    @staticmethod
    def _v1d(x, t, mu, s2):
        # E[x1 - e | x_t = x] for x1 ~ N(mu, s2), e ~ N(0, 1)
        return mu + (t * s2 - (1 - t)) / (t**2 * s2 + (1 - t)**2) * (x - t * mu)

    @property
    def blocks(self):
        return {"y": lambda t, x: self._v1d(x, t, 0.0, 1.0),
                "theta": lambda t, x, y: self._v1d(x, t, RHO * y, S1SQ)}

    def __call__(self, x_dict, t, active_blocks=None):
        out = {}
        for name in active_blocks or self.structure.block_order:
            parents = [x_dict[p] for p in self.structure.dependencies[name]]
            out[name] = self.blocks[name](t, x_dict[name], *parents)
        return out


def _base():
    return {"y": Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1)),
            "theta": Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1))}


# ---------------------------------------------------------------------------------------------
# C1: static conditioning is exact on a correlated fixture
# ---------------------------------------------------------------------------------------------

def test_static_sampler_exact_conditional_with_correlated_parent():
    """C1. Exact conditional fields through the library's static sampler give N(rho c, 1-rho^2).
    Catches: parent values not reaching the block field (the independent fixture cannot)."""
    flow = StaticConditionalFlow(_ExactConditionalStaticVelocity(), n_steps=100)
    m, v = _sample_theta(StaticConditionalDistribution(flow, _base()))
    assert abs(m - TRUE_MEAN) < _mc_tol(TRUE_VAR), f"mean {m:.4f} vs {TRUE_MEAN}"
    assert abs(v - TRUE_VAR) < 0.03, f"var {v:.4f} vs {TRUE_VAR}"


# ---------------------------------------------------------------------------------------------
# C2: dynamic conditioning at the population optimum
# ---------------------------------------------------------------------------------------------

def test_closed_form_full_field_recovers_target():
    """Positive control for the derivation and the covariance ODE: the unconstrained field
    transports N(0, I) to N(0, Sigma)."""
    np.testing.assert_allclose(_push_cov(_A_full), SIGMA, atol=1e-6)


def test_triangular_optimum_exact_for_independent_blocks():
    """With rho = 0 the triangular projection is the true field: dynamic conditioning is exact.
    (This is the only case the older numerical tests cover.)"""
    S0 = np.eye(2)
    np.testing.assert_allclose(_push_cov(_A_tri, S0), S0, atol=1e-6)


def test_triangular_optimum_inconsistent_for_correlated_blocks():
    """C2. The population optimum of joint flow matching over block-triangular fields does not
    recover the joint when blocks are correlated: at rho = 0.9 it gives cov(y, theta) = 0.5712
    and p(theta | y=c) = N(0.5712 c, 0.4357) instead of N(0.9 c, 0.19). Consistency holds iff
    E[ydot | y_t, theta_t] = E[ydot | y_t]. If this test starts failing, re-derive before
    asserting dynamic conditioning is consistent (rem:dynamic-consistency, Jeong et al. 2025)."""
    P = _push_cov(_A_tri)
    assert abs(P[0, 0] - 1.0) < 1e-6, "y-marginal must still be exact"
    assert abs(P[0, 1] - 0.5712) < 2e-3, f"cov {P[0, 1]:.4f}"
    assert abs(P[0, 1] - RHO) > 0.3


# y must NOT be N(0, 1) here: with the base equal to the y-marginal the y-flow is the identity,
# so backward(y=c) = c and a sampler that skips the backward inversion would still pass
# (found by mutation M3). var(y) = 2 makes the inversion matter.
SIGMA_DYN = np.array([[2.0, RHO * np.sqrt(2.0)], [RHO * np.sqrt(2.0), 1.0]])


def test_dynamic_sampler_reproduces_triangular_optimum_conditional():
    """C2, library path. Feeding the triangular-optimum field to the dynamic (parallel)
    sampler must give the conditional predicted by the covariance ODE -- not the truth.
    Catches regressions in the backward(y) / forward(theta) conditioning path on a fixture
    where y and theta interact and the y-flow is not the identity."""
    P = _push_cov(_A_tri, SIGMA_DYN)
    m_pred, v_pred = _conditional_from_cov(P)
    m_true = SIGMA_DYN[0, 1] / SIGMA_DYN[0, 0] * C
    flow = ParallelFlow(_LinearParallelVelocity(SIGMA_DYN, triangular=True), n_steps=100)
    m, v = _sample_theta(ParallelBlockTriangularDistribution(flow, _base()))
    assert abs(m - m_pred) < _mc_tol(v_pred), f"mean {m:.4f} vs predicted {m_pred:.4f}"
    assert abs(v - v_pred) < 0.04, f"var {v:.4f} vs predicted {v_pred:.4f}"
    assert abs(m_pred - m_true) > 0.2, "fixture no longer separates the optimum from the truth"


# ---------------------------------------------------------------------------------------------
# C3: masks stay fixed under optimisation
# ---------------------------------------------------------------------------------------------

def _masks(model):
    leaves = jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, DenseMaskedLinear))
    return [np.asarray(l.mask) for l in leaves if isinstance(l, DenseMaskedLinear)]


def test_dynamic_masks_unchanged_and_triangular_after_training():
    """C3. After 300 Adam steps at lr 1e-2 on correlated data, every mask is bit-identical and
    d v_y / d theta_t is exactly 0. Catches: the mask being an optimised array leaf (before the
    fix it drifted by ~1e-2 within 200 steps)."""
    opt = optax.adam(1e-2)
    model, step = create_dynamic_flow(STRUCT, depth=3, width=32, model_key=random.PRNGKey(0),
                                      optimizer=opt, use_coupling=True)
    m0 = _masks(model)
    assert m0 and all(np.isin(m, (0.0, 1.0)).all() for m in m0)
    opt_state, key = opt.init(eqx.filter(model, eqx.is_array)), random.PRNGKey(1)
    for _ in range(300):
        key, kd, ks = random.split(key, 3)
        model, opt_state, _, _ = step(model, opt_state, None, _toy_batch(kd), ks)
    for a, b in zip(_masks(model), m0):
        np.testing.assert_array_equal(a, b)
    for t in (0.1, 0.5, 0.9):
        for x in ([0.5, -1.0], [-1.2, 0.3], [1.0, 1.0]):
            J = jax.jacfwd(lambda z: model.get_velocity_b(z, t))(jnp.array(x))
            assert float(J[0, 1]) == 0.0, f"d v_y / d theta = {float(J[0, 1])} at t={t}, x={x}"


# ---------------------------------------------------------------------------------------------
# C4: train/sample parity of the noise, end to end through create_static_flow
# ---------------------------------------------------------------------------------------------

def _train_static(use_coupling, steps=3000, seed=0):
    opt = optax.adam(optax.cosine_decay_schedule(3e-3, steps, alpha=0.01))
    model, step = create_static_flow(STRUCT, depth=3, width=64, model_key=random.PRNGKey(seed),
                                     optimizer=opt, use_coupling=use_coupling)
    opt_state, key = opt.init(eqx.filter(model, eqx.is_array)), random.PRNGKey(100 + seed)
    step = eqx.filter_jit(step)
    for _ in range(steps):
        key, kd, ks = random.split(key, 3)
        model, opt_state, _, _ = step(model, opt_state, None, _toy_batch(kd), ks)
    return create_distribution("static", STRUCT_DICT, model, n_steps=50, integrator_name="rk4")


@pytest.mark.expensive
@pytest.mark.slow
@pytest.mark.parametrize("use_coupling", [
    False,
    "c2ot",
    pytest.param(True, marks=pytest.mark.xfail(
        strict=True, reason="known: joint OT coupling makes the training noise depend on the "
                            "condition; at rho=0.9, y=1.5 it gives N(0.89, 0.30)")),
])
def test_static_trained_conditional_matches_truth(use_coupling):
    """C4, reported quantity. Train a static flow on the correlated toy with each coupling and
    compare p(theta | y=1.5) with the truth. Tolerance: 5 MC se + 0.08 for finite training
    (independent coupling reaches ~0.06 at 30k steps). The joint-OT bias is ~0.45."""
    m, v = _sample_theta(_train_static(use_coupling))
    assert abs(m - TRUE_MEAN) < _mc_tol(TRUE_VAR, disc=0.08), f"mean {m:.3f} vs {TRUE_MEAN}"
    assert abs(v - TRUE_VAR) < 0.06, f"var {v:.3f} vs {TRUE_VAR}"


# ---------------------------------------------------------------------------------------------
# C5: guidance weight against the exact conditional velocity
# ---------------------------------------------------------------------------------------------

SIG_O = 0.3          # soft observation y_obs ~ N(y1, SIG_O^2) keeps every term finite


class _AnalyticEta1(eqx.Module):
    """Exact E[x1 | x_t] for x_t = (1-t) e + t x1, x1 ~ N(0, Sigma)."""
    def __call__(self, x, t, *args):
        S = jnp.asarray(SIGMA)
        return t * S @ jnp.linalg.solve(t**2 * S + (1 - t)**2 * jnp.eye(2), x)


def _loglik_t(x, t):
    """log p_t(y_obs = C | x_t = x) in closed form."""
    S = jnp.asarray(SIGMA)
    M = t**2 * S + (1 - t)**2 * jnp.eye(2)
    m = t * S @ jnp.linalg.solve(M, x)
    cov = S - t**2 * S @ jnp.linalg.solve(M, S)
    var = cov[0, 0] + SIG_O**2
    return -0.5 * (C - m[0])**2 / var - 0.5 * jnp.log(var)


def _posterior_velocity(x, t):
    """Exact velocity of the interpolant to the posterior p(x1 | y_obs = C) = N(m', S')."""
    S = np.asarray(SIGMA)
    k = S[:, 0] / (S[0, 0] + SIG_O**2)
    mp = jnp.asarray(k * C)
    Sp = jnp.asarray(S - np.outer(k, S[0, :]))
    M = t**2 * Sp + (1 - t)**2 * jnp.eye(2)
    return mp + (t * Sp - (1 - t) * jnp.eye(2)) @ jnp.linalg.solve(M, x - t * mp)


class _ExactScoreGuided(AbstractGuidedVelocity):
    def get_guidance_vector(self, x, t, observation):
        return jax.grad(_loglik_t)(x, t)


GUIDE_POINTS = [(t, x) for t in (0.2, 0.5, 0.8) for x in ([0.3, -0.7], [-1.1, 0.4], [1.2, 1.5])]


def test_exact_guidance_weight_derivation():
    """Positive control (independent of the library weight): the exact conditional velocity
    equals prior velocity + K(t) * grad log p_t(y|x) with K = (alpha^2+gamma^2) beta'/beta
    - (alpha alpha' + gamma gamma') = (1-t)/t for OneSidedLinear."""
    field = OneSidedField_eta1(OneSidedLinear(), _AnalyticEta1())
    for t, x in GUIDE_POINTS:
        x = jnp.array(x)
        lhs = _posterior_velocity(x, t) - field.get_velocity_b(x, t)
        np.testing.assert_allclose(lhs, (1 - t) / t * jax.grad(_loglik_t)(x, t), atol=1e-5)


@pytest.mark.xfail(strict=True, reason="known: get_weight returns -scale*gamma*gammadot = (1-t), "
                                       "the exact weight is (1-t)/t (FINDINGS.md, finding 4)")
def test_library_guidance_weight_gives_exact_conditional_velocity():
    """C5. With the exact likelihood score and scale 1, the library's guided field must equal
    the exact conditional velocity. Catches any error in AbstractGuidedVelocity.get_weight."""
    field = OneSidedField_eta1(OneSidedLinear(), _AnalyticEta1())
    guided = _ExactScoreGuided(field, scale=1.0)
    for t, x in GUIDE_POINTS:
        x = jnp.array(x)
        np.testing.assert_allclose(guided(x, t, None), _posterior_velocity(x, t), atol=1e-4)


def test_library_guidance_pushes_along_the_likelihood_score():
    """C5, the part that holds despite the known factor-of-t error: the exact weight
    K = (1-t)/t is positive on (0, 1), so guidance must move the velocity *along* the
    likelihood score. Kept separate from the xfail above so that a sign error is still
    caught while the known weight error stands (found by mutation M6)."""
    field = OneSidedField_eta1(OneSidedLinear(), _AnalyticEta1())
    guided = _ExactScoreGuided(field, scale=1.0)
    for t, x in GUIDE_POINTS:
        x = jnp.array(x)
        delta = guided(x, t, None) - field.get_velocity_b(x, t)
        score = jax.grad(_loglik_t)(x, t)
        assert float(delta @ score) > 0.0, f"guidance opposes the likelihood at t={t}, x={x}"


# --------------------------------------------------------------------------------------
# C6: per-block coupling in topological order (Knothe-Rosenblatt) makes dynamic
#     conditioning consistent; all couplings are exact permutations ("fair").
# --------------------------------------------------------------------------------------

def _A_full_coupled(t, S, T):
    """Unconstrained optimum for the deterministic coupling x1 = T e (T T^T = S), derived from
    the joint moments of (x_t, x1 - e); independent of the library's code."""
    I = np.eye(len(S))
    K = (1 - t)**2 * I + t**2 * S + t * (1 - t) * (T + T.T)
    C = (1 - t) * T + t * S - (1 - t) * I - t * T.T
    return C @ np.linalg.inv(K)


def test_kr_coupling_makes_true_field_block_triangular():
    """Claim C6 (population): with the Knothe-Rosenblatt coupling T = chol(S) the *unconstrained*
    flow-matching optimum is already lower-triangular, so the dynamic net's constrained optimum
    equals it and transports N(0, I) exactly to N(0, S). The symmetric (joint-OT) coupling does
    not. Catches: a 'consistency' argument that only holds for one coupling, or a wrong block order."""
    for S in (SIGMA, SIGMA_DYN):
        L = np.linalg.cholesky(S)
        R = np.real(sqrtm(S))
        ts = np.linspace(0.01, 0.99, 25)
        assert max(abs(_A_full_coupled(t, S, L)[0, 1]) for t in ts) < 1e-12
        assert max(abs(_A_full_coupled(t, S, R)[0, 1]) for t in ts) > 0.1
        P = _push_cov(lambda t, S_: _A_full_coupled(t, S_, np.linalg.cholesky(S_)), S)
        assert np.abs(P - S).max() < 1e-6, P


def _coupled_cross_cov(use_coupling, n=2048, seed=0):
    from structflow.conditional_flows.flow_factory import _resolve_coupling, _couple
    x = _toy_batch(random.PRNGKey(seed), n)
    k1, k2 = random.split(random.PRNGKey(seed + 1))
    e = {"y": random.normal(k1, (n, 1)), "theta": random.normal(k2, (n, 1))}
    kind, coupling = _resolve_coupling(use_coupling)
    xc, ec = _couple(kind, coupling, random.PRNGKey(2), x, e, STRUCT)
    E = np.concatenate([np.asarray(ec["y"]), np.asarray(ec["theta"])], 1)
    X = np.concatenate([np.asarray(xc["y"]), np.asarray(xc["theta"])], 1)
    return E, X, e


def test_per_block_coupling_approaches_knothe_rosenblatt():
    """Claim C6 (code): the library's 'c2ot' option (one matching per block, parents as the
    condition) is a minibatch Knothe-Rosenblatt coupling: Cov(e, x1) -> chol(S)^T. In particular
    the child's noise is uncorrelated with the parent, while joint Hungarian OT gives
    Cov(e, x1) -> S^{1/2} (child noise correlated with the parent, 0.53 at rho = 0.9).
    Catches: couple_blocks ignoring the parents, or coupling all blocks with one matching."""
    E, X, _ = _coupled_cross_cov(C2OTCoupling(target_ratio=0.05))
    cc = E.T @ X / len(E)                           # rows: e_y, e_theta; cols: y, theta
    assert cc[0, 0] > 0.97 and abs(cc[0, 1] - RHO) < 0.05, cc       # e_y ~ y (1-D OT ~ sorting)
    assert abs(cc[1, 0]) < 0.05, cc                                  # e_theta independent of y
    assert cc[1, 1] > 0.2, cc                                        # ... but OT-coupled to theta
    E, X, _ = _coupled_cross_cov("hungarian")
    cc = E.T @ X / len(E)
    assert abs(cc[1, 0] - 0.531) < 0.05 and abs(cc[0, 1] - 0.531) < 0.05, cc


@pytest.mark.parametrize("use_coupling", ["hungarian", "c2ot"])
def test_hard_couplings_are_exact_permutations_of_the_noise(use_coupling):
    """Fairness: 'hungarian' and 'c2ot' reorder noise rows and never average them, so the noise
    the network is trained from is exactly the N(0, I) draw (Sinkhorn's barycentric projection
    shrinks its variance). Catches: a soft/barycentric solver behind the hard options."""
    E, _, e = _coupled_cross_cov(use_coupling, n=256)
    for j, b in enumerate(["y", "theta"]):
        assert np.array_equal(np.sort(E[:, j]), np.sort(np.asarray(e[b]).ravel()))


@pytest.mark.parametrize("factory", ["static", "dynamic", "joint"])
@pytest.mark.parametrize("use_coupling", [False, True, "hungarian", "c2ot"])
def test_every_factory_accepts_every_coupling(factory, use_coupling):
    """Config parity: every flow factory takes every coupling option and trains one finite step."""
    from structflow.conditional_flows.flow_factory import create_joint_flow
    make = {"static": create_static_flow, "dynamic": create_dynamic_flow, "joint": create_joint_flow}[factory]
    opt = optax.adam(1e-3)
    model, step = make(structure=STRUCT, depth=2, width=16, model_key=random.PRNGKey(0),
                       optimizer=opt, use_coupling=use_coupling)
    state = opt.init(eqx.filter(model, eqx.is_array))
    _, _, _, loss = step(model, state, None, _toy_batch(random.PRNGKey(1), 32), random.PRNGKey(2))
    assert np.isfinite(float(loss))
    with pytest.raises(ValueError):
        make(structure=STRUCT, depth=2, width=16, model_key=random.PRNGKey(0), optimizer=opt,
             use_coupling="sinkhorn-typo")
