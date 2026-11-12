"""Worked example: one collider PGM, two compiled flows, three kinds of query.

The graph is a collider with two independent causes and one common effect,

    x1 --> x0 <-- x2,        x1, x2 ~ N(0, 1),   x0 = x1 + x2 + 0.5 * eps,   eps ~ N(0, 1).

The compiler turns the DAG plus a query into a block-triangular flow structure:

* ``compile_causal()`` keeps the DAG order ``[x1, x2, x0]``. A flow trained with it is a
  structural causal model, so it answers interventional and counterfactual queries.
* ``compile_inference(targets={x1, x2}, conditions={x0})`` puts the observation first and
  *moralizes* the graph: observing the common effect couples its causes, so the inference
  structure gains an edge x2 -> x1 ("explaining away"). A flow trained with it samples the
  posterior p(x1, x2 | x0) directly, with no MCMC.

Both flows are trained on the same samples, and each answer is checked against its closed
form (the model is linear-Gaussian):

1. posterior given x0:  E[x1 | x0] = E[x2 | x0] = x0 / 2.25,  corr(x1, x2 | x0) = -0.8,
   although x1 and x2 are independent a priori;
2. intervention:       E[x0 | do(x1 = 2)] = 2;
3. counterfactual:     had x1 been 2, each observed individual's x0 would have been x0 + (2 - x1).

Run:  python examples/structflow_quickstart.py     (about a minute on a laptop CPU)
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from structflow.pgms import BlockTriangularStructure, StructuralPGM, PGMCompiler
from structflow.pgms.mechanisms import LinearMechanism
from structflow.conditional_flows import create_static_flow, create_distribution

NOISE = 0.5                                    # std of the noise on x0


def collider_pgm():
    """x1, x2 ~ N(0, 1); x0 = x1 + x2 + NOISE * eps."""
    structure = BlockTriangularStructure(block_order=["x1", "x2", "x0"],
                                         dependencies={"x1": [], "x2": [], "x0": ["x1", "x2"]},
                                         dims={"x1": 1, "x2": 1, "x0": 1})
    root = lambda: LinearMechanism(parent_names=[], input_dim=0, output_dim=1,
                                   A=jnp.zeros((1, 0)), B=jnp.eye(1), bias=jnp.zeros(1))
    effect = LinearMechanism(parent_names=["x1", "x2"], input_dim=2, output_dim=1,
                             A=jnp.ones((1, 2)), B=NOISE * jnp.eye(1), bias=jnp.zeros(1))
    return StructuralPGM(structure, {"x1": root(), "x2": root(), "x0": effect})


def train_flow(structure, data, steps, width, depth, lr, key):
    """Fit a static structured flow to `data` (a dict of arrays) with full-batch Adam."""
    model_key, key = random.split(key)
    optimizer = optax.adam(lr)
    model, step_fn = create_static_flow(structure=structure, depth=depth, width=width,
                                        model_key=model_key, optimizer=optimizer)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    for _ in range(steps):
        key, k = random.split(key)
        model, opt_state, _, loss = step_fn(model, opt_state, None, data, k)
    dist = create_distribution(mode="static", model=model, structure_dict={
        "dependencies": structure.dependencies, "block_order": structure.block_order,
        "dims": structure.dims})
    return dist, float(loss)


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def run(n_train=4096, train_steps=2000, width=64, depth=3, lr=3e-3, n_eval=2000, n_cf=100,
        seed=0, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    k_data, k_causal, k_inf, k_post, k_do = random.split(random.PRNGKey(seed), 5)

    pgm = collider_pgm()
    data = pgm.b_sample(k_data, n_train)              # {'x1': (n, 1), 'x2': (n, 1), 'x0': (n, 1)}

    # -- compile the DAG for the two kinds of query -------------------------------------
    compiler = PGMCompiler(pgm.structure.dependencies, pgm.structure.dims)
    causal = compiler.compile_causal()
    inference = compiler.compile_inference(targets={"x1", "x2"}, conditions={"x0"})
    say("causal    order", causal.block_order, " parents", dict(causal.dependencies))
    say("inference order", inference.block_order, " parents", dict(inference.dependencies))

    # -- train one static flow per structure, on the same data ---------------------------
    causal_flow, causal_loss = train_flow(causal, data, train_steps, width, depth, lr, k_causal)
    inf_flow, inf_loss = train_flow(inference, data, train_steps, width, depth, lr, k_inf)
    say(f"final training loss: causal {causal_loss:.3f}, inference {inf_loss:.3f}")

    # -- 1. posterior p(x1, x2 | x0) from the inference-aligned flow ----------------------
    x0_obs = jnp.array([2.0])
    post = inf_flow.b_sample(k_post, n_eval, conditions={"x0": x0_obs})
    flow_mean = np.array([float(post["x1"].mean()), float(post["x2"].mean())])
    flow_corr = corr(post["x1"], post["x2"])
    exact_mean, exact_corr = float(x0_obs[0]) / (2 + NOISE ** 2), -1.0 / (1 + NOISE ** 2)
    say(f"\n1. posterior given x0 = {float(x0_obs[0]):.1f}")
    say(f"   corr(x1, x2): a priori {corr(data['x1'], data['x2']):+.2f}; "
        f"given x0: flow {flow_corr:+.2f}, exact {exact_corr:+.2f}")
    say(f"   E[x1 | x0], E[x2 | x0]: flow {flow_mean[0]:.2f}, {flow_mean[1]:.2f}; exact {exact_mean:.2f}")

    # -- 2. intervention p(x0 | do(x1 = 2)) from the causal flow --------------------------
    x1_do = jnp.array([2.0])
    x0_do = jax.vmap(lambda k: causal_flow.sample(k, targets={"x2", "x0"},
                                                 interventions={"x1": x1_do})["x0"])(
        random.split(k_do, n_eval))
    say(f"\n2. E[x0 | do(x1 = 2)]: flow {float(x0_do.mean()):.2f}, exact 2.00")

    # -- 3. counterfactuals for observed individuals ------------------------------------
    # For each individual: had x1 been 2, what would x0 have been? Exact: x0 + (2 - x1).
    factuals = {b: v[:n_cf] for b, v in data.items()}
    cf = jax.vmap(lambda d: causal_flow.counterfactual(
        d, interventions={"x1": x1_do}, targets={"x0"}, compute_log_prob=False)[0]["x0"])(factuals)
    cf_exact = factuals["x0"] + 2.0 - factuals["x1"]
    cf_err = float(jnp.mean(jnp.abs(cf - cf_exact)))
    say(f"\n3. counterfactual x0 had x1 been 2, for {n_cf} observed individuals")
    say(f"   first: observed x1 = {float(factuals['x1'][0, 0]):.2f}, x0 = {float(factuals['x0'][0, 0]):.2f}; "
        f"flow {float(cf[0, 0]):.2f}, exact {float(cf_exact[0, 0]):.2f}")
    say(f"   mean |flow - exact| = {cf_err:.2f}   (noise s.d. on x0: {NOISE})")

    return dict(causal_structure=(causal.block_order, dict(causal.dependencies)),
                inference_structure=(inference.block_order, dict(inference.dependencies)),
                losses=(causal_loss, inf_loss), flow_corr=flow_corr, exact_corr=exact_corr,
                flow_mean=flow_mean, exact_mean=exact_mean, x0_do=float(x0_do.mean()),
                x0_cf_err=cf_err)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(n_train=a.n_train, train_steps=a.train_steps, seed=a.seed)
