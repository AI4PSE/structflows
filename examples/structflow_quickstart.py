"""Worked example: one collider PGM, two compiled flows, three kinds of query.

The graph is a collider with two independent causes and one common effect,

    x1 --> y <-- x2,        x1, x2 ~ N(0, 1),   y = x1 + x2 + 0.5 * eps,   eps ~ N(0, 1).

The compiler turns the DAG plus a query into a block-triangular flow structure:

* ``compile_causal()`` keeps the DAG order ``[x1, x2, y]``. A flow trained with it is a
  structural causal model, so it answers interventional and counterfactual queries.
* ``compile_inference(targets={x1, x2}, conditions={y})`` puts the observation first and
  *moralizes* the graph: observing the common effect couples its causes, so the inference
  structure gains an edge x2 -> x1 ("explaining away"). A flow trained with it samples the
  posterior p(x1, x2 | y) directly, with no MCMC.

Both flows are trained on the same samples, and each answer is checked against its closed
form (the model is linear-Gaussian):

1. posterior given y:  E[x1 | y] = E[x2 | y] = y / 2.25,  corr(x1, x2 | y) = -0.8,
   although x1 and x2 are independent a priori;
2. intervention:       E[y | do(x1 = 2)] = 2;
3. counterfactual:     had x1 been 2, this individual's y would have been y + (2 - x1).

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

NOISE = 0.5                                    # std of the noise on y


def collider_pgm():
    """x1, x2 ~ N(0, 1); y = x1 + x2 + NOISE * eps."""
    structure = BlockTriangularStructure(block_order=["x1", "x2", "y"],
                                         dependencies={"x1": [], "x2": [], "y": ["x1", "x2"]},
                                         dims={"x1": 1, "x2": 1, "y": 1})
    root = lambda: LinearMechanism(parent_names=[], input_dim=0, output_dim=1,
                                   A=jnp.zeros((1, 0)), B=jnp.eye(1), bias=jnp.zeros(1))
    effect = LinearMechanism(parent_names=["x1", "x2"], input_dim=2, output_dim=1,
                             A=jnp.ones((1, 2)), B=NOISE * jnp.eye(1), bias=jnp.zeros(1))
    return StructuralPGM(structure, {"x1": root(), "x2": root(), "y": effect})


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


def run(n_train=4096, train_steps=2000, width=64, depth=3, lr=3e-3, n_eval=2000, seed=0,
        verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    k_data, k_causal, k_inf, k_post, k_do = random.split(random.PRNGKey(seed), 5)

    pgm = collider_pgm()
    data = pgm.b_sample(k_data, n_train)              # {'x1': (n, 1), 'x2': (n, 1), 'y': (n, 1)}

    # -- compile the DAG for the two kinds of query -------------------------------------
    compiler = PGMCompiler(pgm.structure.dependencies, pgm.structure.dims)
    causal = compiler.compile_causal()
    inference = compiler.compile_inference(targets={"x1", "x2"}, conditions={"y"})
    say("causal    order", causal.block_order, " parents", dict(causal.dependencies))
    say("inference order", inference.block_order, " parents", dict(inference.dependencies))

    # -- train one static flow per structure, on the same data ---------------------------
    causal_flow, causal_loss = train_flow(causal, data, train_steps, width, depth, lr, k_causal)
    inf_flow, inf_loss = train_flow(inference, data, train_steps, width, depth, lr, k_inf)
    say(f"final training loss: causal {causal_loss:.3f}, inference {inf_loss:.3f}")

    # -- 1. posterior p(x1, x2 | y) from the inference-aligned flow ----------------------
    y_obs = jnp.array([2.0])
    post = inf_flow.b_sample(k_post, n_eval, conditions={"y": y_obs})
    flow_mean = np.array([float(post["x1"].mean()), float(post["x2"].mean())])
    flow_corr = corr(post["x1"], post["x2"])
    exact_mean, exact_corr = float(y_obs[0]) / (2 + NOISE ** 2), -1.0 / (1 + NOISE ** 2)
    say(f"\n1. posterior given y = {float(y_obs[0]):.1f}")
    say(f"   corr(x1, x2): a priori {corr(data['x1'], data['x2']):+.2f}; "
        f"given y: flow {flow_corr:+.2f}, exact {exact_corr:+.2f}")
    say(f"   E[x1 | y], E[x2 | y]: flow {flow_mean[0]:.2f}, {flow_mean[1]:.2f}; exact {exact_mean:.2f}")

    # -- 2. intervention p(y | do(x1 = 2)) from the causal flow --------------------------
    x1_do = jnp.array([2.0])
    y_do = jax.vmap(lambda k: causal_flow.sample(k, targets={"x2", "y"},
                                                 interventions={"x1": x1_do})["y"])(
        random.split(k_do, n_eval))
    say(f"\n2. E[y | do(x1 = 2)]: flow {float(y_do.mean()):.2f}, exact 2.00")

    # -- 3. counterfactual for one observed individual -----------------------------------
    factual = {b: v[0] for b, v in data.items()}
    cf, _ = causal_flow.counterfactual(factual, interventions={"x1": x1_do}, targets={"y"})
    cf_exact = float(factual["y"][0] + 2.0 - factual["x1"][0])
    say(f"\n3. observed x1 = {float(factual['x1'][0]):.2f}, x2 = {float(factual['x2'][0]):.2f}, "
        f"y = {float(factual['y'][0]):.2f}")
    say(f"   had x1 been 2, y would have been: flow {float(cf['y'][0]):.2f}, exact {cf_exact:.2f}")

    return dict(causal_structure=(causal.block_order, dict(causal.dependencies)),
                inference_structure=(inference.block_order, dict(inference.dependencies)),
                losses=(causal_loss, inf_loss), flow_corr=flow_corr, exact_corr=exact_corr,
                flow_mean=flow_mean, exact_mean=exact_mean, y_do=float(y_do.mean()),
                y_cf=(float(cf["y"][0]), cf_exact))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(n_train=a.n_train, train_steps=a.train_steps, seed=a.seed)
