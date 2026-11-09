"""End-to-end StructFlow example: compile a PGM, train a structured flow, query it.

This script walks through the core workflow of the library on a small **collider**
PGM ``x1 -> y <- x2`` (two independent causes, one common effect):

1. **Define** a ground-truth structural PGM and sample training data from it.
2. **Compile** the DAG with :class:`pgms.PGMCompiler` into two block-triangular
   structures — a *causal* one (topological order, sparse) and an *inference*
   one (moralized + reordered so the observation precedes the targets). The
   moralization step adds the "explaining away" edge between the two causes,
   which is the central observation of the accompanying paper.
3. **Train** a static structured flow on the data with the ``conditional_flows``
   factories.
4. **Query** the trained flow: draw samples and evaluate ``log_prob``.
5. **Counterfactual** reasoning on the ground-truth SCM via the
   abduction--action--prediction procedure.

Run it directly::

    python examples/structflow_quickstart.py            # default settings
    python examples/structflow_quickstart.py --train-steps 1000 --n-train 2048

The defaults are deliberately small so the script finishes in a few seconds on
CPU; increase ``--train-steps`` / ``--n-train`` for a better fit.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

from structflow.pgms import make_collider_pgm, PGMCompiler
from structflow.conditional_flows import create_static_flow, create_distribution


def _fmt(d):
    """Format a dict of (possibly 1-D) arrays for printing."""
    return {k: [round(float(x), 3) for x in jnp.atleast_1d(v).ravel()] for k, v in d.items()}


def run(n_train: int = 1024, train_steps: int = 400, width: int = 32, depth: int = 2,
        learning_rate: float = 1e-3, seed: int = 0, verbose: bool = True):
    """Run the full pipeline and return a dict of results.

    Args:
        n_train: number of training examples sampled from the ground-truth PGM.
        train_steps: number of optimisation steps (full-batch).
        width, depth: size of each block's velocity network.
        learning_rate: Adam learning rate.
        seed: PRNG seed.
        verbose: print a narrated walk-through.

    Returns:
        A dict with the compiled structures, final training loss, learned and
        ground-truth sample means, and a counterfactual outcome.
    """
    key = random.PRNGKey(seed)
    model_key, data_key, train_key, eval_key = random.split(key, 4)

    def log(*a):
        if verbose:
            print(*a)

    # ------------------------------------------------------------------ 1. PGM
    # Collider: x1 -> y <- x2.  The two causes x1, x2 are marginally independent.
    pgm = make_collider_pgm(n_causes=2, mechanism_type="linear", key=model_key)
    structure = pgm.structure
    log("Ground-truth DAG dependencies:", dict(structure.dependencies))

    # --------------------------------------------------------------- 2. Compile
    compiler = PGMCompiler(structure.dependencies, structure.dims)
    causal = compiler.compile_causal()
    # Inference query: recover the causes {x1, x2} from the observed effect {y}.
    inference = compiler.compile_inference(targets={"x1", "x2"}, conditions={"y"})
    log("\nCausal structure   (order, deps):", causal.block_order, dict(causal.dependencies))
    log("Inference structure (order, deps):", inference.block_order, dict(inference.dependencies))
    log("  -> note the moralization edge: x1 now depends on x2 ('explaining away').")

    # ------------------------------------------------------------ 3. Train flow
    # Full-batch training data sampled from the ground-truth PGM.
    data_keys = random.split(data_key, n_train)
    x_batch = {b: jnp.stack([pgm.sample(k)[b] for k in data_keys])
               for b in structure.block_order}

    optimizer = optax.adam(learning_rate)
    model, step_fn = create_static_flow(
        structure=structure, depth=depth, width=width,
        model_key=model_key, optimizer=optimizer, use_coupling=False,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    normaliser = None
    loss = jnp.nan
    k = train_key
    for step in range(train_steps):
        k, step_key = random.split(k)
        model, opt_state, normaliser, loss = step_fn(model, opt_state, normaliser, x_batch, step_key)
        if verbose and (step % max(1, train_steps // 5) == 0 or step == train_steps - 1):
            log(f"  step {step:4d}  loss {float(loss):.4f}")

    # ------------------------------------------------------------- 4. Query flow
    structure_dict = {
        "dependencies": structure.dependencies,
        "block_order": structure.block_order,
        "dims": structure.dims,
    }
    dist = create_distribution(mode="static", structure_dict=structure_dict, model=model)

    sample = dist.sample(eval_key)
    logp = dist.log_prob(sample)
    log("\nLearned-flow sample:", _fmt(sample), " log_prob =", round(float(logp), 3))

    gt_samples = pgm.b_sample(random.PRNGKey(seed + 1), 2000)
    learned_samples = dist.b_sample(random.PRNGKey(seed + 2), 2000)
    gt_means = {b: float(jnp.mean(gt_samples[b])) for b in structure.block_order}
    learned_means = {b: float(jnp.mean(learned_samples[b])) for b in structure.block_order}
    log("Ground-truth means:", {k: round(v, 3) for k, v in gt_means.items()})
    log("Learned means:     ", {k: round(v, 3) for k, v in learned_means.items()})

    # ----------------------------------------------------------- 5. Counterfactual
    # "Given this observed individual, what would y have been if x1 had been 2.0?"
    factual = pgm.sample(random.PRNGKey(seed + 3))
    cf_samples, cf_logp = pgm.counterfactual(
        factual, interventions={"x1": jnp.array([2.0])}, targets={"y"})
    log("\nCounterfactual (ground-truth SCM):")
    log("  factual     :", _fmt(factual))
    log("  do(x1 = 2.0) -> y_cf =", _fmt({"y": cf_samples["y"]}))

    return {
        "causal_structure": (causal.block_order, dict(causal.dependencies)),
        "inference_structure": (inference.block_order, dict(inference.dependencies)),
        "final_loss": float(loss),
        "gt_means": gt_means,
        "learned_means": learned_means,
        "log_prob": float(logp),
        "counterfactual_y": [float(x) for x in jnp.atleast_1d(cf_samples["y"]).ravel()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(n_train=args.n_train, train_steps=args.train_steps, width=args.width,
        depth=args.depth, learning_rate=args.learning_rate, seed=args.seed)


if __name__ == "__main__":
    main()
