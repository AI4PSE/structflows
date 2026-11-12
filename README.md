# StructFlow

Compile a probabilistic graphical model (PGM) and a query into a structured,
block-triangular continuous normalizing flow, in
[JAX](https://github.com/google/jax) / [Equinox](https://github.com/patrick-kidger/equinox).
This is the library behind the paper
**"Structured Flows for Causal Modelling and Statistical Inference"**
(Telfar & Sandfeld, TMLR 2026, [OpenReview](https://openreview.net/forum?id=5cFwygAUFt)).
The experiments, evaluation and LaTeX source of the paper live in
[structflows-tmlr](https://github.com/AI4PSE/structflows-tmlr).

- **Causal alignment** keeps the DAG factorization: the trained flow is a
  structural causal model, so it answers interventional and counterfactual
  queries.
- **Inference alignment** re-orders and moralizes the graph for amortized
  posterior inference.
- Conditions enter each block **statically** (integrated to completion, then
  fixed) or **dynamically** (co-evolving with the targets in one parallel ODE
  solve). Dynamic conditioning is consistent only with a per-block
  (Knothe–Rosenblatt) noise–data coupling, `use_coupling='c2ot'`; see the
  paper's appendix.

## Packages

| package | purpose |
| :--- | :--- |
| [`structflow.pgms`](structflow/pgms/README.md) | synthetic PGM benchmarks with known structural equations; the compiler from a DAG to causal / inference block structures |
| [`structflow.conditional_flows`](structflow/conditional_flows/README.md) | conditional flows `p(targets \| conditions)`: `static`, `dynamic` (parallel), `guided` (reconstruction guidance), and the CAREFL / DCM baselines; factories `create_*_flow`, `create_distribution` |
| [`structflow.bayes`](structflow/bayes/README.md) | distribution objects, NUTS reference posteriors, metrics (SWD, Sinkhorn, KL) |

Dependency maintained separately: [`sinterp-jax`](https://github.com/act65/sinterp),
which provides two packages, `sinterp` (interpolants and couplings, including the
C²OT / per-block coupling) and `flow` (ODE/SDE integration).

## Install

Install JAX for your hardware first (`pip install "jax[cpu]"` or
`"jax[cuda12]"`), then

```bash
pip install "structflow-jax @ git+https://github.com/AI4PSE/structflows@v0.1.0"   # import structflow
```

This pulls `sinterp-jax` from its GitHub tag. From a clone:
`pip install -e ".[dev]"`.

## Quickstart

```python
import jax
from structflow.pgms import make_chain_pgm

key = jax.random.PRNGKey(0)
pgm = make_chain_pgm(n_nodes=5, mechanism_type="linear", key=key)
sample = pgm.sample(key)                                              # {'x0': ..., 'x1': ..., ...}
logp = pgm.log_prob(sample, targets={'x1'}, conditions={'x0'})        # p(x1 | x0)
```

Exact `log_prob` exists only for tractable (ancestral) queries; others raise
`NotImplementedError`. A complete example, which compiles a collider PGM,
trains a structured flow and answers causal, inference and counterfactual
queries, is in [`examples/`](examples/):

```bash
python examples/structflow_quickstart.py
```

## Tests

```bash
JAX_PLATFORMS=cpu pytest -m "not slow"     # fast subset
pytest --run-expensive                      # everything (a GPU helps)
```

`structflow/conditional_flows/tests/test_claims.py` holds regression tests for
the paper's claims (triangular masks stay triangular after training, coupling
parity, dynamic-conditioning consistency under the per-block coupling).

## Citation

```bibtex
@article{telfar2026structured,
  title   = {Structured Flows for Causal Modelling and Statistical Inference},
  author  = {Telfar, Alexander and Sandfeld, Stefan},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=5cFwygAUFt}
}
```

## License

MIT, see [LICENSE](LICENSE).
