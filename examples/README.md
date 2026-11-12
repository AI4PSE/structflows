# Examples

Self-contained, runnable examples for the library, kept working and covered by the
test suite.

## `structflow_quickstart.py`

A worked example on a collider `x1 -> x0 <- x2` with explicit linear-Gaussian equations
(`x1, x2 ~ N(0, 1)`, `x0 = x1 + x2 + 0.5 eps`), so every answer has a closed form:

1. **Compile** the DAG twice with `pgms.PGMCompiler`: a *causal* structure (DAG order) and
   an *inference* structure for `p(x1, x2 | x0)`, which moralizes the collider by adding the
   "explaining away" edge `x2 -> x1`.
2. **Train** one static structured flow per structure on the same data.
3. **Query** and check against the closed form: the posterior `p(x1, x2 | x0)` (the causes
   become correlated, -0.8, once x0 is observed), the intervention `E[x0 | do(x1 = 2)]`, and
   a counterfactual for one observed individual.

```bash
python examples/structflow_quickstart.py        # about a minute on a laptop CPU
```

The accompanying `test_structflow_quickstart.py` runs the same pipeline with tiny settings
as a fast smoke test (`pytest examples/`).
