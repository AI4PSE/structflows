# Examples

Self-contained, runnable examples for the library. Unlike
`notebooks-and-scripts/` (exploratory, untested), these are kept working and are
covered by the test suite.

## `structflow_quickstart.py`

An end-to-end walk-through of the core StructFlow workflow on a small collider
PGM `x1 -> y <- x2`:

1. **Define** a ground-truth structural PGM and sample training data.
2. **Compile** the DAG into causal- and inference-aligned block-triangular
   structures with `pgms.PGMCompiler`. The inference compilation *moralizes* the
   collider, adding the "explaining away" edge between the two causes.
3. **Train** a static structured flow with the `conditional_flows` factories.
4. **Query** the trained flow (`sample`, `log_prob`).
5. **Counterfactual** reasoning on the ground-truth SCM.

```bash
# Defaults finish in a few seconds on CPU.
python examples/structflow_quickstart.py

# A better fit (slower):
python examples/structflow_quickstart.py --train-steps 1000 --n-train 2048
```

The accompanying `test_structflow_quickstart.py` runs the same pipeline with
tiny settings as a fast smoke test (`pytest examples/`).
