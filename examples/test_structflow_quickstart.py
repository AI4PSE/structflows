"""Smoke test for the worked example in ``examples/structflow_quickstart.py``.

Runs the full compile -> train -> query -> counterfactual pipeline with tiny
settings so it stays fast, and checks the example's key claims hold:
the inference compilation moralizes the collider, training produces a finite
loss, and queries return finite values.
"""

import importlib.util
import pathlib

import jax
import numpy as np

# Import the sibling example module by path so the test does not depend on how
# pytest is invoked or on sys.path layout.
_spec = importlib.util.spec_from_file_location(
    "structflow_quickstart",
    pathlib.Path(__file__).resolve().parent / "structflow_quickstart.py",
)
quickstart = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quickstart)


def test_quickstart_runs_and_moralizes():
    result = quickstart.run(n_train=64, train_steps=20, width=16, depth=2,
                            seed=0, verbose=False)

    # Training produced a finite loss and finite log-probability.
    assert np.isfinite(result["final_loss"])
    assert np.isfinite(result["log_prob"])

    # Causal structure leaves the collider parents independent ...
    _, causal_deps = result["causal_structure"]
    assert causal_deps["x1"] == [] and causal_deps["x2"] == []

    # ... while the inference compilation adds the moralization edge between the
    # two causes (explaining away): one cause now conditions on the other.
    inf_order, inf_deps = result["inference_structure"]
    assert inf_order[0] == "y", "observation must precede targets in inference order"
    moral_edge = ("x2" in inf_deps.get("x1", [])) or ("x1" in inf_deps.get("x2", []))
    assert moral_edge, f"expected a moralization edge between x1 and x2, got {inf_deps}"

    # Counterfactual returns a finite outcome for y.
    assert all(np.isfinite(v) for v in result["counterfactual_y"])
