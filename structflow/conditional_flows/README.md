# Conditional flows

Flows that represent conditional distributions over a block-triangular structure. Every
distribution implements

- `sample(key, conditions: Dict[str, Array], targets: Set[str])`
- `log_prob(data: Dict[str, Array], targets: Set[str], conditions: Set[str])`
- `counterfactual(data, interventions, targets)` (abduction, action, prediction)

with batched versions `b_sample`, `b_log_prob`, `b_counterfactual`.

| folder | mode | distribution | conditioning |
| :--- | :--- | :--- | :--- |
| `static/` | `static` | `StaticConditionalDistribution` | each block is integrated to completion, then its final value is fixed context for its children |
| `dynamic/` | `dynamic` | `DynamicDistribution` | all blocks share one ODE solve and see their parents' co-evolving states; see [`dynamic/README.md`](dynamic/README.md) for the two inference implementations |
| `guidance/` | `guided` | `GuidanceBasedDistribution` | one unstructured joint flow; arbitrary queries by reconstruction guidance |
| `carefl/` | `carefl` | `CAREFLDistribution` | affine autoregressive flow (CAREFL baseline) |
| `dcm/` | `dcm` | `DCMDistribution` | per-node denoising diffusion with DDIM (DCM baseline) |

Build a trainable model with the factories in `flow_factory.py` (`create_static_flow`,
`create_dynamic_flow`, `create_joint_flow`, `create_carefl_flow`, `create_dcm_flow`,
`create_universal_flow`), then wrap the trained model with
`create_distribution(mode, structure_dict, model)` from `dist_factory.py`.
