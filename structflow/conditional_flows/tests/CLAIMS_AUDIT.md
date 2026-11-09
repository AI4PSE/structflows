# Claims audit of the conditional-flows test suite (2026-10-21)

Done with the `claim-driven-tests` skill, after four issues reached a TMLR submission with a green suite. New tests: `test_claims.py`, plus `TestC2OTCoupling` in `sinterp/tests/test_couplings.py`.

## Why the pre-existing suite was blind

- **Degenerate fixture.** `test_numerical.py` uses an *independent* two-block structure. Dynamic conditioning is exact there, and coupling bias is zero.
- **Oracle injection.** Exact target velocities are injected, so the training objective, masks and couplings are never exercised.
- **Config mismatch.** Every learning and integration test sets `use_coupling=False`, while the experiments use the default `True`.
- **Invariant checked in the wrong place.** Triangularity is checked at init, on the static net only, never after training.
- **Weak oracles.** Many tests check smoke-level properties: not None, finite, loss decreased, parameters changed.
- **Wrong quantity.** Marginals are checked; conditionals and cross-covariances are not.

## Claims × test levels (after this audit)

| claim | formula (independent derivation) | invariant after training | population optimum | end-to-end, reported quantity | train/sample parity |
|---|---|---|---|---|---|
| C1 static conditioning is exact | — | — | `test_static_sampler_exact_conditional_with_correlated_parent` | `test_static_trained_conditional_matches_truth[False, c2ot]` | — |
| C2 dynamic conditioning (inconsistent for correlated blocks) | `test_closed_form_full_field_recovers_target` (control) | — | `test_triangular_optimum_*` (×2), `test_dynamic_sampler_reproduces_triangular_optimum_conditional` | **gap**: no trained dynamic e2e | — |
| C3 masked net is block-triangular | — | `test_dynamic_masks_unchanged_and_triangular_after_training` | — | — | — |
| C4 coupling keeps noise ⊥ condition | `TestC2OTCoupling` (7 tests) | — | — | `test_static_trained_conditional_matches_truth` (joint = xfail) | `test_removes_condition_dependence` |
| C5 guidance weight | `test_exact_guidance_weight_derivation` | — | — | **gap**: no guided-sampling e2e with a correlated prior | `test_library_guidance_weight_...` (xfail), `..._pushes_along_the_likelihood_score` |

Remaining gaps, ranked by how much they could change a reported number:
1. No trained end-to-end dynamic test on a correlated target.
2. No test runs the paper's actual benchmark configs (linear4 and linear8) against their closed-form truth.
3. No guided-sampling end-to-end test.
4. The normaliser together with coupling (coupling runs after normalisation) is untested.
5. `OneSidedField_eta1` clamps t ≤ 0.99, and its effect is unmeasured.

## Mutation check

Each mutation was applied to a scratch copy. "Old" means the pre-existing numerical, triangular-map, learning, integration and sinterp tests (127 tests).

| mutation | old suite | killed by (new) |
|---|---|---|
| M1 mask trainable (drop `stop_gradient`) | survived | `test_dynamic_masks_unchanged_and_triangular_after_training` |
| M2 static sampler drops parent values | survived | `test_static_sampler_exact_conditional_with_correlated_parent` |
| M3 dynamic sampler skips the backward inversion | survived | `test_dynamic_sampler_reproduces_triangular_optimum_conditional`, but only after giving y variance 2. With y ~ N(0, 1) the y-flow is the identity and the mutation survived. |
| M4 C²OT ignores the condition term | survived | `test_huge_w_is_identity_for_distinct_conditions`, `test_removes_condition_dependence` |
| M5 `'c2ot'` wired with no conditions | survived | `test_static_trained_conditional_matches_truth[c2ot]` (expensive only) |
| M6 guidance weight sign flipped | survived | `test_library_guidance_pushes_along_the_likelihood_score`. It was added because the xfail weight test masked the mutation. |

The old suite killed 0 of 6 mutations; the new tests kill 6 of 6. Expensive tests run with `--run-expensive`, taking about 20 s each on GPU.
