# Conditional Flows Unified Test Suite

Unified, parametrized test suite for conditional distributions across all modes: **static**, **dynamic**, **joint**, and **carefl**.

## Overview

This test suite refactors scattered mode-specific tests into a generalized framework, reducing duplication and ensuring consistent coverage across all distribution types.

### File Organization

- **`conftest.py`**: Shared pytest fixtures and parametrization
  - Mode parametrization (static, dynamic, joint, carefl)
  - Structure fixtures (chain, fork, collider)
  - Distribution and flow creation fixtures
  - PGM and data sampling fixtures

- **`test_utils.py`**: Shared utility functions
  - Jacobian triangularity verification
  - Loss and learning assertions
  - Gradient magnitude checking
  - Data flattening/unflattening

- **`test_structure.py`**: Jacobian block-triangularity tests
  - Verify velocity has correct sparsity pattern
  - Check upper-triangular part is zero
  - Validate dependencies match Jacobian structure

- **`test_api.py`**: API completeness and correctness
  - Sample and batch sampling operations
  - Log probability computation (tractable queries)
  - Inverse flow (transform_to_noise)
  - Counterfactual reasoning
  - Gradient flow verification

- **`test_learning.py`**: Gradient and learning capability
  - Non-zero gradient verification
  - Parameter update validation
  - Simple PGM fitting (2-node chain, 20 steps)
  - Stability across multiple steps

- **`test_integration.py`**: Factory function integration
  - `create_distribution()` across all modes/structures
  - `create_flow()` + step_fn pairs
  - Normaliser wrapping and Jacobian correction
  - Cross-factory compatibility

## Distribution Modes

### Static (StaticConditionalDistribution)

- **Flow Type**: Autoregressive block-by-block integration
- **Jacobian**: ✅ Block-triangular (testable)
- **Zero Velocity**: ✅ Identity flow
- **API Support**: ✅ Full (all tractable queries)
- **Learnable**: ✅ Yes
- **Key Property**: Sequential sampling for each block

### Dynamic/Parallel (ParallelBlockTriangularDistribution)

- **Flow Type**: Parallel integration with dependency masking
- **Jacobian**: ✅ Block-triangular (testable)
- **Zero Velocity**: ✅ Identity flow
- **API Support**: ✅ Full (all tractable queries)
- **Learnable**: ✅ Yes
- **Key Property**: Efficient parallel processing while maintaining structure

### Joint/Guidance (GuidanceBasedDistribution)

- **Flow Type**: Joint flat flow + reconstruction guidance
- **Jacobian**: ❌ No explicit block structure
- **Zero Velocity**: ✅ Identity flow
- **API Support**: ⚠️ Limited (joint queries only)
  - `log_prob()` requires `targets + conditions = all_vars`
  - `sample()` works for any query via guidance
  - `counterfactual()` requires full specification
- **Learnable**: ✅ Yes
- **Key Property**: Flexible guidance for arbitrary queries

### CAREFL (CAREFLDistribution)

- **Flow Type**: Affine autoregressive (not velocity-based)
- **Jacobian**: ✅ Block-triangular structure (affine preserves sparsity)
- **Zero Velocity**: ❌ Not applicable (affine, not flow)
- **API Support**: ✅ Full (all tractable queries)
- **Learnable**: ✅ Yes
- **Key Property**: Identifiable through affine structure

## Test Parametrization

### Modes Tested in Each Suite

| Test Suite | Static | Dynamic | Joint | CAREFL |
|-----------|--------|---------|-------|--------|
| test_structure | ✅ | ✅ | ❌ | ✅ |
| test_api | ✅ | ✅ | ⚠️ | ✅ |
| test_learning | ✅ | ✅ | ✅ | ✅ |
| test_integration | ✅ | ✅ | ✅ | ✅ |

**Legend**: ✅ Full support, ⚠️ Limited support (mode-specific), ❌ Not applicable

## Test Fixtures

### Primary Fixtures (conftest.py)

```python
@pytest.fixture(params=["static", "dynamic", "joint", "carefl"])
def mode(request):
    """Parametrize tests across all distribution modes."""
    return request.param

@pytest.fixture
def dist(mode, simple_pgm, pgm_key):
    """Create distribution for the given mode."""
    # Returns appropriate ConditionalDistribution

@pytest.fixture
def flow_and_step_fn(mode, simple_pgm, pgm_key):
    """Create flow model and training step function."""
    # Returns (model, step_fn, structure)

@pytest.fixture
def sample_data_batch(simple_pgm, pgm_key):
    """Sample 32-sample batch from simple 2-node chain PGM."""
    # Returns {block: (batch=32, dim)} dict
```

### PGM Fixture

- **Structure**: 2-node linear chain (x0 -> x1)
- **Mechanism**: Linear Gaussian
- **Dimensions**: 1-dimensional each
- **Purpose**: Simple, fast testing of learning and sampling

## Running Tests

### Run All Tests

```bash
pytest structflow/conditional_flows/tests/ -v
```

### Run Specific Suite

```bash
pytest structflow/conditional_flows/tests/test_structure.py -v
pytest structflow/conditional_flows/tests/test_api.py -v
pytest structflow/conditional_flows/tests/test_learning.py -v
pytest structflow/conditional_flows/tests/test_integration.py -v
```

### Run Tests for Specific Mode

```bash
pytest structflow/conditional_flows/tests/ -v -k "static"
pytest structflow/conditional_flows/tests/ -v -k "dynamic"
pytest structflow/conditional_flows/tests/ -v -k "joint"
pytest structflow/conditional_flows/tests/ -v -k "carefl"
```

### Run Tests for Specific Structure

```bash
pytest structflow/conditional_flows/tests/test_structure.py -v -k "chain"
pytest structflow/conditional_flows/tests/test_api.py -v -k "fork"
```

## Test Coverage Summary

### Jacobian Structure Tests

- Verify velocity Jacobian is block-triangular
- Check that upper-triangular (future blocks) is zero
- Validate dependencies match Jacobian non-zeros
- Chain-rule validation for sequential dependencies

**Modes**: static, dynamic, carefl

### API Tests

- **Sampling**: unconditional, conditional, batched
- **Log Probability**: unconditional, conditional (tractable queries), batched
- **Inverse Flow**: transform_to_noise (data -> noise), batched
- **Counterfactual**: single/multiple interventions, with targets, batched
- **Gradient Flow**: through sample, log_prob
- **Consistency**: same key = same samples, different key = different samples

**Modes**: All

### Learning Tests

- Non-zero gradient computation
- Parameter changes after update
- Simple PGM fitting: 2-node chain in 20 steps with lr=0.01
- Loss should decrease to <95% of initial
- Stability: 10+ steps without explosion/vanishing
- Gradient magnitude: within reasonable bounds

**Modes**: All

### Integration Tests

- Factory creation: `create_distribution()` all modes/structures
- Flow creation: `create_flow()` returns valid model + step_fn
- Normaliser wrapping: DictNormaliser applied correctly
- Jacobian correction: log_prob adjusted by log|det J|
- Cross-factory compatibility: static flow + dist_factory work together

**Modes**: All

## Mode-Specific Behavior

### Static Mode Specifics

- Autoregressive sampling: sample x0, then x1 | x0
- Velocity function has explicit block structure
- Full log_prob support for all tractable queries
- Clean Jacobian triangularity due to sequential structure

### Dynamic Mode Specifics

- Parallel integration with masked dependencies
- Velocity function maintains block sparsity via masks
- Same log_prob support as static
- Jacobian triangularity due to parallel mask structure

### Joint/Guidance Mode Specifics

- Flat joint representation (no block structure in velocity)
- Guidance enables arbitrary conditional queries
- **Limitation**: `log_prob()` only supports p(all_vars)
  - To get p(targets | conditions), use counterfactual instead
- No explicit Jacobian triangularity test
- Guidance scale parameter affects reconstruction

### CAREFL Mode Specifics

- Affine autoregressive: x_j = exp(s_j(pa_j)) * z_j + t_j(pa_j)
- Block-triangular structure in affine parameters
- Full log_prob and counterfactual support
- **Note**: Zero velocity test not applicable (affine, not flow)
- Identifiability constraint: Laplace base distribution

## Known Limitations

1. **Joint/Guidance Mode**:
   - Cannot compute arbitrary conditional log_prob directly
   - Use counterfactual query for p(targets | conditions)
   - Guidance-based sampling is more flexible than log_prob

2. **Parallel Mode**:
   - Requires careful masking to maintain triangularity
   - Jacobian verification uses masked velocity

3. **CAREFL Mode**:
   - Not flow-based, so zero velocity test skipped
   - Affine transforms may have numerical sensitivity
   - Requires Laplace base distribution for identifiability

## Assertion Utilities

### Jacobian Triangularity

```python
from test_utils import assert_triangular_jacobian

assert_triangular_jacobian(
    jac,                      # (n, n) Jacobian matrix
    structure.block_order,    # ["x0", "x1", ...]
    structure.dependencies,   # {"x0": [], "x1": ["x0"], ...}
    structure.dims,           # {"x0": 1, "x1": 1, ...}
    atol=1e-5
)
```

### Learning Progress

```python
from test_utils import assert_loss_decreased

assert_loss_decreased(loss_before, loss_after, min_ratio=0.9)
# Asserts: loss_after < 0.9 * loss_before
```

### Gradient Quality

```python
from test_utils import assert_gradients_nonzero, assert_parameters_changed

assert_gradients_nonzero(grads, threshold=1e-7)
assert_parameters_changed(params_before, params_after, threshold=1e-7)
```

## Extending the Tests

### Adding a New Mode

1. Implement mode-specific fixture in `conftest.py`
2. Add mode to `MODES` constant
3. Update mode-specific skip logic in test files
4. Run full test suite to verify integration

### Adding a New Distribution Operation

1. Add test to `test_api.py` with `@pytest.mark.parametrize("mode", MODES)`
2. Implement mode-specific behavior if needed
3. Add assertion utility to `test_utils.py` if reusable
4. Document in README

### Adding a New Structure

1. Add fixture to `conftest.py`
2. Add to `STRUCTURE_TYPES` parametrization
3. Tests parametrized by structure will automatically run
4. Update README if structure has specific considerations

## Performance Notes

- Simple 2-node PGM tests run in <1s each
- Full suite (all modes × all test suites) takes ~5-10 minutes
- Jacobian tests (jacfwd) are slower (~1-2s per test)
- Learning tests depend on convergence (10-20 steps)

## Debugging Failed Tests

### Typical Failure Modes

1. **"Jacobian has upper triangular non-zero"**
   - Check velocity function respects block dependencies
   - Verify topological ordering of blocks

2. **"Loss did not decrease"**
   - Learning rate may be too small
   - PGM batch may be too small
   - Check step_fn implementation for bugs

3. **"Gradients are zero"**
   - Check loss function is differentiable
   - Verify model parameters are not frozen
   - Check for NaN/Inf in forward pass

4. **"Intractable query error"**
   - Verify targets+conditions satisfy parent/descendant rules
   - For joint mode, ensure targets + conditions covers all variables

## References

- Main refactoring plan: `/plans/conditional_flows_test_refactoring.md`
- Distribution API: `/structflow/conditional_flows/dist.py`
- Factory functions: `/structflow/conditional_flows/dist_factory.py`
- Flow factory: `/structflow/conditional_flows/flow_factory.py`
