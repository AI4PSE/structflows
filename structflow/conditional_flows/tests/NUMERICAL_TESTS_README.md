# Numerical Tests for Conditional Flows

## Overview

The numerical test suite (`test_numerical.py`) verifies that conditional distributions produce **statistically correct samples and log probabilities** when the velocity function is exactly known.

### Core Strategy

1. Use [`exact_si_gaussian.py`](../sinterp/exact_si_gaussian.py) to construct **oracle velocity functions** that exactly transform:
   - Base distribution: **N(0, I)** at t=0
   - Target distribution: **N(m₂, C₂)** at t=1

2. Inject these oracle velocities into distribution classes via **mode-specific adapters**

3. Verify two key properties:
   - **Sample statistics**: `mean(samples) ≈ m₂` and `var(samples) ≈ diag(C₂)`
   - **Log probability**: `log_prob(x)` matches analytical Gaussian density

## Test Configuration

### Variables

Two independent 1D variables with different target distributions:

```python
TARGET_PARAMS = {
    'x0': {'mean': 1.5, 'var': 2.0},      # N(1.5, 2.0)
    'x1': {'mean': -0.5, 'var': 0.5},     # N(-0.5, 0.5)
}
```

### Structure

```python
INDEPENDENT_STRUCTURE = BlockTriangularStructure(
    block_order=['x0', 'x1'],
    dependencies={'x0': [], 'x1': []},    # No dependencies
    dims={'x0': 1, 'x1': 1}
)
```

### Integration

- **Number of steps**: 100
- **Interpolant**: LinearDeterministic (simplest, most direct)
- **Tolerance**: 1e-3 (numerical integration error)

## Mode-Specific Adapters

Each distribution mode expects a different velocity function interface. The adapter classes bridge this gap.

### 1. Static Mode: `ExactGaussianStaticVelocity`

**Mode interface requirement:**
```python
velocity_fn.structure          # BlockTriangularStructure
velocity_fn.blocks            # Dict[str, Callable]: block velocity functions
velocity_fn(x_dict, t, active_blocks)  # Full velocity call
```

**Block velocity signature:**
```python
blocks[name](t, x_block, *parents) -> velocity
```

**Implementation:**

```python
class ExactGaussianStaticVelocity(eqx.Module):
    structure: BlockTriangularStructure
    v_fns: Dict[str, Callable]  # (x, t) -> v
    
    @property
    def blocks(self):
        # Wrap each velocity function to match block signature
        return {name: ExactBlockVelocity(self.v_fns[name]) 
                for name in self.structure.block_order}
    
    def __call__(self, x_dict, t, active_blocks=None):
        # Compute velocity for each block independently
        return {name: self.v_fns[name](x_dict[name], t) 
                for name in active_blocks or self.structure.block_order}
```

**How it works:**
- `blocks` property lazily wraps each scalar velocity function
- `ExactBlockVelocity` adapter reorders arguments: `(t, x, *parents) -> (x, t)`
- For independent variables, parents are ignored
- `__call__` delegates to individual block functions

**Limitations:**
- ✅ Supports independent variables (no parent dependencies)
- ⚠️ Parent dependencies would require proper context passing (TODO)

### 2. Dynamic/Parallel Mode: `ExactGaussianParallelVelocity`

**Mode interface requirement:**
```python
velocity_fn.structure              # BlockTriangularStructure
velocity_fn.block_offsets          # Dict[str, (int, int)]: block slices
velocity_fn(x_vec, t) -> v_vec     # Flat vector operations
```

**Implementation:**

```python
class ExactGaussianParallelVelocity(eqx.Module):
    structure: BlockTriangularStructure
    v_fns: Dict[str, Callable]     # (x, t) -> v
    block_offsets: Dict[str, tuple] # (start, end) indices in flat vector
    
    def __init__(self, structure, v_fns):
        self.structure = structure
        self.v_fns = v_fns
        # Compute block_offsets from structure
        offsets = {}
        idx = 0
        for name in structure.block_order:
            dim = structure.dims[name]
            offsets[name] = (idx, idx + dim)
            idx += dim
        self.block_offsets = offsets
    
    def __call__(self, x_vec, t):
        # Split, compute, concatenate
        v_parts = []
        for name in self.structure.block_order:
            start, end = self.block_offsets[name]
            x_block = x_vec[start:end]
            v_block = self.v_fns[name](x_block, t)
            v_parts.append(v_block)
        return jnp.concatenate(v_parts)
```

**How it works:**
- Maintains block offsets for efficient slicing
- Splits flat vector by block using offsets
- Computes velocity for each block independently
- Concatenates results back into flat vector

**Limitations:**
- ✅ Works for arbitrary variable dimensions
- ✅ Efficient flat vector operations
- ⚠️ Currently assumes no dependencies (similar to static)

### 3. Guided Mode: `ExactGaussianGuidedVelocity`

**Mode interface requirement:**
```python
velocity_fn.get_velocity_b(x, t) -> velocity  # Main method
# x can be dict or flat vector (depends on implementation)
```

**Implementation:**

```python
class ExactGaussianGuidedVelocity(eqx.Module):
    v_fns: Dict[str, Callable]     # (x, t) -> v
    key_order: List[str]           # Variable ordering
    
    def get_velocity_b(self, x, t):
        if isinstance(x, dict):
            # Dict input: compute velocity for each variable
            return {name: self.v_fns[name](x[name], t) 
                    for name in self.key_order if name in x}
        else:
            # Flat vector input (TODO): requires flattening logic
            raise NotImplementedError(
                "Guided velocity adapter currently expects dict input"
            )
```

**How it works:**
- Simple dict-to-dict mapping of velocities
- Each variable's velocity computed independently
- `key_order` ensures consistent ordering

**Limitations:**
- ✅ Dict-based interface natural for guided flows
- ⚠️ No flat vector support yet (GuidedFlow sometimes uses flat vecs)
- ⚠️ No dependency support

### 4. CAREFL Mode: **Not Applicable**

CAREFL is **not velocity-based** - it uses affine autoregressive transforms:

```
x_j = exp(s_j(pa_j)) * z_j + t_j(pa_j)
```

The exact Gaussian velocity approach doesn't apply. Testing CAREFL would require:
- Direct affine transform properties (invertibility, Jacobian)
- Different oracle construction (not from `exact_si_gaussian.py`)
- Separate test suite

**Decision**: Exclude CAREFL from numerical velocity tests.

## Test Cases

### 1. Sample Mean Test (`test_sample_mean_matches_target`)

**What it tests:**
- Does the empirical sample mean converge to the target mean?

**Statistical tolerance:**
```
tolerance = 3 * stderr = 3 * sigma / sqrt(n)
```

With n=2000 samples, tolerance ≈ 0.05 for σ=1.

**Why it works:**
- Central Limit Theorem guarantees that sample mean ~ N(μ, σ²/n)
- 3σ rule captures ~99.7% of the distribution

### 2. Sample Variance Test (`test_sample_variance_matches_target`)

**What it tests:**
- Does the empirical sample variance match the target variance?

**Tolerance:**
```
rtol = 20%  # Relative tolerance
```

**Why this tolerance:**
- Variance estimation has higher variance than mean
- Chi-squared distribution: Var(sample_var) ≈ (μ₄ - σ⁴) / n
- For Gaussian: 20% is reasonable with n=2000

### 3. Log Probability Tests

Three variants test different regimes:

#### At the Means (`test_log_prob_matches_analytical_at_means`)
```
x = (1.5, -0.5)  # At target means
```
- Highest density point
- Most numerically stable test

#### At the Origin (`test_log_prob_matches_analytical_at_origin`)
```
x = (0, 0)  # Away from both means
```
- Tests tail regime
- Checks proper variance scaling

#### Away from Means (`test_log_prob_matches_analytical_away_from_means`)
```
x = (1.5 + 2σ₀, -0.5 - 2σ₁)  # 2 std devs away
```
- Far tail regime
- Validates log density falloff

#### Batched Test (`test_batched_log_prob_matches_analytical`)
```
samples ~ N(m₂, C₂)  # Draw 50 samples
lp = b_log_prob(samples)
```
- Verifies batch consistency
- Tests vectorized operations

### 4. Log Probability Ordering Test (`test_log_prob_ordering`)

**What it tests:**
- Does log_prob respect the density gradient?

**Property:**
```
log p(x_at_mean) > log p(x_at_1σ) > log p(x_at_2σ)
```

**Why it works:**
- Sanity check: higher density should give higher log_prob
- Catches bugs in score accumulation

### 5. Forward-Backward Consistency (`test_transform_to_noise_inverse_property`)

**What it tests:**
- Is `transform_to_noise` approximately invertible?

**Property:**
```
z = transform_to_noise(x)
# z should be approximately N(0, I)
# Check: ||z|| < 5 (reasonable magnitude for standard normal)
```

## Integration Tolerance Analysis

The tolerance of **1e-3** for log_prob tests accounts for:

1. **ODE integration error** (primary):
   - With n_steps=100 and linear interpolant
   - Expected error: ~1e-3 to 1e-4

2. **Numerical differentiation** (divergence computation):
   - JVP of velocity field
   - Automatic differentiation error: ~1e-6

3. **Accumulation over trajectory**:
   - Integration from t=0 to t=1 accumulates error
   - Total: ~1e-3 is realistic

**If tests fail with atol=1e-3:**
- Increase n_steps to 200 (but slower)
- Check divergence computation for bugs
- Verify interpolant is being used correctly

## How to Run the Tests

### Run all numerical tests:
```bash
pytest structflow/conditional_flows/tests/test_numerical.py -v
```

### Run for specific mode:
```bash
pytest structflow/conditional_flows/tests/test_numerical.py -v -k "static"
pytest structflow/conditional_flows/tests/test_numerical.py -v -k "dynamic"
pytest structflow/conditional_flows/tests/test_numerical.py -v -k "guided"
```

### Run specific test:
```bash
pytest structflow/conditional_flows/tests/test_numerical.py::test_sample_mean_matches_target -v
```

## Expected Behavior

All tests should **pass** for modes with working implementations:
- ✅ **Static**: Full support expected
- ✅ **Dynamic**: Full support expected  
- ⚠️ **Guided**: Should work but may need integration steps tuning
- ❌ **CAREFL**: Not included (non-velocity-based)

If a mode fails:
1. Check `n_steps` (may need 200+ for some modes)
2. Check that `b_log_prob` is implemented (required for tests)
3. Check velocity function returns correct shapes
4. Enable debug output and compare trajectories

## Future Improvements

### 1. Add Conditional Queries
```python
# Test p(x1 | x0) for chain structure
test_conditional_mean_given_parent()
test_conditional_variance_given_parent()
test_conditional_log_prob_consistency()
```

### 2. Add Non-Independent Variables
```python
# Test chain structure: x0 -> x1
# where x1 | x0 ~ N(a*x0 + b, σ²)
test_chain_structure()
```

### 3. Support Dependent Variables in Adapters
```python
# Currently: v_fns[name](x, t)
# Future: v_fns[name](x, t, parents_dict)
```

### 4. CAREFL Testing
- Separate test suite for affine transforms
- Verify Jacobian properties
- Check identifiability constraints

### 5. Multi-dimensional Variables
```python
# Currently: 1D only
# Future: 5D or 10D variables
```

## References

- **Exact SI Gaussian**: [`sinterp/exact_si_gaussian.py`](../sinterp/exact_si_gaussian.py)
- **Interpolants**: [`sinterp/interpolants.py`](../sinterp/interpolants.py)
- **Distribution implementations**: 
  - [`conditional_flows/static/dist.py`](../static/dist.py)
  - [`conditional_flows/parallel/distv2.py`](../parallel/distv2.py)
  - [`conditional_flows/guidance/dist.py`](../guidance/dist.py)
- **Test utilities**: [`tests/numerical_utils.py`](./numerical_utils.py)
