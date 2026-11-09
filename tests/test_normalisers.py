import jax
import jax.numpy as jnp
import equinox as eqx
from structflow.conditional_flows.utils import IncrementalNormaliser, DictNormaliser

def test_incremental_normaliser():
    print("Testing IncrementalNormaliser...")
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (1000, 5))
    
    # 1. Initialization
    norm = IncrementalNormaliser(shape=(5,))
    assert norm.mean.shape == (5,)
    assert norm.var.shape == (5,)
    assert norm.count == 0
    
    # 2. Update with batches
    batch1 = data[:100]
    batch2 = data[100:250]
    batch3 = data[250:]
    
    norm = norm.update(batch1)
    norm = norm.update(batch2)
    norm = norm.update(batch3)
    
    # 3. Reference statistics (exact mean/var)
    ref_mean = jnp.mean(data, axis=0)
    ref_var = jnp.var(data, axis=0)
    
    print(f"  Norm Mean: {norm.mean}")
    print(f"   Ref Mean: {ref_mean}")
    assert jnp.allclose(norm.mean, ref_mean, atol=1e-5)
    assert jnp.allclose(norm.var, ref_var, atol=1e-5)
    assert norm.count == 1000
    
    # 4. Normalise/Unnormalise
    x = jax.random.normal(key, (5,))
    x_norm = norm.normalise(x)
    x_unnorm = norm.unnormalise(x_norm)
    assert jnp.allclose(x, x_unnorm, atol=1e-5)
    
    # 5. Log Det Jacobian
    ldj = norm.log_det_jacobian()
    # Jacobian is diagonal with entries 1/std
    # log_det = sum(-log(std)) = -0.5 * sum(log(var))
    expected_ldj = -0.5 * jnp.sum(jnp.log(jnp.maximum(norm.var, norm.eps)))
    assert jnp.allclose(ldj, expected_ldj, atol=1e-5)
    
    print("IncrementalNormaliser tests passed!")

def test_dict_normaliser():
    print("Testing DictNormaliser...")
    data_shapes = {'a': (2,), 'b': (3,)}
    norm = DictNormaliser(data_shapes=data_shapes)
    
    batch_data = {
        'a': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'b': jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    }
    
    norm = norm.update(batch_data)
    
    assert norm.normalisers['a'].count == 2
    assert norm.normalisers['b'].count == 2
    
    # Test normalise
    x_dict = {'a': jnp.array([2.0, 3.0]), 'b': jnp.array([6.0, 7.0, 8.0])}
    x_norm = norm.normalise(x_dict)
    
    # Manual check for 'a'
    # mean_a = [2, 3], var_a = [1, 1]
    # normalise(2, 3) -> (2-2)/1, (3-3)/1 -> (0, 0)
    assert jnp.allclose(x_norm['a'], jnp.array([0.0, 0.0]), atol=1e-5)
    
    # Test Log Det Jacobian
    ldj = norm.log_det_jacobian()
    ldj_a = norm.normalisers['a'].log_det_jacobian()
    ldj_b = norm.normalisers['b'].log_det_jacobian()
    assert jnp.allclose(ldj, ldj_a + ldj_b)
    
    print("DictNormaliser tests passed!")

def test_shape_robustness():
    print("Testing shape robustness...")
    # Test that scalar inputs are handled and don't collapse dimensions
    norm = IncrementalNormaliser(shape=(1,))
    
    # Update with (B, 1)
    batch = jnp.array([[1.0], [2.0], [3.0]])
    norm = norm.update(batch)
    
    assert norm.mean.shape == (1,)
    assert norm.var.shape == (1,)
    
    # Test normalise returns (1,)
    x = jnp.array([2.0])
    x_norm = norm.normalise(x)
    assert x_norm.shape == (1,)
    
    print("Shape robustness tests passed!")

if __name__ == "__main__":
    test_incremental_normaliser()
    test_dict_normaliser()
    test_shape_robustness()
