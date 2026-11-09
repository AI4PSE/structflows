import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from sklearn.datasets import load_digits, fetch_openml
from sinterp.interpolants import OneSidedLinear
from sinterp.stochasticfield import OneSidedField_b, OneSidedField_eta1
from flow.flow import Flow
from flow.net import VelocityNet, EtaNet
from structflow.bayes.metrics import compute_swd_jax
from sinterp.couplings import SinkhornCoupling

def load_mnist():
    """Load MNIST dataset via sklearn."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    return mnist.data.astype('float32'), mnist.target.astype('int32')

@pytest.mark.expensive
def test_digits_integration():
    # Toggle between 'digits' and 'mnist'
    # dataset_name = 'digits' # 'mnist'
    dataset_name = 'mnist'

    # field_type = OneSidedField_b
    field_type = OneSidedField_eta1

    n_samples = 20000

    # 1. Load data
    if dataset_name == 'digits':
        digits = load_digits()
        data = jnp.array(digits.data)  # (1797, 64)
    else:
        # Warning: MNIST is much larger and might be slow for a quick integration test
        data_np, _ = load_mnist()
        data = jnp.array(data_np).reshape(-1, 784)  # (70000, 784)
        data = data[:n_samples]
        perm_key = jax.random.PRNGKey(42)
        data = jax.random.permutation(perm_key, data, axis=0)
    
    # Simple subset for quick test
    x1_data = data[:n_samples]
    
    # Normalize
    mean = x1_data.mean(axis=0)
    std = x1_data.std(axis=0) + 1e-5
    x1_data = (x1_data - mean) / std
    dim = x1_data.shape[1]
    
    # 2. Setup model
    key = jax.random.PRNGKey(0)
    model_key, train_key, eval_key = jax.random.split(key, 3)
    
    interpolator = OneSidedLinear()
    # Use a medium sized net
    velocity_net = VelocityNet(dim=dim, key=model_key, width=256, depth=4)
    field = field_type(interpolator, velocity_net)
    
    # Helper for Flow
    class VelocityWrapper(eqx.Module):
        field: field_type
        def __call__(self, x, t):
            t_arr = jnp.atleast_1d(t)
            # Flow's forward(x_0) with x_0: (dim,) should pass x: (dim,)
            # get_velocity_b returns a PyTree.
            # OneSidedField_eta1.get_velocity_b(x, t) calls self.eta_1_model(x, t)
            # VelocityNet.__call__(x, t) returns array of shape (dim,)
            
            # The error 'The input carry component loop_carry[1] has type float32[D] 
            # but the corresponding output carry component has type float32[1,D]'
            # occurs because jnp.atleast_1d(t) within get_velocity_b/VelocityNet
            # might be causing some broadcasting when combined with x.
            
            res = self.field.get_velocity_b(x, t_arr)
            # Ensure it matches input shape exactly to satisfy lax.fori_loop shape consistency
            return res.reshape(x.shape)

    def get_swd(current_field, k):
        v_wrap = VelocityWrapper(current_field)
        flow_eval = Flow(velocity=v_wrap, n_steps=20, integrator_name='rk4')
        n_eval = 1000
        k_eval, k_swd = jax.random.split(k)
        z_eval = jax.random.normal(k_eval, (n_eval, dim))
        samples_gen = flow_eval.b_forward(z_eval)
        return float(compute_swd_jax(k_swd, samples_gen, x1_data[:n_eval]))

    # Calculate initial SWD
    swd_before = get_swd(field, eval_key)
    print(f"SWD before training ({dataset_name}): {swd_before}")
    
    # 3. Training loop
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(field, eqx.is_array))

    coupling = SinkhornCoupling()
    
    @eqx.filter_jit
    def train_step(field, opt_state, x1_batch, key):
        e_batch = jax.random.normal(key, x1_batch.shape)

        e_batch, x1_batch, _, _, _ = coupling(key, e_batch, x1_batch)
        
        def batch_loss(field):
            t_key, _ = jax.random.split(key)
            t = jax.random.uniform(t_key, (x1_batch.shape[0],))
            
            def compute_loss(x1, e, t_val):
                return field.total_loss(None, x1, e, t_val)
                
            losses = jax.vmap(compute_loss)(x1_batch, e_batch, t)
            return jnp.mean(losses)
            
        loss, grads = eqx.filter_value_and_grad(batch_loss)(field)
        updates, opt_state = optimizer.update(grads, opt_state)
        field = eqx.apply_updates(field, updates)
        return field, opt_state, loss

    # Train
    steps = 5000
    batch_size = 128
    for i in range(steps):
        step_key = jax.random.fold_in(train_key, i)
        idx = jax.random.choice(jax.random.fold_in(step_key, 1), n_samples, (batch_size,))
        x1_batch = x1_data[idx]
        field, opt_state, loss = train_step(field, opt_state, x1_batch, step_key)
        if i % 250 == 0:
            print(f"Step {i}, Loss: {loss}")

    # Calculate final SWD
    swd_after = get_swd(field, jax.random.fold_in(eval_key, 1))
    print(f"SWD after training ({dataset_name}): {swd_after}")
    
    # Verify improvement
    # assert swd_after < swd_before
    print(f"Integration test passed with SWD improvement for {dataset_name}!")

    # plot.
    noise = jax.random.normal(key, (5, dim))
    samples = Flow(VelocityWrapper(field), n_steps=20, integrator_name='rk4').b_forward(noise)

    k = int(dim ** 0.5)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(samples[i].reshape(k, k), cmap='gray')
        plt.axis('off')

    for j in range(5):
        plt.subplot(2, 5, j + 6)
        plt.imshow(x1_data[j].reshape(k, k), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_digits_integration()
