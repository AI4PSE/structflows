
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List
from flow.net import FourierFeatures

class ResidualLayer(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    activation: Callable

    def __init__(self, key, dim, activation):
        self.linear = eqx.nn.Linear(dim, dim, key=key)
        self.norm = eqx.nn.LayerNorm(dim)
        self.activation = activation

    def __call__(self, x):
        h = self.norm(x)
        h = self.linear(h)
        h = self.activation(h)
        return x + h

class FlatVelocityFn(eqx.Module):
    """
    A simple MLP that takes a flat vector [x, t_embed] and outputs a velocity vector v.
    No structural masking is applied.
    """
    time_embed: FourierFeatures
    layers: List[eqx.Module]
    output_projection: eqx.nn.Linear
    
    def __init__(
        self,
        input_dim: int, # Dimension of the data state x
        hidden_dim: int,
        depth: int,
        key: jax.random.PRNGKey,
        activation: Callable = jax.nn.silu,
        time_embed_dim: int = 16
    ):
        
        # Keys
        keys = jax.random.split(key, depth + 2)
        t_key, in_key, out_key = keys[0], keys[1], keys[-1]
        res_keys = keys[2:-1]
        
        # 1. Time Embedder
        self.time_embed = FourierFeatures(
            output_features=time_embed_dim,
            key=t_key
        )
        
        # 2. Input Projection
        # Input to first layer is [x, t_embed]
        self.layers = []
        self.layers.append(eqx.nn.Linear(input_dim + time_embed_dim, hidden_dim, key=in_key))
        self.layers.append(activation)
        
        # 3. Residual Layers
        for i in range(depth - 2):
            self.layers.append(
                ResidualLayer(
                    key=res_keys[i],
                    dim=hidden_dim,
                    activation=activation
                )
            )
            
        # 4. Output Projection
        self.output_projection = eqx.nn.Linear(hidden_dim, input_dim, key=out_key)

    def __call__(self, x: jnp.ndarray, t: float):
        """
        Args:
            x: Flat state vector (D,)
            t: Scalar time
        """
        t_arr = jnp.array([t])
        t_embed = self.time_embed(t_arr).squeeze() # (time_embed_dim,)
        
        # Concatenate x and t_embed
        x_in = jnp.concatenate([x, t_embed])
        
        h = x_in
        for layer in self.layers:
            h = layer(h)
            
        v = self.output_projection(h)
        return v
