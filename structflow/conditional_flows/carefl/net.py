import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from typing import List, Callable, Optional

class ResidualLayer(eqx.Module):
    """
    A simple residual layer: x + activation(linear(norm(x)))
    """
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    activation: Callable = eqx.field(static=True)

    def __init__(self, key, dim, activation=jax.nn.silu):
        self.linear = eqx.nn.Linear(dim, dim, key=key)
        self.norm = eqx.nn.LayerNorm(dim)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.norm(x)
        h = self.linear(h)
        h = self.activation(h)
        return x + h

class ResNetConditioner(eqx.Module):
    """
    ResNet Conditioner network.
    Takes concatenated parent values and outputs parameters.
    """
    input_projection: Optional[eqx.nn.Linear]
    residual_layers: List[ResidualLayer]
    output_projection: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: jax.Array,
        hidden_dim: int = 64,
        depth: int = 3,
        activation: Callable = jax.nn.silu
    ):
        """
        Args:
            input_dim: Total dimension of all parents concatenated.
            output_dim: Dimension of the output parameters (e.g. 2*dim for affine).
            key: JAX random key.
            hidden_dim: Width of hidden layers.
            depth: Total number of layers (including projection).
            activation: Activation function.
        """
        self.activation = activation
        
        keys = jax.random.split(key, depth + 1)
        
        if input_dim > 0:
            self.input_projection = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        else:
            self.input_projection = None
            
        self.residual_layers = [
            ResidualLayer(key=keys[i+1], dim=hidden_dim, activation=activation)
            for i in range(max(0, depth - 2))
        ]
        
        if input_dim > 0:
            self.output_projection = eqx.nn.Linear(hidden_dim, output_dim, key=keys[-1])
        else:
            # For root nodes, project from dummy input
            self.output_projection = eqx.nn.Linear(1, output_dim, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.input_projection is None:
            return self.output_projection(jnp.array([1.0]))
        
        h = self.activation(self.input_projection(x))
        for layer in self.residual_layers:
            h = layer(h)
        return self.output_projection(h)

# Alias for backward compatibility (if needed) or specific usage
class CAREFLConditioner(ResNetConditioner):
    """Legacy wrapper for affine CAREFL parameters (s, t)."""
    def __init__(self, input_dim, output_dim, **kwargs):
        # CAREFL expects 2 * output_dim params
        super().__init__(input_dim, 2 * output_dim, **kwargs)
        self.real_output_dim = output_dim
