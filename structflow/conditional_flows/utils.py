import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Optional, Union, Set, Tuple
import equinox as eqx

class IncrementalNormaliser(eqx.Module):
    mean: jnp.ndarray
    m2: jnp.ndarray
    count: jnp.ndarray
    eps: float

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-5):
        self.mean = jnp.zeros(shape)
        self.m2 = jnp.zeros(shape)
        self.count = jnp.zeros((), dtype=jnp.int32)
        self.eps = eps

    def update(self, batch: jnp.ndarray) -> "IncrementalNormaliser":
        """Update stats with Welford's algorithm for batches."""
        batch_count = batch.shape[0]
        if batch_count == 0:
            return self

        batch_mean = jnp.mean(batch, axis=0)
        batch_m2 = jnp.sum((batch - batch_mean)**2, axis=0)

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        
        # Calculate new_mean and new_m2 using batch-wise Welford
        new_mean = self.mean + delta * (batch_count / new_count)
        new_m2 = self.m2 + batch_m2 + (delta**2) * (self.count * batch_count / new_count)
        
        return eqx.tree_at(
            lambda x: (x.mean, x.m2, x.count),
            self,
            (new_mean, new_m2, new_count)
        )

    @property
    def var(self) -> jnp.ndarray:
        return self.m2 / jnp.maximum(self.count, 1)

    @property
    def std(self) -> jnp.ndarray:
        variance = self.var
        # Handle zero variance/constant features
        return jnp.sqrt(jnp.where(variance < self.eps**2, 1.0, variance))

    def normalise(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.mean) / self.std

    def unnormalise(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self.std + self.mean
    
    def log_det_jacobian(self) -> jnp.ndarray:
        return -jnp.sum(jnp.log(self.std))

class DictNormaliser(eqx.Module):
    # Map variable name -> IncrementalNormaliser for that variable
    normalisers: Dict[str, IncrementalNormaliser]

    def __init__(self, data_shapes: Dict[str, Tuple[int, ...]], eps: float = 1e-5):
        """
        Args:
            data_shapes: Dictionary of variable shapes { 'x0': (dim,), ... }
        """
        self.normalisers = {
            k: IncrementalNormaliser(shape, eps=eps) 
            for k, shape in data_shapes.items()
        }

    @classmethod
    def from_data(cls, data: Dict[str, jnp.ndarray], eps: float = 1e-5) -> "DictNormaliser":
        """Initialize from a batch of data."""
        shapes = {k: v.shape[1:] for k, v in data.items()}
        inst = cls(shapes, eps=eps)
        return inst.update(data)

    def update(self, data: Dict[str, jnp.ndarray]) -> "DictNormaliser":
        """Update all internal normalisers with a new batch of data."""
        new_normalisers = {
            k: self.normalisers[k].update(v)
            for k, v in data.items()
        }
        return eqx.tree_at(lambda x: x.normalisers, self, new_normalisers)

    def normalise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        # Use eqx.tree_pformat to handle JAX tracers safely
        # Convert the dict to a list of (key, value) pairs we can iterate over
        result = {}
        for k in self.normalisers:
            if k in x:
                result[k] = self.normalisers[k].normalise(x[k])
        return result

    def unnormalise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return {
            k: self.normalisers[k].unnormalise(v)
            for k, v in x.items()
        }

    def log_det_jacobian(self, targets: Optional[Union[Set[str], frozenset]] = None) -> jnp.ndarray:
        if targets is None:
            relevant_keys = self.normalisers.keys()
        else:
            relevant_keys = targets
            
        return sum(self.normalisers[k].log_det_jacobian() for k in relevant_keys)
