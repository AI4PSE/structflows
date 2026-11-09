import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import equinox as eqx
from typing import Dict, List, Optional, Union, Set
from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.dist import ConditionalDistribution
from flow.utils import divergence_jvp
from flow.integrators import get_integrator
from structflow.conditional_flows.parallel.flow import ParallelFlow
from structflow.conditional_flows.parallel.block_extractor import BlockBasedVelocityFn
from sinterp.stochasticfield import StochasticField


from structflow.conditional_flows.eta1.net import ParallelStructuredFn_wEta1

class Eta1VelocityWrapper(eqx.Module):
    """
    Wraps ParallelStructuredFn_wEta1 to automatically compute eta1 from a guidance field.
    Exposes the same interface as ParallelStructuredFn: v(x, t).
    """
    net: ParallelStructuredFn_wEta1
    guidance_field: StochasticField
    
    @property
    def structure(self):
        return self.net.structure
        
    @property
    def block_offsets(self):
        return self.net.block_offsets
    
    def __call__(self, x_flat: jnp.ndarray, t: float) -> jnp.ndarray:
        # 1. Get eta1 prediction from guidance field
        # guidance_field.get_denoiser_eta1(x, t) returns full eta1 vector
        eta1_flat = self.guidance_field.get_denoiser_eta1(x_flat, t)
        
        # 2. Concatenate [x_t, eta1]
        x_concat = jnp.concatenate([x_flat, eta1_flat])
        
        # 3. Call net
        return self.net(x_concat, t)