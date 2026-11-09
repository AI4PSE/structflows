import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from typing import Callable, Any
import optax

from sinterp.stochasticfield import TwoSidedField
from structflow.bayes.distribution import Distribution
from flow.flow import Flow

class AbstractGuidedVelocity(eqx.Module):
    """
    Base class for guided vector fields using Stochastic Interpolants.
    
    Implements: u(x, t) = b_prior(x, t) + w(t) * guidance_vector(x, t)
    """
    prior_field: TwoSidedField
    scale: float

    def __init__(self, prior_field, scale=1.0):
        self.prior_field = prior_field
        self.scale = scale

    def get_weight(self, t):
        # Heuristic weight w(t) = - scale * gamma(t) * d(gamma)/dt (= scale*(1-t) for OneSidedLinear).
        # Not the exact score-to-velocity factor, which is
        # (alpha^2 + gamma^2) * beta'/beta - (alpha*alpha' + gamma*gamma') (= (1-t)/t for OneSidedLinear)
        # and diverges at t=0; `scale` (7 in the paper) compensates. See the paper's guided-baseline appendix.
        gamma = self.prior_field.interpolator.gamma(t)
        dgamma = self.prior_field.interpolator.dgammadt(t)
        
        return - self.scale * gamma * dgamma

    @abstractmethod
    def get_guidance_vector(self, x, t, observation):
        """Computes the direction of the guidance (e.g. grad log p(y|...))"""
        pass

    def __call__(self, x, t, observation):
        # 1. Prior Velocity (Probability Flow)
        b_prior = self.prior_field.get_velocity_b(x, t)
        
        # 2. Guidance Direction
        guidance_vector = self.get_guidance_vector(x, t, observation)
        
        # 3. Weighting
        w = self.get_weight(t)
        
        # Support both array and dictionary guidance vectors
        def _apply_weight(v):
            if w.size == 1:
                return v * w
            elif w.size == v.size:
                return v * w
            elif len(w.shape) == 2 and w.shape[1] == v.shape[1]:
                return jnp.matmul(w, v)
            else:
                raise ValueError("Weight and guidance vector component must have compatible sizes")

        weighted_guidance = jax.tree_util.tree_map(_apply_weight, guidance_vector)
        
        # Combine prior and weighted guidance
        out = jax.tree_util.tree_map(lambda b, g: b + g, b_prior, weighted_guidance)
        # jax.debug.print("t={t}, w={w}, b_prior={b}, grad={g}, out={o}", t=t, w=w, b=b_prior, g=guidance_vector, o=out)
        return out

# --- 1. Reconstruction Guidance (Chain Rule) ---
# v_guided = v_prior + w(t) * grad_x log p(y | eta_1(x))

class ReconstructionGuidedVelocity(AbstractGuidedVelocity):
    likelihood_log_prob_fn: Callable

    def __init__(self, prior_field, likelihood_log_prob_fn, scale=1.0):
        super().__init__(prior_field, scale)
        self.likelihood_log_prob_fn = likelihood_log_prob_fn

    def get_guidance_vector(self, x, t, observation):
        def composite_log_prob(_x):
            # We predict eta1 from x
            eta1 = self.prior_field.get_denoiser_eta1(_x, t)
            # We compute likelihood of predicted clean data
            log_prob = self.likelihood_log_prob_fn(observation, eta1)
            # Ensure it returns a scalar even if it's a 0-dim array
            return jnp.sum(log_prob)
        
        grad = jax.grad(composite_log_prob)(x)
        
        # CRITICAL FIX: jax.grad may prune unused keys from the gradient dictionary.
        # If x is dict and some keys don't contribute to log_prob, grad will be missing those keys.
        # For example, when conditioning on 'label' to get p(image|label), the gradient w.r.t. 
        # 'label' is zero, so JAX drops it. We must preserve the full structure by filling 
        # missing keys with zeros.
        if isinstance(x, dict) and isinstance(grad, dict):
            full_grad = {}
            for key in x.keys():
                if key in grad:
                    full_grad[key] = grad[key]
                else:
                    # Key was pruned by JAX - fill with zeros
                    full_grad[key] = jnp.zeros_like(x[key])
            return full_grad
        
        return grad


# --- 2. Projected Guidance (Jacobian Free) ---
# v_guided = v_prior + w(t) * s_likelihood(y, eta_1(x))
# This assumes grad_x eta_1 is approx Identity. Faster than Reconstruction.

class ProjectedGuidedVelocity(AbstractGuidedVelocity):
    likelihood_score_fn: Callable

    def __init__(self, prior_field, likelihood_score_fn, scale=1.0):
        super().__init__(prior_field, scale)
        self.likelihood_score_fn = likelihood_score_fn

    def get_guidance_vector(self, x, t, observation):
        # Stop gradient on eta1 to ensure we don't backprop through the network
        # This makes it "Jacobian Free" / "Projected"
        eta1 = jax.lax.stop_gradient(self.prior_field.get_denoiser_eta1(x, t))
        
        # Compute grad_x log p(y|x) approximated at eta1
        # The likelihood_score_fn should return grad_z log p(y|z)
        return self.likelihood_score_fn(observation, eta1)


# --- 3. Direct/Noisy Guidance ---
# v_guided = v_prior + w(t) * grad_x log p(y | x, t)
# Requires a trained time-dependent classifier or analytical likelihood on x_t.

class DirectGuidedVelocity(AbstractGuidedVelocity):
    time_dependent_log_prob_fn: Callable

    def __init__(self, prior_field, time_dependent_log_prob_fn, scale=1.0):
        super().__init__(prior_field, scale)
        self.time_dependent_log_prob_fn = time_dependent_log_prob_fn

    def get_guidance_vector(self, x, t, observation):
        return jax.grad(lambda _x: self.time_dependent_log_prob_fn(observation, _x, t))(x)
    

# 4. instead of using eta1, we could actually integrate the flow to get x_1(x_t)?
# 5. what about some other kinds of scaling? or the OOD / multi-fidelity settings.