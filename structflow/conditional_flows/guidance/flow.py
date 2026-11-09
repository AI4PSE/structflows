import jax
import jax.numpy as jnp
from jax import vmap
from typing import Callable, Optional, Any, Dict
import equinox as eqx

from flow.flow import Flow
from structflow.conditional_flows.flow import ConditionalFlow
from sinterp.stochasticfield import StochasticField
from structflow.conditional_flows.guidance.guidance_based import ReconstructionGuidedVelocity

class GuidedFlow(ConditionalFlow):
    """
    Flow for joint distributions that supports reconstruction guidance.
    Strictly operates on vector inputs/outputs.
    """
    model: StochasticField
    guidance_scale: float = 1.0
    unflatten_fn: Optional[Callable] = None

    def __init__(self, model: StochasticField, n_steps: int, guidance_scale: float = 1.0, integrator_name: str = 'rk4', unflatten_fn: Optional[Callable] = None):
        super().__init__(n_steps, integrator_name)
        self.model = model
        self.guidance_scale = guidance_scale
        self.unflatten_fn = unflatten_fn

    def get_velocity_fn(self, conditions: Dict[str, jnp.ndarray]) -> Callable:
        """
        Returns an unconditional velocity function v(x, t) with guidance applied.
        """
        if self.unflatten_fn is None:
            def log_likelihood_fn(x_pred_vec):
                # Assume x_pred is already a dict if unflatten_fn is not provided
                x_pred = x_pred_vec
                mse = 0.0
                for k, y_val in conditions.items():
                    if k in x_pred:
                        mse += jnp.sum((x_pred[k] - y_val)**2)
                return -mse
        else:
            def log_likelihood_fn(x_pred_vec):
                x_pred = self.unflatten_fn(x_pred_vec)
                mse = 0.0
                for k, y_val in conditions.items():
                    if k in x_pred:
                        mse += jnp.sum((x_pred[k] - y_val)**2)
                return -mse


        guided_vel = ReconstructionGuidedVelocity(
            prior_field=self.model,
            likelihood_log_prob_fn=lambda _, x_pred: log_likelihood_fn(x_pred),
            scale=self.guidance_scale
        )
        return lambda x, t: guided_vel(x, t, None)

    def trajectory_forward(self,
                           x0: jnp.ndarray,
                           conditions: Any,
                           **kwargs) -> jnp.ndarray:
        """
        Compute forward trajectory with guidance.
        """
        bounded_vel = self.get_velocity_fn(conditions)
        flow = Flow(bounded_vel, self.n_steps, self.integrator_name)
        return flow.forward_trajectory(x0)

    def trajectory_backward(self,
                            x1: jnp.ndarray,
                            conditions: Any,
                            **kwargs) -> jnp.ndarray:
        """
        Compute backward trajectory with guidance.
        """
        bounded_vel = self.get_velocity_fn(conditions)
        flow = Flow(bounded_vel, self.n_steps, self.integrator_name)
        return flow.backward_trajectory(x1)
