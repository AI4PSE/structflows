import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Dict, List, Optional, Union, Callable

class ConditionalFlow(eqx.Module):
    """
    Abstract base class for conditional flows.
    
    Provides a consistent interface for flows that depend on external conditions,
    whether static (fixed values) or dynamic (likelihood gradients).
    """
    n_steps: int = eqx.field(static=True)
    integrator_name: str = eqx.field(static=True)

    def __init__(self, n_steps: int, integrator_name: str = 'rk4'):
        self.n_steps = n_steps
        self.integrator_name = integrator_name

    def trajectory_forward(self, x0: Any, **kwargs) -> Any:
        """
        Compute forward trajectory x(0) -> x(1).
        """
        raise NotImplementedError

    def trajectory_backward(self, x1: Any, **kwargs) -> Any:
        """
        Compute backward trajectory x(1) -> x(0).
        """
        raise NotImplementedError

    def forward(self, x0: Any, **kwargs) -> Any:
        """
        Compute forward mapping x(0) -> x(1).
        Default implementation takes the last point of the trajectory.
        """
        traj = self.trajectory_forward(x0, **kwargs)
        if isinstance(traj, dict):
            return {k: v[-1] for k, v in traj.items()}
        else:
            return traj[-1]

    def backward(self, x1: Any, **kwargs) -> Any:
        """
        Compute backward mapping x(1) -> x(0).
        Default implementation takes the last point of the trajectory.
        """
        traj = self.trajectory_backward(x1, **kwargs)
        if isinstance(traj, dict):
            return {k: v[-1] for k, v in traj.items()}
        else:
            return traj[-1]
