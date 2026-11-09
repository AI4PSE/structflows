"""
Mechanism implementations for Structural PGMs.

This module provides various mechanism types for the structural equation:
    x_i = f(pa(i), e_i)

where:
    - x_i is the value of node i
    - pa(i) are the parent values (stacked)  
    - e_i is independent noise sampled from N(0, I)

Mechanism types:
    - LinearMechanism: x = A @ parents + B @ noise + bias
    - NonlinearParentMechanism: x = f(parents) + g(parents) @ noise (Gaussian conditional)
    - NonlinearNoiseMechanism: x = A @ parents + f(noise) (odd f for zero-mean)
    - UniversalMechanism: x = f(parents, noise) using rational quadratic splines
    - CAREFLMechanism: x = exp(s(pa)) * z + t(pa) (Affine Autoregressive)
    - ConditionerMechanism: Generic wrapper for any affine conditioner
"""

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import List, Tuple, Optional
import distrax

from structflow.pgms.structural_pgm import Mechanism


class LinearMechanism(Mechanism):
    """
    Linear mechanism: x = A @ parents + B @ noise + bias
    
    For a node with no parents (root), this reduces to:
        x = B @ noise + bias
    
    The inverse is: noise = B^{-1} @ (x - A @ parents - bias)
    
    Attributes:
        A: Weight matrix for parents, shape (output_dim, input_dim)
        B: Scale matrix for noise, shape (output_dim, output_dim) - lower triangular
        bias: Bias vector, shape (output_dim,)
    """
    A: jnp.ndarray  # (output_dim, input_dim)
    B: jnp.ndarray  # (output_dim, output_dim) - lower triangular for invertibility
    bias: jnp.ndarray  # (output_dim,)
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        A: Optional[jnp.ndarray] = None,
        B: Optional[jnp.ndarray] = None,
        bias: Optional[jnp.ndarray] = None,
        key: Optional[jax.Array] = None
    ):
        """
        Initialize a LinearMechanism.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs
            output_dim: Dimension of this node's output
            A: Weight matrix (output_dim, input_dim). Random if None.
            B: Scale matrix (output_dim, output_dim). Identity if None.
            bias: Bias vector (output_dim,). Zeros if None.
            key: Random key for initialization (required if A is None)
        """
        self.parent_names = parent_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if A is None:
            if key is None:
                raise ValueError("key is required when A is not provided")
            k1, k2, k3 = random.split(key, 3)
            if input_dim > 0:
                A = random.normal(k1, (output_dim, input_dim)) * 0.5
            else:
                A = jnp.zeros((output_dim, 0))
        
        if B is None:
            # Default to identity with small random perturbation for positive definiteness
            B = jnp.eye(output_dim)
            if key is not None:
                k2 = random.fold_in(key, 1)
                # Add small lower-triangular noise
                noise = random.normal(k2, (output_dim, output_dim)) * 0.1
                B = B + jnp.tril(noise)
        
        if bias is None:
            bias = jnp.zeros(output_dim)
            if key is not None:
                k3 = random.fold_in(key, 2)
                bias = random.normal(k3, (output_dim,)) * 0.5
        
        # Ensure B is lower triangular
        self.B = jnp.tril(B)
        self.A = A
        self.bias = bias
    
    def forward(self, parents: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute x = A @ parents + B @ noise + bias.
        
        The Jacobian dx/d_noise = B, so log|det| = sum(log(|diag(B)|))
        """
        mean = self.bias
        if self.input_dim > 0:
            mean = mean + self.A @ parents
        x = mean + self.B @ noise
        
        # Log determinant of B (lower triangular, so det = prod of diagonal)
        log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(self.B))))
        
        return x, log_det
    
    def inverse(self, parents: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Recover noise = B^{-1} @ (x - A @ parents - bias).
        """
        residual = x - self.bias
        if self.input_dim > 0:
            residual = residual - self.A @ parents
        
        # Solve B @ noise = residual for noise (B is lower triangular)
        noise = jax.scipy.linalg.solve_triangular(self.B, residual, lower=True)
        
        # Log determinant of inverse is negative of forward
        log_det = -jnp.sum(jnp.log(jnp.abs(jnp.diag(self.B))))
        
        return noise, log_det


class NonlinearParentMechanism(Mechanism):
    """
    Nonlinear dependence on parents, Gaussian noise:
        x = mu(parents) + sigma(parents) @ noise
    
    where mu and sigma are neural networks.
    
    The conditional p(x | parents) is still Gaussian:
        p(x | parents) = N(x; mu(parents), sigma(parents) @ sigma(parents)^T)
    
    This is the architecture used in VAEs and many generative models.
    """
    mu_net: eqx.nn.MLP
    sigma_net: eqx.nn.MLP
    sigma_min: float = eqx.field(static=True)
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        key: jax.Array,
        hidden_width: int = 32,
        hidden_depth: int = 2,
        sigma_min: float = 0.01
    ):
        """
        Initialize a NonlinearParentMechanism.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs
            output_dim: Dimension of this node's output
            key: Random key for initialization
            hidden_width: Width of hidden layers in networks
            hidden_depth: Number of hidden layers
            sigma_min: Minimum value for sigma (for numerical stability)
        """
        self.parent_names = parent_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_min = sigma_min
        
        k1, k2 = random.split(key)
        
        # For root nodes (no parents), use a constant input
        effective_input = max(input_dim, 1)
        
        self.mu_net = eqx.nn.MLP(
            in_size=effective_input,
            out_size=output_dim,
            width_size=hidden_width,
            depth=hidden_depth,
            activation=jnp.tanh,
            key=k1
        )
        
        # Sigma net outputs log(sigma) for each dimension (diagonal covariance)
        self.sigma_net = eqx.nn.MLP(
            in_size=effective_input,
            out_size=output_dim,
            width_size=hidden_width,
            depth=hidden_depth,
            activation=jnp.tanh,
            key=k2
        )
    
    def _get_params(self, parents: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get mu and sigma from parent values."""
        if self.input_dim == 0:
            # Root node: use zeros as dummy input
            inp = jnp.zeros(1)
        else:
            inp = parents
        
        mu = self.mu_net(inp)
        log_sigma = self.sigma_net(inp)
        sigma = jax.nn.softplus(log_sigma) + self.sigma_min
        
        return mu, sigma
    
    def forward(self, parents: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute x = mu(parents) + sigma(parents) * noise.
        
        Using diagonal sigma, so log|det| = sum(log(sigma))
        """
        mu, sigma = self._get_params(parents)
        x = mu + sigma * noise
        log_det = jnp.sum(jnp.log(sigma))
        return x, log_det
    
    def inverse(self, parents: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Recover noise = (x - mu(parents)) / sigma(parents).
        """
        mu, sigma = self._get_params(parents)
        noise = (x - mu) / sigma
        log_det = -jnp.sum(jnp.log(sigma))
        return noise, log_det


class NonlinearNoiseMechanism(Mechanism):
    """
    Linear dependence on parents, nonlinear in noise:
        x = A @ parents + bias + f(noise)
    
    where f is an odd function (f(-e) = -f(e)) to preserve E[f(noise)] = 0.
    
    We implement f using a neural network made odd:
        f(e) = g(e) - g(-e)
    where g is any neural network.
    
    For invertibility, we use a monotonic spline transformation.
    """
    A: jnp.ndarray
    bias: jnp.ndarray
    spline: distrax.RationalQuadraticSpline
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        key: jax.Array,
        A: Optional[jnp.ndarray] = None,
        bias: Optional[jnp.ndarray] = None,
        num_bins: int = 8,
        range_min: float = -4.0,
        range_max: float = 4.0
    ):
        """
        Initialize a NonlinearNoiseMechanism.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs  
            output_dim: Dimension of this node's output
            key: Random key for initialization
            A: Weight matrix for parents (output_dim, input_dim)
            bias: Bias vector (output_dim,)
            num_bins: Number of spline bins
            range_min: Lower bound of spline domain
            range_max: Upper bound of spline domain
        """
        self.parent_names = parent_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        k1, k2, k3 = random.split(key, 3)
        
        if A is None:
            if input_dim > 0:
                A = random.normal(k1, (output_dim, input_dim)) * 0.5
            else:
                A = jnp.zeros((output_dim, 0))
        self.A = A
        
        if bias is None:
            bias = random.normal(k2, (output_dim,)) * 0.5
        self.bias = bias
        
        # Initialize spline parameters - we use random initialization
        # For an odd function, we'll symmetrize in forward/inverse
        # Shape for params: (output_dim, 3 * num_bins + 1)
        # widths, heights, derivatives
        params = random.normal(k3, (output_dim, 3 * num_bins + 1)) * 0.3
        
        self.spline = distrax.RationalQuadraticSpline(
            params,
            range_min=range_min,
            range_max=range_max
        )
    
    def _odd_transform(self, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply odd transformation: f(e) = (g(e) - g(-e)) / 2
        
        This ensures E[f(noise)] = 0 when noise ~ N(0, I).
        """
        # Forward pass for e and -e
        fe, log_det_e = self.spline.forward_and_log_det(noise)
        f_neg_e, log_det_neg = self.spline.forward_and_log_det(-noise)
        
        # Odd transform: (f(e) - f(-e)) / 2
        f_odd = (fe - f_neg_e) / 2
        
        # Log det for odd function: d/de [(f(e) - f(-e))/2] = (f'(e) + f'(-e))/2
        # This is tricky - we approximate by the average of derivatives
        # Actually, for a proper bijector we need: exp(log_det_e) and exp(log_det_neg)
        # d(f_odd)/d(noise) = (df(e)/de + df(-e)/d(-e) * (-1) * (-1)) / 2 = (df(e)/de + df(-e)/de) / 2
        # Since df(-e)/de = df(-e)/d(-e) * d(-e)/de = f'(-e) * (-1), we have
        # d(f_odd)/de = (f'(e) + f'(-e)) / 2
        det_e = jnp.exp(log_det_e)
        det_neg = jnp.exp(log_det_neg)
        det_odd = (det_e + det_neg) / 2
        log_det_odd = jnp.log(det_odd + 1e-8).sum()
        
        return f_odd, log_det_odd
    
    def _odd_inverse(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Invert the odd transformation.
        
        Finding e such that (g(e) - g(-e))/2 = y is non-trivial.
        We use Newton's method for each dimension.
        """
        # Use fixed-point iteration / Newton's method
        def solve_for_noise(y_i, spline_i):
            """Solve for e such that (g(e) - g(-e))/2 = y_i."""
            # Initial guess
            e = y_i
            
            for _ in range(10):  # Newton iterations
                fe, log_det_e = spline_i.forward_and_log_det(e)
                f_neg_e, _ = spline_i.forward_and_log_det(-e)
                f_odd = (fe - f_neg_e) / 2
                
                # Derivative
                det_e = jnp.exp(log_det_e)
                _, log_det_neg = spline_i.forward_and_log_det(-e)
                det_neg = jnp.exp(log_det_neg)
                df_de = (det_e + det_neg) / 2
                
                # Newton update
                residual = f_odd - y_i
                e = e - residual / (df_de + 1e-8)
                e = jnp.clip(e, -4.0, 4.0)  # Stay in domain
            
            return e
        
        # This is a simplified implementation - in practice we'd vectorize better
        # For now, we use a batched approach
        noise = jnp.zeros_like(y)
        
        # Simple fixed-point iteration
        for _ in range(15):
            fe, _ = self.spline.forward_and_log_det(noise)
            f_neg_e, _ = self.spline.forward_and_log_det(-noise)
            f_odd = (fe - f_neg_e) / 2
            
            # Gradient
            _, log_det_e = self.spline.forward_and_log_det(noise)
            _, log_det_neg = self.spline.forward_and_log_det(-noise)
            df_de = (jnp.exp(log_det_e) + jnp.exp(log_det_neg)) / 2
            
            residual = f_odd - y
            noise = noise - residual / (df_de + 1e-8)
            noise = jnp.clip(noise, -3.9, 3.9)
        
        # Compute log det of inverse
        _, log_det_odd = self._odd_transform(noise)
        log_det_inv = -log_det_odd
        
        return noise, log_det_inv
    
    def forward(self, parents: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute x = A @ parents + bias + f_odd(noise).
        """
        f_noise, log_det = self._odd_transform(noise)
        
        mean = self.bias
        if self.input_dim > 0:
            mean = mean + self.A @ parents
        
        x = mean + f_noise
        return x, log_det
    
    def inverse(self, parents: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Recover noise such that x = A @ parents + bias + f_odd(noise).
        """
        mean = self.bias
        if self.input_dim > 0:
            mean = mean + self.A @ parents
        
        y = x - mean  # This should equal f_odd(noise)
        noise, log_det = self._odd_inverse(y)
        
        return noise, log_det


class UniversalMechanism(Mechanism):
    """
    Fully nonlinear mechanism: x = f(parents, noise)
    
    Uses a conditional rational quadratic spline where spline parameters
    are computed from parent values via a neural network.
    
    This is the most expressive form, capable of representing any
    invertible transformation.
    """
    param_net: Optional[eqx.nn.MLP]
    base_params: jnp.ndarray
    num_bins: int = eqx.field(static=True)
    range_min: float = eqx.field(static=True)
    range_max: float = eqx.field(static=True)
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        key: jax.Array,
        num_bins: int = 8,
        hidden_width: int = 32,
        hidden_depth: int = 2,
        range_min: float = -4.0,
        range_max: float = 4.0
    ):
        """
        Initialize a UniversalMechanism.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs
            output_dim: Dimension of this node's output
            key: Random key for initialization
            num_bins: Number of spline bins
            hidden_width: Width of parameter network
            hidden_depth: Depth of parameter network
            range_min: Lower bound of spline domain
            range_max: Upper bound of spline domain
        """
        self.parent_names = parent_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        
        k1, k2 = random.split(key)
        
        # Number of parameters per output dimension: 3 * num_bins + 1
        # (widths, heights, derivatives)
        n_params = 3 * num_bins + 1
        total_params = output_dim * n_params
        
        # Base parameters (used for root nodes or added to network output)
        self.base_params = random.normal(k1, (output_dim, n_params)) * 0.3
        
        if input_dim > 0:
            self.param_net = eqx.nn.MLP(
                in_size=input_dim,
                out_size=total_params,
                width_size=hidden_width,
                depth=hidden_depth,
                activation=jnp.tanh,
                key=k2
            )
        else:
            self.param_net = None
    
    def _get_spline(self, parents: jnp.ndarray) -> distrax.RationalQuadraticSpline:
        """Get spline conditioned on parent values."""
        if self.param_net is not None and self.input_dim > 0:
            net_params = self.param_net(parents)
            params = self.base_params + net_params.reshape(self.output_dim, -1)
        else:
            params = self.base_params
        
        return distrax.RationalQuadraticSpline(
            params,
            range_min=self.range_min,
            range_max=self.range_max
        )
    
    def forward(self, parents: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute x = spline(noise; params(parents)).
        """
        spline = self._get_spline(parents)
        x, log_det = spline.forward_and_log_det(noise)
        return x, log_det.sum()
    
    def inverse(self, parents: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Recover noise = spline^{-1}(x; params(parents)).
        """
        spline = self._get_spline(parents)
        noise, log_det = spline.inverse_and_log_det(x)
        return noise, log_det.sum()


# --- Convenience constructors ---

def make_linear_mechanism(
    parent_names: List[str],
    parent_dims: List[int],
    output_dim: int,
    key: jax.Array,
    weight_scale: Optional[float] = None,
    noise_scale: Optional[float] = None,
    root_scale: Optional[float] = None,
) -> LinearMechanism:
    """Create a LinearMechanism. Random initialization by default; when
    weight_scale/noise_scale are given, uses deterministic A = weight_scale *
    I-like and B = noise_scale * I (e.g. strong-coupling chains where the
    end-to-end correlation must survive depth). root_scale, when given, sets
    the root's B independently of noise_scale: with the stationary default
    the noise scale cancels out of every correlation, so sustained-headroom
    chains (e.g. the w=1 random walk) need root variance >> link noise."""
    input_dim = sum(parent_dims)
    A = B = None
    if weight_scale is not None and input_dim > 0:
        A = weight_scale * jnp.eye(output_dim, input_dim)
    if noise_scale is not None:
        scale = noise_scale
        if input_dim == 0:
            if root_scale is not None:
                scale = root_scale
            elif weight_scale is not None and abs(weight_scale) < 1:
                # Stationary initialization for roots: with x_{i+1} = w x_i + s e,
                # var -> s^2/(1-w^2); starting the root there makes every link
                # correlation equal w, so end-to-end corr = w^(n-1) exactly.
                scale = noise_scale / (1.0 - weight_scale ** 2) ** 0.5
        B = scale * jnp.eye(output_dim)
    return LinearMechanism(parent_names, input_dim, output_dim, A=A, B=B, key=key)


def make_nonlinear_parent_mechanism(
    parent_names: List[str],
    parent_dims: List[int],
    output_dim: int,
    key: jax.Array,
    hidden_width: int = 32
) -> NonlinearParentMechanism:
    """Create a NonlinearParentMechanism."""
    input_dim = sum(parent_dims)
    return NonlinearParentMechanism(
        parent_names, input_dim, output_dim, key, hidden_width=hidden_width
    )


def make_nonlinear_noise_mechanism(
    parent_names: List[str],
    parent_dims: List[int],
    output_dim: int,
    key: jax.Array,
    num_bins: int = 8
) -> NonlinearNoiseMechanism:
    """Create a NonlinearNoiseMechanism."""
    input_dim = sum(parent_dims)
    return NonlinearNoiseMechanism(
        parent_names, input_dim, output_dim, key, num_bins=num_bins
    )


def make_universal_mechanism(
    parent_names: List[str],
    parent_dims: List[int],
    output_dim: int,
    key: jax.Array,
    num_bins: int = 8,
    hidden_width: int = 32
) -> UniversalMechanism:
    """Create a UniversalMechanism."""
    input_dim = sum(parent_dims)
    return UniversalMechanism(
        parent_names, input_dim, output_dim, key,
        num_bins=num_bins, hidden_width=hidden_width
    )


class CAREFLMechanism(Mechanism):
    """
    Causal Autoregressive Flow (CAREFL) mechanism.
    
    Implements affine autoregressive transformation:
        x = exp(s(pa)) * z + t(pa)
    
    where:
        - s(pa) and t(pa) are neural network conditioners
        - z is sampled from a base distribution (Laplace or Gaussian)
    
    This mechanism is used in CAREFL for identifiable causal discovery.
    
    Reference: Khemakhem et al., "Causal Autoregressive Flows", AISTATS 2021.
    """
    conditioner: eqx.nn.MLP  # Outputs (s, t) given parents
    sigma_min: float = eqx.field(static=True)
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        key: jax.Array,
        hidden_width: int = 64,
        hidden_depth: int = 3,
        sigma_min: float = 0.01
    ):
        """
        Initialize a CAREFLMechanism.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs
            output_dim: Dimension of this node's output
            key: Random key for initialization
            hidden_width: Width of hidden layers in conditioner
            hidden_depth: Number of hidden layers
            sigma_min: Minimum value for scale (for numerical stability)
        """
        self.parent_names = parent_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_min = sigma_min
        
        # Effective input dimension: max(input_dim, 1) for root nodes
        effective_input = max(input_dim, 1)
        
        # Conditioner outputs 2 * output_dim for (s, t)
        self.conditioner = eqx.nn.MLP(
            in_size=effective_input,
            out_size=2 * output_dim,
            width_size=hidden_width,
            depth=hidden_depth,
            activation=jnp.tanh,
            key=key
        )
    
    def _get_affine_params(self, parents: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get s and t parameters from parent values.
        
        Args:
            parents: Stacked parent values, shape (input_dim,)
            
        Returns:
            s: Log-scale, shape (output_dim,)
            t: Shift, shape (output_dim,)
        """
        if self.input_dim == 0:
            # Root node: use dummy input
            inp = jnp.array([1.0])
        else:
            inp = parents
        
        params = self.conditioner(inp)
        s, t = jnp.split(params, 2, axis=-1)
        return s, t
    
    def forward(self, parents: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute x = exp(s(parents)) * noise + t(parents).
        
        The Jacobian dx/d_noise = exp(s), so log|det| = sum(s)
        """
        s, t = self._get_affine_params(parents)
        x = jnp.exp(s) * noise + t
        log_det = jnp.sum(s)
        return x, log_det
    
    def inverse(self, parents: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Recover noise = (x - t(parents)) * exp(-s(parents)).
        """
        s, t = self._get_affine_params(parents)
        noise = (x - t) * jnp.exp(-s)
        log_det = -jnp.sum(s)
        return noise, log_det


def make_carefl_mechanism(
    parent_names: List[str],
    parent_dims: List[int],
    output_dim: int,
    key: jax.Array,
    hidden_width: int = 64,
    hidden_depth: int = 3,
    sigma_min: float = 0.01
) -> CAREFLMechanism:
    """
    Create a CAREFLMechanism.
    
    Args:
        parent_names: List of parent variable names
        parent_dims: List of dimensions for each parent
        output_dim: Dimension of the output variable
        key: JAX random key for initialization
        hidden_width: Width of neural network hidden layers
        hidden_depth: Depth of neural network
        sigma_min: Minimum scale value for numerical stability
        
    Returns:
        CAREFLMechanism instance
    """
    input_dim = sum(parent_dims)
    return CAREFLMechanism(
        parent_names, input_dim, output_dim, key,
        hidden_width=hidden_width, hidden_depth=hidden_depth,
        sigma_min=sigma_min
    )


class ConditionerMechanism(CAREFLMechanism):
    """
    A CAREFL-style affine mechanism that wraps an external conditioner network.
    
    This allows reusing existing conditioner implementations with the
    structural equation framework.
    """
    conditioner: eqx.Module  # Can be any module that outputs (s, t)
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        conditioner: eqx.Module,
        sigma_min: float = 0.01
    ):
        """
        Initialize with an external conditioner.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs
            output_dim: Dimension of this node's output
            conditioner: Neural network that outputs 2 * output_dim for (s, t)
            sigma_min: Minimum scale value for numerical stability
        """
        self.parent_names = parent_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_min = sigma_min
        self.conditioner = conditioner


class UniversalConditionerMechanism(UniversalMechanism):
    """
    A Universal mechanism that wraps an external conditioner network.
    """
    
    def __init__(
        self,
        parent_names: List[str],
        input_dim: int,
        output_dim: int,
        conditioner: eqx.Module,
        key: jax.Array,
        num_bins: int = 8,
        range_min: float = -4.0,
        range_max: float = 4.0
    ):
        """
        Initialize with an external conditioner.
        
        Args:
            parent_names: List of parent variable names
            input_dim: Total dimension of parent inputs
            output_dim: Dimension of this node's output
            conditioner: Network outputs (output_dim * (3 * num_bins + 1)) params
            key: Random key (needed for base params initialization)
            num_bins: Number of spline bins
            range_min: Lower bound of spline domain
            range_max: Upper bound of spline domain
        """
        super().__init__(
            parent_names=parent_names,
            input_dim=input_dim,
            output_dim=output_dim,
            key=key,
            num_bins=num_bins,
            hidden_width=32, # Dummy
            hidden_depth=2,  # Dummy
            range_min=range_min,
            range_max=range_max
        )
        self.param_net = conditioner
