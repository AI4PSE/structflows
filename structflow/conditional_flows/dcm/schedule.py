"""Variance-preserving cosine noise schedule for the DCM backbone.

Time convention matches the rest of this codebase's flows: ``t = 1`` is data,
``t = 0`` is noise. (Diffusion papers usually run the opposite way; we flip
their schedule so that ``ddim_forward`` integrates noise -> data over t: 0 -> 1
exactly like ``StaticConditionalFlow.trajectory_forward``.)

The marginal is ``x_t = sqrt(alpha_bar(t)) * x_data + sqrt(1 - alpha_bar(t)) * eps``
with the Nichol & Dhariwal (2021) cosine schedule

    alpha_bar(t) = f(1 - t) / f(0),   f(u) = cos((u + s) / (1 + s) * pi / 2)^2

normalised so alpha_bar(1) = 1 (data end is exact — required for the
round-trip / abduction contract) and clipped below at ``ALPHA_BAR_MIN`` so the
noise end stays numerically invertible (division by sqrt(alpha_bar)).
"""

import jax.numpy as jnp

S_OFFSET = 0.008
ALPHA_BAR_MIN = 1e-4


def alpha_bar(t):
    """Squared data-scale ᾱ(t); ᾱ(1) = 1 (data), ᾱ(0) ≈ 0 (noise)."""
    u = 1.0 - t  # diffusion time: 0 at data, 1 at noise
    f = jnp.cos((u + S_OFFSET) / (1.0 + S_OFFSET) * jnp.pi / 2.0) ** 2
    f0 = jnp.cos(S_OFFSET / (1.0 + S_OFFSET) * jnp.pi / 2.0) ** 2
    return jnp.clip(f / f0, ALPHA_BAR_MIN, 1.0)


def sqrt_alpha_bar(t):
    return jnp.sqrt(alpha_bar(t))


def sigma(t):
    """Noise scale: sigma(t) = sqrt(1 - alpha_bar(t))."""
    return jnp.sqrt(1.0 - alpha_bar(t))


def time_grid(n_steps: int):
    """Uniform discretization to ``n_steps`` DDIM steps: n_steps + 1 points
    from noise (t=0) to data (t=1)."""
    return jnp.linspace(0.0, 1.0, n_steps + 1)
