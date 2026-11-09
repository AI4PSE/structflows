"""Deterministic DDIM sampling and inversion (η = 0) for one block.

Both directions apply the *same* update rule on a uniform time grid,
re-using ε̂ evaluated at the **current** point:

    x_data_pred = (x - sigma(t_cur) * ε̂(x, t_cur)) / sqrt_alpha_bar(t_cur)
    x_next      = sqrt_alpha_bar(t_next) * x_data_pred + sigma(t_next) * ε̂(x, t_cur)

``ddim_forward`` walks the grid noise -> data (t: 0 -> 1); ``ddim_invert``
walks it data -> noise (t: 1 -> 0). This is the inversion as published
(Song et al., 2021; used by Diff-SCM): the exact inverse of a forward step
would need ε̂ at the *other* endpoint, so re-using the current point carries a
first-order O(1/N) global bias. Do not "improve" this — that bias is the
object of study in the backbone ablation (abduction error vs NFE).
"""

import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple

from structflow.conditional_flows.dcm.schedule import alpha_bar, time_grid


def ddim_step(eps_fn: Callable, x: jnp.ndarray,
              t_cur: jnp.ndarray, t_next: jnp.ndarray,
              parents: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    """One deterministic DDIM update from t_cur to t_next (either direction).

    ``eps_fn`` has the Block signature ``(t_arr, x, *parents) -> ε̂``.
    """
    ab_cur = alpha_bar(t_cur)
    ab_next = alpha_bar(t_next)
    eps_hat = eps_fn(jnp.atleast_1d(t_cur), x, *parents)
    x_data_pred = (x - jnp.sqrt(1.0 - ab_cur) * eps_hat) / jnp.sqrt(ab_cur)
    return jnp.sqrt(ab_next) * x_data_pred + jnp.sqrt(1.0 - ab_next) * eps_hat


def ddim_forward(eps_fn: Callable, u: jnp.ndarray,
                 parents: Tuple[jnp.ndarray, ...], n_steps: int) -> jnp.ndarray:
    """Integrate one block from noise ``u`` (t=0) to data (t=1).

    Exactly ``n_steps`` network evaluations (NFE = n_steps)."""
    ts = time_grid(n_steps)

    def body(i, x):
        return ddim_step(eps_fn, x, ts[i], ts[i + 1], parents)

    return lax.fori_loop(0, n_steps, body, u)


def ddim_invert(eps_fn: Callable, x: jnp.ndarray,
                parents: Tuple[jnp.ndarray, ...], n_steps: int) -> jnp.ndarray:
    """Invert one block from data ``x`` (t=1) to noise (t=0) — abduction.

    The published DDIM inversion: reversed grid, ε̂ re-used at the current
    point; carries the first-order step-size bias (see module docstring).
    Exactly ``n_steps`` network evaluations (NFE = n_steps)."""
    ts = time_grid(n_steps)[::-1]

    def body(i, xx):
        return ddim_step(eps_fn, xx, ts[i], ts[i + 1], parents)

    return lax.fori_loop(0, n_steps, body, x)
