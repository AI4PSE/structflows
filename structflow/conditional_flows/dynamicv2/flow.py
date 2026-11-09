"""
DynamicV2Flow: Flow with weight clipping and condition trajectory interpolation.
"""

import equinox as eqx
import jax.numpy as jnp
from typing import Dict, List, Optional
from structflow.conditional_flows.flow import ConditionalFlow
from structflow.conditional_flows.parallel.flow import ParallelFlow
from structflow.conditional_flows.dynamicv2.subset_velocity import (
    clip_tail,
    SubsetVelocityFn,
    interp_trajectory,
)
from flow.flow import Flow


class ClampedFullVelocityFn(eqx.Module):
    """Conditional velocity that evaluates the FULL network on the full state:
    active blocks from the integrated state, condition blocks clamped to their
    interpolated trajectories, all remaining blocks zeroed (safe: by block
    triangularity every block influencing an active block is one of its
    ancestors, and all such ancestors are active or conditions).

    This replaces the Mode-A weight-clipping path (clip_tail +
    SubsetVelocityFn) for conditional integration: clipping severed the
    cross-block hidden-unit pathways (active blocks' hidden units consume
    condition blocks' hidden units at every layer under the structure masks),
    which distorted conditional samples — e.g. 2.5x overdispersion on
    inference-compiled chains. Cost: one full-network
    evaluation per step instead of a clipped one.
    """
    velocity_fn: eqx.Module
    condition_trajs: Dict[str, jnp.ndarray]
    traj_ts: jnp.ndarray
    cond_block_names: List[str] = eqx.field(static=True)
    active_block_names: List[str] = eqx.field(static=True)

    def __call__(self, x_active_flat, t):
        struct = self.velocity_fn.structure
        offs = self.velocity_fn.block_offsets
        full = jnp.zeros((self.velocity_fn.output_dim,))
        start = 0
        for b in self.active_block_names:
            d = struct.dims[b]
            s, e = offs[b]
            full = full.at[s:e].set(x_active_flat[start:start + d])
            start += d
        for b in self.cond_block_names:
            s, e = offs[b]
            full = full.at[s:e].set(
                interp_trajectory(self.condition_trajs[b], self.traj_ts, t))
        v_full = self.velocity_fn(full, t)
        return jnp.concatenate(
            [v_full[offs[b][0]:offs[b][1]] for b in self.active_block_names])


class DynamicV2Flow(ConditionalFlow):
    """
    Flow for block-structured variables using weight clipping and condition
    interpolation.

    Wraps a :class:`~conditional_flows.parallel.flow.ParallelFlow` to provide
    efficient conditional integration:

    * **Tail clipping** remove blocks not needed for targets
    * **Head conditioning** interpolate parent trajectories instead of
      integrating them
    """
    parallel_flow: ParallelFlow

    def __init__(self, parallel_flow: ParallelFlow):
        super().__init__(parallel_flow.n_steps, parallel_flow.integrator_name)
        self.parallel_flow = parallel_flow

    @property
    def velocity_fn(self):
        """Expose the underlying velocity_fn for compatibility."""
        return self.parallel_flow.velocity_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_condition_trajectories(
        self, conditions: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Backward-integrate conditions to get their full *forward* trajectories
        (t=0 → t=1).

        Delegates to :meth:`trajectory_backward` with ``conditions=None``
        (which takes the unconditional parallel-flow path), then reverses the
        result.

        Args:
            conditions: Dictionary of condition values at t=1.

        Returns:
            Dictionary mapping condition block names to forward trajectories,
            each of shape ``(n_steps+1, dim)``.
        """
        structure = self.velocity_fn.structure
        cond_block_names = list(conditions.keys())

        # Blocks needed: conditions + ALL their ancestors (transitive closure).
        # Using only direct deps was a bug: for a chain x0->x1->x2 with
        # conditions={'x2'}, x1 was omitted from keep_set, causing clip_tail
        # to raise "Cannot remove block 'x1': it is a parent of kept block
        # 'x2'" later in _conditional_trajectory.
        keep_set = structure.get_ancestral_closure(set(cond_block_names))
        keep_blocks = [b for b in structure.block_order if b in keep_set]

        # Build x1 with conditions and zeros for dependency-only blocks
        x1_full: Dict[str, jnp.ndarray] = {}
        for block in keep_blocks:
            if block in conditions:
                x1_full[block] = conditions[block]
            else:
                x1_full[block] = jnp.zeros((structure.dims[block],))

        # Backward integrate unconditionally (conditions=None takes the
        # parallel_flow path — no recursion)
        traj_bwd = self.trajectory_backward(
            x1_full, active_blocks=keep_blocks, conditions=None
        )

        # Reverse to forward order; return trajectories for ALL integrated
        # blocks (including intermediate ancestors such as x1 in
        # x0->x1->x2 when conditions={'x2'}).  _conditional_trajectory
        # then includes these ancestor blocks in cond_block_names /
        # keep_blocks, satisfying clip_tail's requirement that every parent
        # of a kept block is itself kept.
        return {
            block: jnp.flip(traj_bwd[block], axis=0)
            for block in keep_set
        }

    # ------------------------------------------------------------------
    # Shared conditional integration logic
    # ------------------------------------------------------------------

    def _conditional_trajectory(
        self,
        x_boundary: Dict[str, jnp.ndarray],
        active_blocks: List[str],
        conditions: Optional[Dict[str, jnp.ndarray]],
        direction: str,
    ) -> Dict[str, jnp.ndarray]:
        """
        Clip weights, optionally interpolate condition trajectories, and
        integrate.

        Args:
            x_boundary: State dict for *active_blocks* at the integration
                        boundary (t=0 for forward, t=1 for backward).
            active_blocks: Blocks to integrate.
            conditions: Fixed condition values at t=1 (may be ``None`` or
                        empty when only tail clipping is needed).
            direction: ``"forward"`` or ``"backward"``.

        Returns:
            Dictionary mapping block names to trajectories ``(n_steps+1, dim)``.
        """
        structure = self.velocity_fn.structure

        # 1. Get forward condition trajectories (skip when no conditions)
        if conditions:
            condition_trajs = self._get_condition_trajectories(conditions)
            # Exclude any blocks that are also being actively integrated —
            # they must not be double-counted as both condition and active.
            active_set = set(active_blocks)
            cond_block_names = [
                b for b in structure.block_order
                if b in condition_trajs and b not in active_set
            ]
        else:
            condition_trajs = {}
            cond_block_names = []

        # 2.-4. Build the clamped full-network velocity (correctness fix: the
        # former Mode-A clip_tail + SubsetVelocityFn path severed cross-block
        # hidden pathways; see ClampedFullVelocityFn docstring).
        ts = jnp.linspace(0, 1, self.n_steps + 1)
        subset_vfn = ClampedFullVelocityFn(
            velocity_fn=self.velocity_fn,
            condition_trajs={b: condition_trajs[b] for b in cond_block_names},
            traj_ts=ts,
            cond_block_names=cond_block_names,
            active_block_names=active_blocks,
        )

        # 5. Flatten active state & integrate
        x_flat = jnp.concatenate([x_boundary[b] for b in active_blocks])
        flow = Flow(subset_vfn, self.n_steps, self.integrator_name)
        if direction == "forward":
            traj_flat = flow.forward_trajectory(x_flat)
        else:
            traj_flat = flow.backward_trajectory(x_flat)

        # 6. Unflatten trajectories
        result: Dict[str, jnp.ndarray] = {}
        start = 0
        for block in active_blocks:
            dim = structure.dims[block]
            result[block] = traj_flat[:, start : start + dim]
            start += dim

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _covers_all_blocks(
        self,
        active_blocks: List[str],
        conditions: Optional[Dict[str, jnp.ndarray]],
    ) -> bool:
        """True when active + condition blocks span every block (no clipping
        needed)."""
        all_blocks = set(self.velocity_fn.structure.block_order)
        covered = set(active_blocks) | set(conditions or {})
        return covered == all_blocks

    def trajectory_forward(
        self,
        x0: Dict[str, jnp.ndarray],
        active_blocks: Optional[List[str]] = None,
        conditions: Optional[Dict[str, jnp.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute forward trajectories for active blocks with condition
        interpolation and weight clipping.

        Falls back to the full parallel flow only when *all* blocks are
        covered (active ∪ conditions), meaning no tail clipping is required.

        Args:
            x0: Initial state dictionary for active blocks.
            active_blocks: Blocks to integrate.  Defaults to ``list(x0)``.
            conditions: Fixed condition values at t=1.

        Returns:
            Dictionary mapping block names to trajectories (n_steps+1, dim).
        """
        if active_blocks is None:
            active_blocks = list(x0.keys())

        if self._covers_all_blocks(active_blocks, conditions) and not conditions:
            return self.parallel_flow.trajectory_forward(x0)

        return self._conditional_trajectory(
            x0, active_blocks, conditions, direction="forward"
        )

    def trajectory_backward(
        self,
        x1: Dict[str, jnp.ndarray],
        active_blocks: Optional[List[str]] = None,
        conditions: Optional[Dict[str, jnp.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute backward trajectories for active blocks with condition
        interpolation and weight clipping.

        Falls back to the full parallel flow only when *all* blocks are
        covered (active ∪ conditions), meaning no tail clipping is required.

        Args:
            x1: State dictionary at t=1 for active blocks.
            active_blocks: Blocks to integrate.  Defaults to ``list(x1)``.
            conditions: Fixed condition values at t=1.

        Returns:
            Dictionary mapping block names to trajectories (n_steps+1, dim).
        """
        if active_blocks is None:
            active_blocks = list(x1.keys())

        if self._covers_all_blocks(active_blocks, conditions) and not conditions:
            return self.parallel_flow.trajectory_backward(
                x1, active_blocks, conditions, **kwargs
            )

        return self._conditional_trajectory(
            x1, active_blocks, conditions, direction="backward"
        )