"""DCM (per-node diffusion / Diff-SCM style) conditional distribution.

Drop-in ``ConditionalDistribution`` backbone for the StructFlow backbone
ablation: identical compiled structure (``StructuredVelocityFn`` over
``BlockTriangularStructure``), identical counterfactual graph logic
(``CounterfactualMixin``), with the per-block continuous-time generative
primitive swapped from a CNF to a per-node DDIM chain.

Capabilities (this is the point of the ablation):

* ``sample``       — topological loop of ``ddim_forward`` per block.
* ``transform_to_noise`` — per-block ``ddim_invert``.
* ``counterfactual`` — from the mixin: abduct via ``ddim_invert``, predict via
  ``ddim_forward``. Samples are well-defined; the log_prob is NaN.
* ``log_prob``     — **NotImplementedError**, raised eagerly from plain
  Python before any tracing. Per-node diffusion has no tractable conditional
  density; this is a real capability gap and must surface as one, never as a
  silent zero. Callers (e.g. ``pgm_paper_utils/eval.py``) must decide this
  capability statically and never call ``log_prob`` on a DCMDistribution.
"""

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from typing import Dict, List, Optional, Set, Union

from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.dist import ConditionalDistribution
from structflow.conditional_flows.counterfactual import CounterfactualMixin
from structflow.conditional_flows.static.net import StructuredVelocityFn
from structflow.conditional_flows.dcm.sampler import ddim_forward, ddim_invert


class DCMDistribution(CounterfactualMixin, ConditionalDistribution):
    """Distribution defined by one conditional DDIM chain per graph block.

    Attributes:
        eps_fn: StructuredVelocityFn container of DiffusionBlocks (the
            structure handling is shared with the CNF backbone verbatim).
        base_distributions: block name -> base (exogenous noise) distribution.
        n_steps: number of DDIM steps (NFE) used for sampling and inversion.
    """
    eps_fn: StructuredVelocityFn
    base_distributions: Dict[str, Distribution]
    n_steps: int = eqx.field(static=True)

    def __init__(self,
                 eps_fn: StructuredVelocityFn,
                 base_distributions: Dict[str, Distribution],
                 n_steps: int = 50):
        if not hasattr(eps_fn, 'structure'):
            raise TypeError(f"eps_fn must expose .structure, got {type(eps_fn)}")
        if not isinstance(base_distributions, dict):
            raise TypeError(f"base_distributions must be a dict, got {type(base_distributions)}")
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(f"n_steps must be a positive int, got {n_steps}")

        structure_blocks = set(eps_fn.structure.block_order)
        base_blocks = set(base_distributions.keys())
        if structure_blocks != base_blocks:
            missing = structure_blocks - base_blocks
            extra = base_blocks - structure_blocks
            raise ValueError(f"Base distributions mismatch. Missing: {missing}, Extra: {extra}")

        self.eps_fn = eps_fn
        self.base_distributions = base_distributions
        self.n_steps = n_steps

    # --- Internal simulation helpers ---

    def _simulate(self,
                  x0: Dict[str, jnp.ndarray],
                  conditions: Dict[str, jnp.ndarray],
                  blocks) -> Dict[str, jnp.ndarray]:
        """Topological loop of per-block ddim_forward, feeding each block's
        output as conditioning context for its descendants (mirrors
        StaticConditionalFlow.trajectory_forward's autoregressive sequencing)."""
        structure = self.eps_fn.structure
        block_set = set(blocks)
        sorted_blocks = [b for b in structure.block_order if b in block_set]
        if len(sorted_blocks) != len(block_set):
            raise ValueError(f"Unknown blocks requested: {block_set - set(structure.block_order)}")

        current = dict(conditions)
        out = {}
        blocks_unstacked = self.eps_fn.blocks  # one gather, outside the DDIM loops
        for block in sorted_blocks:
            parent_names = structure.dependencies.get(block, [])
            parents = tuple(current[p] for p in parent_names)
            x = ddim_forward(blocks_unstacked[block], x0[block], parents, self.n_steps)
            out[block] = x
            current[block] = x
        return out

    def _invert_blocks(self,
                       data: Dict[str, jnp.ndarray],
                       blocks) -> Dict[str, jnp.ndarray]:
        """Per-block ddim_invert; parents are the observed (factual) values."""
        structure = self.eps_fn.structure
        noises = {}
        blocks_unstacked = self.eps_fn.blocks  # one gather, outside the DDIM loops
        for block in blocks:
            parent_names = structure.dependencies.get(block, [])
            parents = tuple(data[p] for p in parent_names)
            noises[block] = ddim_invert(blocks_unstacked[block], data[block],
                                        parents, self.n_steps)
        return noises

    # --- ConditionalDistribution API ---

    def sample(self,
               key: jax.Array,
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """Sample from p(targets | conditions) by ancestral per-block DDIM."""
        structure = self.eps_fn.structure
        all_blocks = set(structure.block_order)

        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_blocks)

        if not structure.is_ancestral_query(target_set, cond_keys):
            raise NotImplementedError("Intractable query: conditions depend on targets (requires inversion).")

        simulation_blocks = structure.get_required_simulation_blocks(target_set, cond_keys)

        keys = random.split(key, len(simulation_blocks))
        x0_sim = {}
        for i, block in enumerate(simulation_blocks):
            x0_sim[block] = self.base_distributions[block].sample(keys[i])

        result = self._simulate(x0_sim, conditions or {}, simulation_blocks)

        return {t: result[t] for t in target_set if t in result}

    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        # NOTE: raised eagerly from plain Python, before any array touches a
        # tracer. Never move this below jax code and never catch it inside a
        # jitted/vmapped caller (a mid-trace exception poisons XLA state).
        raise NotImplementedError(
            "DCMDistribution has no tractable log_prob: the per-node DDIM "
            "backbone exposes no conditional density (no amortized posterior, "
            "no moralization). This is a real capability gap of the diffusion "
            "backbone; decide capabilities statically before calling."
        )

    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Maps Real Data -> Noise for a single example via per-block DDIM inversion."""
        return self._invert_blocks(x, self.eps_fn.structure.block_order)

    # --- Counterfactual backbone hooks (graph logic lives in CounterfactualMixin) ---

    @property
    def _cf_structure(self):
        return self.eps_fn.structure

    def _abduct(self, data: Dict[str, jnp.ndarray],
                blocks: Set[str]) -> Dict[str, jnp.ndarray]:
        return self._invert_blocks(data, list(blocks))

    def _predict(self, x0_sim: Dict[str, jnp.ndarray],
                 conditions: Dict[str, jnp.ndarray],
                 simulation_blocks: List[str]) -> tuple:
        results = self._simulate(x0_sim, conditions, simulation_blocks)
        return results, None

    def _predict_log_prob(self, x0_sim, simulation_blocks, results, aux) -> jnp.ndarray:
        # Counterfactual *samples* are well-defined for the DDIM backbone but
        # the density is not; report NaN (matching the static backbone's
        # samples-only contract) rather than raising from traced code.
        return jnp.array(jnp.nan)
