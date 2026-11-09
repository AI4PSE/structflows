"""Backbone-agnostic counterfactual graph logic.

The Abduction–Action–Prediction procedure for a structured (block-triangular)
model factors cleanly into

* pure graph traversal — ancestral closure of the query targets, descendant
  sets of the intervened nodes, selection of the abduction blocks and of the
  effective conditioning values — which depends only on the ``Structure``; and
* two backbone-specific primitives — recovering per-block exogenous noise from
  observed data (*abduction*), and re-simulating the modified system forward
  from that noise (*prediction*).

``CounterfactualMixin`` implements the former once and delegates the latter to
abstract hooks, so that every backbone (CNF blocks, per-node diffusion, ...)
runs the *identical* graph logic. This is load-bearing for backbone ablations:
if each backbone reimplemented the traversal, any behavioural difference would
be confounded with implementation drift.

The code here is a verbatim lift of the graph logic formerly inlined in
``StaticConditionalDistribution.counterfactual``; see
``conditional_flows/tests/test_static_cf_regression.py`` for the bit-identical
regression harness.
"""

import jax.numpy as jnp
from typing import Dict, List, Optional, Set, Tuple, Any


class CounterfactualMixin:
    """Mixin providing ``counterfactual`` via Abduction–Action–Prediction.

    Subclasses must implement:

    * ``_cf_structure`` (property): the ``Structure`` describing the graph.
    * ``_abduct(data, blocks)``: recover exogenous noise for ``blocks`` (a set)
      from observed ``data``; returns ``{block: noise}``.
    * ``_predict(x0_sim, conditions, simulation_blocks)``: simulate
      ``simulation_blocks`` (a list) forward from noise ``x0_sim`` with fixed
      ``conditions``; returns ``(results, aux)`` where ``results`` maps each
      simulation block to its value and ``aux`` is arbitrary backbone state
      passed through to ``_predict_log_prob``.
    * ``_predict_log_prob(x0_sim, simulation_blocks, results, aux)``: log
      probability of the counterfactual sample (may be ``jnp.nan`` for
      backbones without a tractable density).
    """

    # --- Abstract hooks (plain-Python: raise before any tracing) ---

    @property
    def _cf_structure(self):
        raise NotImplementedError

    def _abduct(self, data: Dict[str, jnp.ndarray],
                blocks: Set[str]) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    def _predict(self, x0_sim: Dict[str, jnp.ndarray],
                 conditions: Dict[str, jnp.ndarray],
                 simulation_blocks: List[str]) -> Tuple[Dict[str, jnp.ndarray], Any]:
        raise NotImplementedError

    def _predict_log_prob(self, x0_sim: Dict[str, jnp.ndarray],
                          simulation_blocks: List[str],
                          results: Dict[str, jnp.ndarray],
                          aux: Any) -> jnp.ndarray:
        raise NotImplementedError

    # --- Shared counterfactual logic ---

    def counterfactual(self,
                       data: Dict[str, jnp.ndarray],
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Set[str]] = None,
                       compute_log_prob: bool = True) -> tuple:
        """
        Perform counterfactual query: p(targets | data, do(interventions)).
        Returns both the sample and its log probability.

        Uses the Abduction-Action-Prediction framework:
        1. Abduction: Recover latent noise u from data given the structural
           equations (backbone-specific inverse).
        2. Action: Interventions fix the values of specific variables.
        3. Prediction: Simulate the modified system forward using the
           recovered noise u (backbone-specific forward).

        Args:
            data: Observed data values for a single example.
            interventions: Values to fix for specific variables.
            targets: Variables to query. If None, returns all variables.
            compute_log_prob: If False, skip the (possibly expensive) density
                computation; log_prob comes back as NaN.

        Returns:
            (sample_dict, log_prob) where sample_dict is the counterfactual
            sample and log_prob is the log probability of the sample.
        """
        structure = self._cf_structure
        all_blocks = structure.block_order
        intervened_keys = set(interventions.keys())

        # Identify targets
        if targets is None:
            targets_query = set(all_blocks)
        else:
            targets_query = set(targets)

        # 1. Abduction
        # Identify which blocks need noise recovery.
        # We need noise for:
        # a) Targets (to compute log_prob and/or re-simulate)
        # b) Ancestors of targets that will change due to interventions

        # Get ancestral closure (nodes that affect targets in the intervened graph)
        relevant_ancestors = structure.get_ancestral_closure(targets_query, exclude=intervened_keys)

        # Identify nodes affected by interventions
        changed_nodes = set()
        if intervened_keys:
            for k in intervened_keys:
                changed_nodes.add(k)
                changed_nodes.update(structure.get_all_descendants(k))

        # Abduction blocks: (Relevant AND Changed) OR Targets
        abduction_blocks = (relevant_ancestors & changed_nodes) | targets_query

        # Remove intervened blocks (we don't need their noise)
        abduction_blocks = abduction_blocks - intervened_keys

        # Recover noise only for these blocks (backbone hook)
        if abduction_blocks:
            noise_obs = self._abduct(data, abduction_blocks)
        else:
            noise_obs = {}

        # 2. Action & 3. Prediction

        # We simulate the blocks for which we have noise.
        # Other relevant blocks (unchanged ancestors) are treated as conditions.
        simulation_blocks = list(abduction_blocks)

        x0_sim = {k: noise_obs[k] for k in simulation_blocks}

        # Prepare conditions for simulation:
        # Interventions + Unchanged Relevant Ancestors (from factual data)
        effective_conditions = dict(interventions)

        # Unchanged ancestors = relevant_ancestors - abduction_blocks - intervened_keys
        unchanged_ancestors = relevant_ancestors - abduction_blocks - intervened_keys

        for k in unchanged_ancestors:
            if k in data:
                effective_conditions[k] = data[k]
            # If k is missing from data but needed, the backbone will raise later.

        sim_results, aux = self._predict(x0_sim, effective_conditions, simulation_blocks)

        # Combine Results
        results = {}
        # Simulation results
        for k in simulation_blocks:
            results[k] = sim_results[k]

        # Intervention values
        for k in intervened_keys:
            results[k] = interventions[k]

        # 4. Log probability (backbone hook). Optional: samples-only callers
        # set compute_log_prob=False and get log_prob = NaN.
        if not compute_log_prob:
            sample_dict = results if targets is None else {k: results[k] for k in targets if k in results}
            return sample_dict, jnp.array(jnp.nan)

        log_prob = self._predict_log_prob(x0_sim, simulation_blocks, results, aux)

        # 5. Filter targets and return
        sample_dict = results if targets is None else {k: results[k] for k in targets if k in results}

        return sample_dict, log_prob
