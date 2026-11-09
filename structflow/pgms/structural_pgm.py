"""
Structural PGM: A unified framework for PGMs with structural equations.

The core structural equation form is: x_i = f(pa(i), e_i)
where:
    - x_i is the value of node i
    - pa(i) are the parent values (stacked)
    - e_i is independent noise sampled from N(0, I)
    - f is a (possibly nonlinear) mechanism

This module provides:
    - Mechanism: Abstract base class for node-wise mechanisms
    - StructuralPGM: A PGM built from a DAG structure and mechanisms
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
import jax.scipy.stats as stats
import equinox as eqx
from typing import Dict, List, Set, Optional, Tuple, Callable, Any, Union
from abc import abstractmethod

from structflow.pgms.utilities import BasePGM, run_nuts
from structflow.pgms.structure import BlockTriangularStructure, topological_sort
from structflow.bayes.distribution import Distribution


class Mechanism(eqx.Module):
    """
    Abstract base class for structural equation mechanisms.
    
    A mechanism defines the functional relationship x = f(parents, noise).
    
    Attributes:
        parent_names: List of parent variable names (ordered)
        input_dim: Dimension of parent input (sum of parent dims)
        output_dim: Dimension of this node's output
    """
    parent_names: List[str] = eqx.field(static=True)
    input_dim: int = eqx.field(static=True)  # sum of parent dims
    output_dim: int = eqx.field(static=True)  # dimension of this node
    
    @abstractmethod
    def forward(self, parents: jnp.ndarray, noise: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute x = f(parents, noise).
        
        Args:
            parents: Stacked parent values, shape (input_dim,)
            noise: Independent noise, shape (output_dim,)
            
        Returns:
            x: Output value, shape (output_dim,)
            log_det: Log determinant of Jacobian dx/d_noise, scalar
        """
        raise NotImplementedError
    
    @abstractmethod
    def inverse(self, parents: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Recover noise: noise = f^{-1}(parents, x).
        
        Args:
            parents: Stacked parent values, shape (input_dim,)
            x: Output value, shape (output_dim,)
            
        Returns:
            noise: Recovered noise, shape (output_dim,)
            log_det: Log determinant of Jacobian d_noise/dx, scalar (negative of forward)
        """
        raise NotImplementedError
    
    def log_prob(
        self,
        parents: jnp.ndarray,
        x: jnp.ndarray,
        base_dist: Optional[Distribution] = None
    ) -> jnp.ndarray:
        """
        Compute log p(x | parents) using change of variables.
        
        log p(x | parents) = log p(noise) + log |d_noise/dx|
        
        Args:
            parents: Stacked parent values, shape (input_dim,)
            x: Output value, shape (output_dim,)
            base_dist: Optional noise distribution. Defaults to standard Gaussian.
            
        Returns:
            Log probability, scalar
        """
        noise, log_det = self.inverse(parents, x)
        if base_dist is not None:
            log_p_noise = base_dist.log_prob(noise)
        else:
            log_p_noise = stats.norm.logpdf(noise, 0, 1).sum()
        return log_p_noise + log_det


class StructuralPGM(BasePGM):
    """
    A PGM defined by structural equations: x_i = f_i(pa(i), e_i).
    
    This class provides a unified interface for:
    - Linear Gaussian PGMs
    - Nonlinear (in parents) PGMs
    - Nonlinear (in noise) PGMs
    - Fully nonlinear PGMs
    
    The specific behavior is determined by the Mechanism objects.
    
    Attributes:
        structure: BlockTriangularStructure defining DAG and dimensions
        mechanisms: Dict mapping node names to Mechanism objects
        topo_order: Topological ordering of nodes
    """
    structure: BlockTriangularStructure = eqx.field(static=True)
    mechanisms: Dict[str, Mechanism]
    base_distributions: Optional[Dict[str, Distribution]] = None
    topo_order: List[str] = eqx.field(static=True)
    
    def __init__(
        self,
        structure: BlockTriangularStructure,
        mechanisms: Dict[str, Mechanism],
        base_distributions: Optional[Dict[str, Distribution]] = None
    ):
        """
        Initialize a StructuralPGM.
        
        Args:
            structure: BlockTriangularStructure defining DAG and dimensions.
                       Must have structure.dims and structure.dependencies.
            mechanisms: Dict mapping each node name to its Mechanism.
        """
        self.structure = structure
        self.mechanisms = mechanisms
        self.base_distributions = base_distributions
        
        # Use the topological order from structure
        self.topo_order = list(structure.block_order)
        
        # Validate mechanisms match structure
        for name in structure.block_order:
            if name not in mechanisms:
                raise ValueError(f"Missing mechanism for node {name}")
            mech = mechanisms[name]
            expected_dim = structure.dims[name]
            if mech.output_dim != expected_dim:
                raise ValueError(
                    f"Mechanism for {name} has output_dim={mech.output_dim}, "
                    f"but structure.dims[{name}]={expected_dim}"
                )
            parents = structure.dependencies.get(name, [])
            expected_input_dim = sum(structure.dims[p] for p in parents)
            if mech.input_dim != expected_input_dim:
                raise ValueError(
                    f"Mechanism for {name} has input_dim={mech.input_dim}, "
                    f"but expected {expected_input_dim} from parents {parents}"
                )
    
    @property
    def variable_dims(self) -> Dict[str, int]:
        return self.structure.dims
    
    @property
    def node_names(self) -> List[str]:
        return list(self.structure.block_order)
    
    def _get_parents_flat(self, node: str, data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Get stacked parent values for a node."""
        parents = self.structure.dependencies.get(node, [])
        if not parents:
            return jnp.array([])
        return jnp.concatenate([jnp.atleast_1d(data[p]).flatten() for p in parents])
    
    def sample(
        self, 
        key: jax.Array,
        conditions: Optional[Dict[str, jnp.ndarray]] = None,
        targets: Optional[Set[str]] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions).
        
        Uses ancestral sampling for causal queries, MCMC for inference queries.
        """
        if conditions is None:
            conditions = {}
        if targets is None:
            targets = set(self.node_names) - set(conditions.keys())
        
        cond_keys = set(conditions.keys()) if conditions else set()
        if self.structure.is_ancestral_query(targets, cond_keys):
            return self._sample_ancestral(key, conditions, targets)
        else:
            return self._sample_mcmc(key, conditions, targets)
    
    def b_sample(
        self,
        key: jax.Array,
        n_samples: int,
        conditions: Optional[Dict[str, jnp.ndarray]] = None,
        targets: Optional[Set[str]] = None
    ) -> Dict[str, jnp.ndarray]:
        """Batched sampling."""
        if conditions is None:
            conditions = {}
        if targets is None:
            targets = set(self.node_names) - set(conditions.keys())
            
        cond_keys = set(conditions.keys()) if conditions else set()
        if self.structure.is_ancestral_query(targets, cond_keys):
            keys = random.split(key, n_samples)
            return vmap(lambda k: self._sample_ancestral(k, conditions, targets))(keys)
        else:
            # Use MCMC chains for efficient batched sampling
            return self._sample_mcmc(key, conditions, targets, num_samples=n_samples)

    def _sample_ancestral(
        self,
        key: jax.Array,
        conditions: Dict[str, jnp.ndarray],
        targets: Set[str]
    ) -> Dict[str, jnp.ndarray]:
        """Sample via forward ancestral sampling."""
        samples = {k: jnp.atleast_1d(v) for k, v in conditions.items()}
        keys = random.split(key, len(self.topo_order))
        
        for i, node in enumerate(self.topo_order):
            if node in samples:
                continue
            
            # No try/except: if a KeyError occurs here it means is_ancestral_query()
            # returned True incorrectly — we want to surface that bug, not hide it.
            parents_flat = self._get_parents_flat(node, samples)
            dim = self.structure.dims[node]
            
            if self.base_distributions and node in self.base_distributions:
                noise = self.base_distributions[node].sample(keys[i])
            else:
                noise = random.normal(keys[i], (dim,))
                
            val, _ = self.mechanisms[node].forward(parents_flat, noise)
            samples[node] = val
        
        return {k: v for k, v in samples.items() if k in targets}
    
    def _sample_mcmc(
        self,
        key: jax.Array,
        conditions: Dict[str, jnp.ndarray],
        targets: Set[str],
        num_samples: int = 1
    ) -> Dict[str, jnp.ndarray]:
        """Sample via MCMC for non-ancestral queries."""
        unknowns = set(self.node_names) - set(conditions.keys())
        
        def joint_log_prob_fn(unknowns_dict):
            full_data = {**conditions, **unknowns_dict}
            lp = 0.0
            for node in self.topo_order:
                parents_flat = self._get_parents_flat(node, full_data)
                x = jnp.atleast_1d(full_data[node])
                base_dist = self.base_distributions.get(node) if self.base_distributions else None
                lp += self.mechanisms[node].log_prob(parents_flat, x, base_dist=base_dist)
            return lp
        
        # Initialize from ancestral prior
        init_full = self._sample_ancestral(key, {}, unknowns)
        init_pos = {k: init_full[k] for k in unknowns}
        
        chains = run_nuts(joint_log_prob_fn, init_pos, key, num_samples=num_samples)
        
        result = {}
        for k in targets:
            if k in chains:
                # If asking for a single sample, squeeze the batch dimension
                # to be consistent with _sample_ancestral
                if num_samples == 1:
                    result[k] = jnp.squeeze(chains[k], axis=0)
                else:
                    result[k] = chains[k]
            elif k in conditions:
                val = jnp.atleast_1d(conditions[k])
                if num_samples == 1:
                    result[k] = val
                else:
                    result[k] = jnp.tile(val, (num_samples, 1))
        
        return result
    
    def log_prob(
        self,
        data: Dict[str, jnp.ndarray],
        targets: Optional[Set[str]] = None,
        conditions: Optional[Set[str]] = None
    ) -> jnp.ndarray:
        """
        Compute log p(targets | conditions).
        
        Only tractable for ancestral (causal) queries.
        """
        if targets is None:
            if conditions is None:
                targets = set(self.node_names)
            else:
                targets = set(self.node_names) - set(conditions)
        if conditions is None:
            conditions = set()
        
        # Check tractability using structure methods
        cond_set = set(conditions) if conditions else set()
        if not self.structure.is_tractable_query(targets, cond_set):
            raise NotImplementedError(
                "Log-probability is intractable for this query. "
                "Either conditions contain descendants of targets, "
                "or the query requires marginalisation over hidden variables."
            )
        
        total_lp = 0.0
        for node in targets:
            parents_flat = self._get_parents_flat(node, data)
            x = jnp.atleast_1d(data[node])
            base_dist = self.base_distributions.get(node) if self.base_distributions else None
            total_lp += self.mechanisms[node].log_prob(parents_flat, x, base_dist=base_dist)
        
        return total_lp
    
    def counterfactual(
        self,
        data: Dict[str, jnp.ndarray],
        interventions: Dict[str, jnp.ndarray],
        targets: Optional[Set[str]] = None
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Compute counterfactual: what would targets be if we had intervened?
        Returns both the sample and its log probability.
        
        Uses the three-step process:
        1. Abduction: recover noise from observed data
        2. Action: apply interventions
        3. Prediction: propagate through modified SCM
        
        Returns:
            (cf_samples, log_prob) where cf_samples are the counterfactual values
            and log_prob is the log probability p(targets | do(interventions), factual_data).
        """
        intervened_keys = set(interventions.keys())
        
        # Determine which nodes contribute to the probability
        # Intervened nodes are fixed - they don't contribute
        # Only non-intervened nodes (targets) contribute
        if targets is None:
            target_set = set(self.node_names)
        else:
            target_set = set(targets)
        
        # Non-intervened targets are the ones whose probability we compute
        non_intervened_targets = target_set - intervened_keys
        
        # Identify nodes that are relevant for the targets in the counterfactual graph.
        # This allows us to handle partial factual data where irrelevant variables are missing.
        # We need the ancestral closure of targets, stopping at intervened nodes.
        relevant_nodes = self.structure.get_ancestral_closure(target_set, exclude=intervened_keys)
        
        # 1. Abduction: recover noise terms for relevant non-intervened nodes
        noises = {}
        for node in self.topo_order:
            if node in relevant_nodes and node not in intervened_keys:
                parents_flat = self._get_parents_flat(node, data)
                x = jnp.atleast_1d(data[node])
                noise, _ = self.mechanisms[node].inverse(parents_flat, x)
                noises[node] = noise
        
        # 2. Action & 3. Prediction
        cf_samples = {}
        for node in self.topo_order:
            if node in intervened_keys:
                cf_samples[node] = jnp.atleast_1d(interventions[node])
            elif node in relevant_nodes:
                parents_flat = self._get_parents_flat(node, cf_samples)
                val, _ = self.mechanisms[node].forward(parents_flat, noises[node])
                cf_samples[node] = val
        
        # 4. Compute log probability of counterfactual for NON-INTERVENED targets only
        # Intervened nodes are fixed by the do() operator - no probability contribution
        log_prob = 0.0
        for node in non_intervened_targets:
            parents_flat = self._get_parents_flat(node, cf_samples)
            x = jnp.atleast_1d(cf_samples[node])
            # Use mechanism's log_prob which handles change of variables correctly
            base_dist = self.base_distributions.get(node) if self.base_distributions else None
            log_prob += self.mechanisms[node].log_prob(parents_flat, x, base_dist=base_dist)
        
        sample_dict = {k: v for k, v in cf_samples.items() if k in target_set}
            
        return sample_dict, log_prob
    
    def get_pgm_structure(self) -> Dict[str, List[str]]:
        """Return the DAG structure."""
        return dict(self.structure.dependencies)
    
    def get_inference_targets_and_conditions(self) -> Tuple[Set[str], Set[str]]:
        """
        Default inference task: p(roots | leaves).
        """
        structure = self.get_pgm_structure()
        roots = {node for node, parents in structure.items() if not parents}
        
        # Leaves are nodes that are not parents of any other node
        all_parents = set()
        for parents in structure.values():
            all_parents.update(parents)
        leaves = set(self.node_names) - all_parents
        
        return roots, leaves - roots
    
    def get_supported_distributions(self) -> List[Dict[str, frozenset]]:
        """
        Returns all tractable distributions for this PGM.
        """
        from structflow.pgms.pgm_compiler import PGMCompiler
        
        supported = set()
        compiler = PGMCompiler(self.structure.dependencies, self.structure.dims)
        
        # Causal distributions
        causal_struct = compiler.compile_causal()
        for dist in causal_struct.get_tractable_distributions():
            supported.add((dist['targets'], dist['conditions']))
        
        # Inference distributions
        targets, conditions = self.get_inference_targets_and_conditions()
        if targets:
            inference_struct = compiler.compile_inference(targets, conditions)
            for dist in inference_struct.get_tractable_distributions():
                supported.add((dist['targets'], dist['conditions']))
        
        # Minimal set
        supported.add((frozenset(self.node_names), frozenset()))
        for child, parents in self.structure.dependencies.items():
            if parents:
                supported.add((frozenset([child]), frozenset(parents)))
                supported.add((frozenset(parents), frozenset([child])))
        
        return [{"targets": t, "conditions": c} for t, c in sorted(list(supported))]
