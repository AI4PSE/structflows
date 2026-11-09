"""
CAREFL Distribution built on the generalized Mechanism framework.

This module provides a CAREFLDistribution that wraps StructuralPGM 
with CAREFL mechanisms, reusing all the sampling, log_prob, and 
counterfactual inference logic from the PGM framework.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Optional, Union, Set
import equinox as eqx

from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.dist import ConditionalDistribution, StructuredPGMDistribution
from structflow.pgms.structure import Structure, BlockTriangularStructure, topological_sort
from structflow.pgms.structural_pgm import StructuralPGM
from structflow.pgms.mechanisms import CAREFLMechanism, ConditionerMechanism



class CAREFLDistribution(StructuredPGMDistribution):
    """
    Causal Autoregressive Flow (CAREFL) Distribution.
    
    A lightweight wrapper around StructuralPGM with CAREFL mechanisms.
    This reuses all the sampling, log_prob, and counterfactual inference
    logic from the generalized PGM framework.
    
    Implements a Structural Equation Model (SEM) using affine autoregressive flows:
        x_j = exp(s_j(pa_j)) * z_j + t_j(pa_j)
        
    where z_j is drawn from a base distribution (e.g., Laplace or Gaussian),
    and pa_j are the parents of x_j in the causal DAG.
    
    The API is kept compatible with the original CAREFL implementation,
    but internally leverages the generalized Mechanism framework.
    
    Reference: Khemakhem et al., "Causal Autoregressive Flows", AISTATS 2021.
    """
    # Expose original API fields for backward compatibility
    structure: Dict[str, List[str]] = eqx.field(static=True)
    topological_order: List[str] = eqx.field(static=True)
    conditioners: Dict[str, eqx.Module] = eqx.field(static=False)
    base_distributions: Dict[str, Distribution] = eqx.field(static=True)
    
    def __init__(
        self, 
        structure: Union[Dict[str, List[str]], Structure], 
        conditioners: Dict[str, eqx.Module],
        base_distributions: Dict[str, Distribution],
        topological_order: Optional[List[str]] = None
    ):
        """
        Initialize a CAREFLDistribution from a structure and conditioners.
        
        Internally creates a StructuralPGM with CAREFL mechanisms from the
        provided conditioners.
        
        Args:
            structure: Either a Structure object or a dictionary mapping node names 
                       to a list of parent names.
            conditioners: Dictionary mapping node names to Neural Networks (eqx.Modules).
                         Each network should output 2 * dim(x_j) for (s, t).
            base_distributions: Dictionary mapping node names to base noise distributions.
            topological_order: List of nodes in causal order. If None, inferred from structure.
                              Ignored if structure is a Structure object.
        """
        if isinstance(structure, Structure):
            self.structure = structure.dependencies
            self.topological_order = structure.block_order
            dims = structure.dims
        else:
            self.structure = structure
            if topological_order is None:
                self.topological_order = topological_sort(structure)
            else:
                self.topological_order = topological_order
            
            # Infer dims from base_distributions
            dims = {node: bd.dim for node, bd in base_distributions.items()}
        
        self.conditioners = conditioners
        self.base_distributions = base_distributions
        
        # Validation
        assert set(self.structure.keys()) == set(self.conditioners.keys()), \
            "Structure nodes must match conditioner nodes"
        assert set(self.structure.keys()) == set(self.base_distributions.keys()), \
            "Structure nodes must match base distribution nodes"
        
        # Build BlockTriangularStructure for internal PGM
        pgm_structure = BlockTriangularStructure(
            block_order=self.topological_order,
            dependencies=self.structure,
            dims=dims
        )
        
        # Build CAREFL mechanisms from conditioners
        mechanisms = {}
        for node in self.topological_order:
            parent_names = self.structure.get(node, [])
            parent_dims = [dims[p] for p in parent_names]
            output_dim = dims[node]
            
            # Create a CAREFLMechanism wrapper around the conditioner
            mechanism = ConditionerMechanism(
                parent_names=parent_names,
                input_dim=sum(parent_dims),
                output_dim=output_dim,
                conditioner=self.conditioners[node]
            )
            mechanisms[node] = mechanism
        
        # Create internal StructuralPGM
        self.pgm = StructuralPGM(pgm_structure, mechanisms)
