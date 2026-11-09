"""
Unified PGM API - Clean interface for all PGM types.

This module re-exports the unified StructuralPGM framework and factory functions.

Usage:
    # Create a linear chain PGM
    pgm = make_chain_pgm(n_nodes=5, mechanism_type="linear", key=key)
    
    # Create a nonlinear random PGM  
    pgm = make_random_pgm(n_nodes=4, mechanism_type="nonlinear_parent", key=key, sparsity=0.3)
    
    # Sample from the joint
    samples = pgm.sample(key)
    
    # Compute log probability
    lp = pgm.log_prob(samples)
    
    # Compute counterfactual
    cf = pgm.counterfactual(data, interventions={"x0": jnp.array([1.0])})
"""

# Core classes
from structflow.pgms.structural_pgm import (
    Mechanism,
    StructuralPGM,
)

# Mechanism implementations
from structflow.pgms.mechanisms import (
    LinearMechanism,
    NonlinearParentMechanism,
    NonlinearNoiseMechanism,
    UniversalMechanism,
    make_linear_mechanism,
    make_nonlinear_parent_mechanism,
    make_nonlinear_noise_mechanism,
    make_universal_mechanism,
)

# Factory functions for PGMs
from structflow.pgms.factories import (
    # Structure builders
    make_chain_structure,
    make_fork_structure,
    make_collider_structure,
    make_random_dag_structure,
    make_dense_structure,
    
    # PGM factories (generic)
    make_structural_pgm,
    make_chain_pgm,
    make_fork_pgm,
    make_collider_pgm,
    make_random_pgm,
    make_dense_pgm,
    
    # Convenience aliases
    make_linear_chain_pgm,
    make_nonlinear_chain_pgm,
    make_universal_chain_pgm,
    make_linear_random_pgm,
    make_nonlinear_random_pgm,
    make_universal_random_pgm,
    
    # Type for mechanism selection
    MechanismType,
    get_mechanism_constructor,
)

# Utilities
from structflow.pgms.utilities import (
    BasePGM,
    run_nuts,
)

__all__ = [
    # Core
    "Mechanism",
    "StructuralPGM",
    "BasePGM",
    
    # Mechanisms
    "LinearMechanism",
    "NonlinearParentMechanism", 
    "NonlinearNoiseMechanism",
    "UniversalMechanism",
    "make_linear_mechanism",
    "make_nonlinear_parent_mechanism",
    "make_nonlinear_noise_mechanism",
    "make_universal_mechanism",
    
    # Structure factories
    "make_chain_structure",
    "make_fork_structure",
    "make_collider_structure",
    "make_random_dag_structure",
    "make_dense_structure",
    
    # PGM factories
    "make_structural_pgm",
    "make_chain_pgm",
    "make_fork_pgm",
    "make_collider_pgm",
    "make_random_pgm",
    "make_dense_pgm",
    
    # Convenience
    "make_linear_chain_pgm",
    "make_nonlinear_chain_pgm",
    "make_universal_chain_pgm",
    "make_linear_random_pgm",
    "make_nonlinear_random_pgm",
    "make_universal_random_pgm",
    
    # Types
    "MechanismType",
    "get_mechanism_constructor",
    
    # Utilities
    "run_nuts",
]
