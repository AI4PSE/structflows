"""Synthetic PGM benchmarks and the PGM-to-flow compiler.

Build a :class:`StructuralPGM` with known structural equations using the
``make_*`` factory functions, then compile its DAG into a structured
block-triangular flow with :class:`PGMCompiler` (causal- or
inference-aligned). See ``pgms/README.md`` for the available mechanisms
(linear, nonlinear-parent, nonlinear-noise, universal) and structures.
"""

from structflow.pgms.structure import (
    Structure,
    BlockTriangularStructure,
    topological_sort,
    get_base_tractable_distributions,
    get_all_tractable_distributions,
)
from structflow.pgms.structural_pgm import StructuralPGM, Mechanism
from structflow.pgms.pgm_compiler import (
    PGMCompiler,
    dag_to_structure,
    compile_inference_structure,
)
from structflow.pgms.factories import (
    MechanismType,
    get_mechanism_constructor,
    make_chain_structure,
    make_fork_structure,
    make_collider_structure,
    make_random_dag_structure,
    make_dense_structure,
    make_structural_pgm,
    make_chain_pgm,
    make_fork_pgm,
    make_collider_pgm,
    make_random_pgm,
    make_dense_pgm,
    make_linear_chain_pgm,
    make_nonlinear_chain_pgm,
    make_universal_chain_pgm,
    make_linear_random_pgm,
    make_nonlinear_random_pgm,
    make_universal_random_pgm,
)

__version__ = "0.1.0"

__all__ = [
    "Structure",
    "BlockTriangularStructure",
    "topological_sort",
    "get_base_tractable_distributions",
    "get_all_tractable_distributions",
    "StructuralPGM",
    "Mechanism",
    "PGMCompiler",
    "dag_to_structure",
    "compile_inference_structure",
    "MechanismType",
    "get_mechanism_constructor",
    "make_chain_structure",
    "make_fork_structure",
    "make_collider_structure",
    "make_random_dag_structure",
    "make_dense_structure",
    "make_structural_pgm",
    "make_chain_pgm",
    "make_fork_pgm",
    "make_collider_pgm",
    "make_random_pgm",
    "make_dense_pgm",
    "make_linear_chain_pgm",
    "make_nonlinear_chain_pgm",
    "make_universal_chain_pgm",
    "make_linear_random_pgm",
    "make_nonlinear_random_pgm",
    "make_universal_random_pgm",
]
