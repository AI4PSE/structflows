import pytest
from structflow.pgms.structure import BlockTriangularStructure

def test_chain_tractability():
    """
    Test tractability of various queries on a 10-node chain PGM.
    x0 -> x1 -> x2 -> x3 -> x4 -> x5 -> x6 -> x7 -> x8 -> x9
    """
    n = 10
    block_order = [f'x{i}' for i in range(n)]
    dependencies = {f'x{i}': ([f'x{i-1}'] if i > 0 else []) for i in range(n)}
    dims = {f'x{i}': 1 for i in range(n)}
    
    structure = BlockTriangularStructure(block_order, dependencies, dims)
    
    # User's examples
    
    # 1. p(x0, x1) -> Tractable (Joint of ancestral set)
    assert structure.is_tractable_query(targets={'x0', 'x1'}, conditions=set()) is True
    
    # 2. p(x4 | x3, x2) -> Tractable (Forward conditional)
    assert structure.is_tractable_query(targets={'x4'}, conditions={'x3', 'x2'}) is True
    
    # 3. p(x5, x6, x7 | x2, x3, x4) -> Tractable (Forward conditional block)
    assert structure.is_tractable_query(targets={'x5', 'x6', 'x7'}, conditions={'x2', 'x3', 'x4'}) is True
    
    # 4. p(x9 | x0) -> Intractable (Requires marginalizing out x1...x8 or inverting flow)
    # Parent Rule: Parent of x9 is x8, but x8 is NOT in {x9, x0}.
    assert structure.is_tractable_query(targets={'x9'}, conditions={'x0'}) is False
    
    # 5. p(x0 | x9) -> Intractable (Descendant Rule failure)
    # x9 is a descendant of x0.
    assert structure.is_tractable_query(targets={'x0'}, conditions={'x9'}) is False

    # 6. Full joint
    assert structure.is_tractable_query(targets=set(block_order), conditions=set()) is True

    # 7. Prefix Joint
    assert structure.is_tractable_query(targets={'x0', 'x1', 'x2'}, conditions=set()) is True

def test_chain_ancestral_vs_tractable():
    """
    Test is_ancestral_query vs is_tractable_query on chain.
    Key case: p(x2 | x0) is ancestral but NOT tractable (requires marginalisation).
    """
    structure = BlockTriangularStructure(
        block_order=['x0', 'x1', 'x2'],
        dependencies={'x0': [], 'x1': ['x0'], 'x2': ['x1']},
        dims={'x0': 1, 'x1': 1, 'x2': 1}
    )
    
    # p(x2 | x0): ancestral=True (x0 is upstream), marginalisation=True (x1 missing), tractable=False
    assert structure.is_ancestral_query({'x2'}, {'x0'}) is True
    assert structure.query_requires_marginalisation({'x2'}, {'x0'}) is True
    assert structure.is_tractable_query({'x2'}, {'x0'}) is False
    
    # p(x0 | x2): ancestral=False (x2 is downstream of x0)
    assert structure.is_ancestral_query({'x0'}, {'x2'}) is False
    assert structure.is_tractable_query({'x0'}, {'x2'}) is False
    
    # p(x2 | x1): ancestral=True, marginalisation=False → tractable=True
    assert structure.is_ancestral_query({'x2'}, {'x1'}) is True
    assert structure.query_requires_marginalisation({'x2'}, {'x1'}) is False
    assert structure.is_tractable_query({'x2'}, {'x1'}) is True
    
    # p(x1, x2 | x0): ancestral=True, marginalisation=False → tractable=True
    assert structure.is_ancestral_query({'x1', 'x2'}, {'x0'}) is True
    assert structure.query_requires_marginalisation({'x1', 'x2'}, {'x0'}) is False
    assert structure.is_tractable_query({'x1', 'x2'}, {'x0'}) is True

def test_chain_10_node_ancestral():
    """Test ancestral queries on a 10-node chain."""
    n = 10
    block_order = [f'x{i}' for i in range(n)]
    dependencies = {f'x{i}': ([f'x{i-1}'] if i > 0 else []) for i in range(n)}
    dims = {f'x{i}': 1 for i in range(n)}
    
    structure = BlockTriangularStructure(block_order, dependencies, dims)
    
    # p(x9 | x0): ancestral=True (x0 is upstream), marginalisation=True
    assert structure.is_ancestral_query({'x9'}, {'x0'}) is True
    assert structure.query_requires_marginalisation({'x9'}, {'x0'}) is True
    assert structure.is_tractable_query({'x9'}, {'x0'}) is False
    
    # p(x0 | x9): NOT ancestral (x9 is downstream)
    assert structure.is_ancestral_query({'x0'}, {'x9'}) is False

def test_collider_tractability():
    """ x1 -> y, x2 -> y """
    structure = BlockTriangularStructure(
        block_order=['x1', 'x2', 'y'],
        dependencies={'x1': [], 'x2': [], 'y': ['x1', 'x2']},
        dims={'x1': 1, 'x2': 1, 'y': 1}
    )
    
    # p(y | x1, x2) -> Tractable
    assert structure.is_tractable_query(targets={'y'}, conditions={'x1', 'x2'}) is True
    
    # p(x1 | y) -> Intractable (Descendant rule: y is descendant of x1)
    assert structure.is_tractable_query(targets={'x1'}, conditions={'y'}) is False
    
    # p(x1, x2 | y) -> Intractable (Descendant rule: y is descendant of x1 and x2)
    assert structure.is_tractable_query(targets={'x1', 'x2'}, conditions={'y'}) is False

def test_collider_ancestral():
    """Test ancestral queries on collider: x1 -> y, x2 -> y"""
    structure = BlockTriangularStructure(
        block_order=['x1', 'x2', 'y'],
        dependencies={'x1': [], 'x2': [], 'y': ['x1', 'x2']},
        dims={'x1': 1, 'x2': 1, 'y': 1}
    )
    
    # p(x1 | y): NOT ancestral (y is descendant of x1)
    assert structure.is_ancestral_query({'x1'}, {'y'}) is False
    
    # p(y | x1): ancestral=True, marginalisation=True (x2 is parent of y, missing)
    assert structure.is_ancestral_query({'y'}, {'x1'}) is True
    assert structure.query_requires_marginalisation({'y'}, {'x1'}) is True
    assert structure.is_tractable_query({'y'}, {'x1'}) is False
    
    # p(y | x1, x2): ancestral=True, marginalisation=False → tractable=True
    assert structure.is_ancestral_query({'y'}, {'x1', 'x2'}) is True
    assert structure.query_requires_marginalisation({'y'}, {'x1', 'x2'}) is False
    assert structure.is_tractable_query({'y'}, {'x1', 'x2'}) is True

def test_reporting():
    """ Test that get_all_tractable_distributions and get_base_tractable_distributions work. """
    from structflow.pgms.structure import (
        get_base_tractable_distributions, 
        get_all_tractable_distributions
    )
    structure = BlockTriangularStructure(
        block_order=['x0', 'x1', 'x2'],
        dependencies={'x0': [], 'x1': ['x0'], 'x2': ['x1']},
        dims={'x0': 1, 'x1': 1, 'x2': 1}
    )
    
    # Test Base Distributions
    base_tractable = get_base_tractable_distributions(structure)
    expected_base = [
        {"targets": frozenset({"x0"}), "conditions": frozenset()},
        {"targets": frozenset({"x1"}), "conditions": frozenset({"x0"})},
        {"targets": frozenset({"x2"}), "conditions": frozenset({"x1"})}
    ]
    assert base_tractable == expected_base
    
    # Test All Tractable Distributions (Generator)
    all_tractable = list(get_all_tractable_distributions(structure))
    
    # Prefix Joints
    assert {"targets": frozenset({"x0"}), "conditions": frozenset()} in all_tractable
    assert {"targets": frozenset({"x0", "x1"}), "conditions": frozenset()} in all_tractable
    assert {"targets": frozenset({"x0", "x1", "x2"}), "conditions": frozenset()} in all_tractable
    
    # Contiguous conditional blocks
    assert {"targets": frozenset({"x1"}), "conditions": frozenset({"x0"})} in all_tractable
    assert {"targets": frozenset({"x1", "x2"}), "conditions": frozenset({"x0"})} in all_tractable
    assert {"targets": frozenset({"x2"}), "conditions": frozenset({"x0", "x1"})} in all_tractable

    # Should NOT contain rules or sections anymore
    assert "Tractability Rules:" not in all_tractable
    assert "Prefix Joints" not in all_tractable