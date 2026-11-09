import pytest
import networkx as nx
from structflow.pgms.pgm_compiler import PGMCompiler

def test_independent_nodes():
    """
    k independent nodes in the PGM (entirely disconnected) should result in a diagonal map / no dependencies in the structure.
    """
    # 3 independent nodes
    pgm_dag = {
        'x1': [],
        'x2': [],
        'x3': []
    }
    compiler = PGMCompiler(pgm_dag)
    
    # Query: Joint distribution p(x1, x2, x3)
    targets = {'x1', 'x2', 'x3'}
    conditions = set()
    
    structure = compiler.compile_inference(targets, conditions)
    
    # Check that all dependencies are empty
    for block in structure.block_order:
        assert structure.dependencies[block] == [], f"Block {block} should have no dependencies"
        
    # Check that all variables are present
    assert set(structure.block_order) == {'x1', 'x2', 'x3'}

def test_cycle_error():
    """
    A loop in the PGM DAG should result in an error being raised?
    Currently PGMCompiler uses networkx topological sort which raises NetworkXUnfeasible if there is a cycle.
    """
    # Cycle: x1 -> x2 -> x1
    pgm_dag = {
        'x1': ['x2'],
        'x2': ['x1']
    }
    compiler = PGMCompiler(pgm_dag)
    
    targets = {'x1', 'x2'}
    conditions = set()
    
    # Expect a NetworkX error or a recursion error depending on implementation,
    # but since it calls topological_sort, it should be NetworkXUnfeasible
    with pytest.raises(nx.NetworkXUnfeasible):
        compiler.compile_inference(targets, conditions)

def test_unconditional_ordering():
    """
    Unconditional variables (ones with no parents) should always be sorted to the front!?
    """
    # x1 -> x2. x3 is independent.
    # x3 and x1 are unconditional. x2 is conditional.
    pgm_dag = {
        'x1': [],
        'x2': ['x1'],
        'x3': []
    }
    compiler = PGMCompiler(pgm_dag)
    
    targets = {'x1', 'x2', 'x3'}
    conditions = set()
    
    structure = compiler.compile_inference(targets, conditions)
    
    # In inference mode, we traverse backwards, so leaves come before roots.
    # x2 is a leaf, x1 is a root. x3 is also a root.
    idx_x1 = structure.block_order.index('x1')
    idx_x2 = structure.block_order.index('x2')
    idx_x3 = structure.block_order.index('x3')
    
    # x2 should appear after its ancestors in topological order,
    # so in REVERSE topological order it should appear BEFORE them.
    assert idx_x2 < idx_x1
    
    # x1 and x3 are both roots. In topo sort they are sorted alphabetically.
    # x1, x3. In reverse topo, they should be x3, x1.
    assert idx_x3 < idx_x1
    
    assert structure.dependencies['x2'] == []
    assert structure.dependencies['x3'] == [] # independent
    assert structure.dependencies['x1'] == ['x2']

def test_chain_structure():
    """
    Standard chain: x1 -> x2 -> x3
    """
    pgm_dag = {
        'x1': [],
        'x2': ['x1'],
        'x3': ['x2']
    }
    compiler = PGMCompiler(pgm_dag)
    
    # Joint
    structure = compiler.compile_inference({'x1', 'x2', 'x3'}, set())
    
    # Order should be x3, x2, x1 (reverse topological)
    assert structure.block_order == ['x3', 'x2', 'x1']
    assert structure.dependencies['x3'] == []
    assert structure.dependencies['x2'] == ['x3']
    assert structure.dependencies['x1'] == ['x2'] # Markov property preserved

    # Posterior
    structure = compiler.compile_inference(targets={'x1', 'x2'}, conditions={'x3'})
    
    # Order should be x3, x2, x1
    assert structure.block_order == ['x3', 'x2', 'x1']
    # now the dependencies are reversed (we go from conditions to targets)
    assert structure.dependencies['x3'] == []
    assert structure.dependencies['x2'] == ['x3']
    assert structure.dependencies['x1'] == ['x2']

def test_fork_structure():
    """
    Fork: x1 <- z -> x2
    """
    pgm_dag = {
        'z': [],
        'x1': ['z'],
        'x2': ['z']
    }
    compiler = PGMCompiler(pgm_dag)
    
    # Joint
    structure = compiler.compile_inference({'x1', 'x2', 'z'}, set())
    
    # x1 and x2 are leaves, z is root. Reverse topo: [x2, x1, z] or [x1, x2, z]
    # alphabetical tie-break in topo: z, x1, x2. Reverse: x2, x1, z.
    assert structure.block_order == ['x2', 'x1', 'z']
    
    # x2 is first
    assert structure.dependencies['x2'] == []
    # x1 depends on x2 (moral edge through z)
    assert structure.dependencies['x1'] == ['x2']
    # z depends on both
    assert set(structure.dependencies['z']) == {'x1', 'x2'}

    # Posterior
    structure = compiler.compile_inference(targets={'z'}, conditions={'x1', 'x2'})
    
    # Order should be x2, x1, z (reverse topological)
    assert structure.block_order == ['x2', 'x1', 'z']
    assert structure.dependencies['x1'] == ['x2']
    assert structure.dependencies['x2'] == []
    assert structure.dependencies['z'] == ['x2', 'x1']

def test_collider_structure():
    """
    Collider: x1 -> y <- x2
    """
    pgm_dag = {
        'x1': [],
        'x2': [],
        'y': ['x1', 'x2']
    }
    compiler = PGMCompiler(pgm_dag)
    
    structure = compiler.compile_inference({'x1'}, {'y', 'x2'})
    
    # y is child, x1, x2 are parents. Reverse topo: y, x2, x1
    assert structure.block_order == ['y', 'x2', 'x1']
    
    # y is first
    assert structure.dependencies['y'] == []
    # x2 depends on y
    assert structure.dependencies['x2'] == ['y']
    # x1 depends on y and x2 (moral edge)
    assert set(structure.dependencies['x1']) == {'y', 'x2'}

def test_collider_posterior():
    """
    Collider: x1 -> y <- x2.
    Condition on y. Target x1, x2.
    Expect x1 and x2 to be dependent given y (explaining away).
    """
    pgm_dag = {
        'x1': [],
        'x2': [],
        'y': ['x1', 'x2']
    }
    compiler = PGMCompiler(pgm_dag)
    
    structure = compiler.compile_inference(targets={'x1', 'x2'}, conditions={'y'})
    
    # Order should be y, then x1, x2 (or x2, x1)
    assert structure.block_order[0] == 'y'
    
    # Check dependencies
    # Second block (say x1) depends on y.
    # Third block (say x2) depends on y AND x1.
    
    second = structure.block_order[1]
    third = structure.block_order[2]
    
    assert structure.dependencies[second] == ['y']
    assert set(structure.dependencies[third]) == {'y', second}

def test_disconnected_components():
    """
    Two disconnected chains: x1->x2 and y1->y2
    """
    pgm_dag = {
        'x1': [], 'x2': ['x1'],
        'y1': [], 'y2': ['y1']
    }
    compiler = PGMCompiler(pgm_dag)
    
    structure = compiler.compile_inference({'x1', 'x2', 'y1', 'y2'}, set())
    
    # x1->x2, y1->y2. Reverse topo: x2, x1, y2, y1 (mixed by alphabetical tie-break)
    # Topo order: x1, y1, x2, y2. Reverse: y2, x2, y1, x1.
    assert structure.block_order == ['y2', 'x2', 'y1', 'x1']
    
    # Check dependencies
    assert structure.dependencies['y2'] == []
    assert structure.dependencies['x2'] == []
    assert structure.dependencies['y1'] == ['y2']
    assert structure.dependencies['x1'] == ['x2']
    
    # Check that x variables don't depend on y variables and vice versa
    assert 'y1' not in structure.dependencies['x1']
    assert 'y2' not in structure.dependencies['x1']

def test_conditions_ordering():
    """
    Test that conditions are placed first in the block order.
    """
    # x1 -> x2
    pgm_dag = {'x1': [], 'x2': ['x1']}
    compiler = PGMCompiler(pgm_dag)
    
    # Condition on x2 (anti-causal / diagnostic reasoning)
    # We want p(x1 | x2).
    # Flow order should be x2, x1.
    # Note: This forces the flow to learn the inverse map / posterior.
    
    structure = compiler.compile_inference(targets={'x1'}, conditions={'x2'})
    
    assert structure.block_order[0] == 'x2'
    assert structure.block_order[1] == 'x1'
    
    # Dependency: x1 depends on x2 in the flow (because x2 comes first)
    # Even though in PGM x1 -> x2.
    # The compiler uses d-separation on the graph.
    # If we condition on x2, and x2 is first.
    # Is x1 d-separated from x2 given {}? No, they are adjacent.
    assert structure.dependencies['x1'] == ['x2']

def test_complex_gmm_inference_structure():
    """
    Tests the structure from the user's report (8-node GMM).
    """
    pgm_dag = {
        'x0': [],
        'x1': ['x0'],
        'x2': ['x0'],
        'x3': ['x0', 'x2'],
        'x4': ['x0', 'x2'],
        'x5': ['x0', 'x1', 'x3'],
        'x6': ['x0', 'x1', 'x3', 'x4'],
        'x7': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    }
    targets = {'x0'}
    conditions = {'x7', 'x3', 'x2', 'x5', 'x4', 'x6', 'x1'}
    
    compiler = PGMCompiler(pgm_dag)
    structure = compiler.compile_inference(targets=targets, conditions=conditions)
    
    # Check ordering: Should be reverse topological. x7 is the deepest leaf.
    assert structure.block_order == ['x7', 'x6', 'x5', 'x4', 'x3', 'x2', 'x1', 'x0']
    
    # Check that conditions have dependencies (dense structure)
    assert structure.dependencies['x7'] == []
    assert structure.dependencies['x6'] == ['x7']
    assert structure.dependencies['x5'] == ['x7', 'x6']
    assert structure.dependencies['x0'] == ['x7', 'x6', 'x5', 'x4', 'x3', 'x2', 'x1']

