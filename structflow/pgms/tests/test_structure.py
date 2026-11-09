import pytest
import jax
import jax.numpy as jnp
from structflow.pgms.structure import BlockTriangularStructure


class TestBlockTriangularStructure:
    """Test input validation and basic functionality of BlockTriangularStructure."""
    
    def test_valid_creation(self):
        """Test that valid inputs create a structure successfully."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        assert structure.block_order == ['x', 'y', 'z']
        assert structure.dims == {'x': 1, 'y': 2, 'z': 1}
    
    def test_empty_block_order(self):
        """Test that empty block_order raises ValueError."""
        with pytest.raises(ValueError, match="block_order cannot be empty"):
            BlockTriangularStructure([], {}, {})
    
    def test_duplicate_blocks(self):
        """Test that duplicate block names raise ValueError."""
        with pytest.raises(ValueError, match="duplicate names"):
            BlockTriangularStructure(['x', 'x'], {}, {})
    
    def test_invalid_dependency_key(self):
        """Test that dependency keys not in block_order raise ValueError."""
        with pytest.raises(ValueError, match="not in block_order"):
            BlockTriangularStructure(
                block_order=['x', 'y'],
                dependencies={'z': []},  # z not in block_order
                dims={'x': 1, 'y': 2}
            )
    
    def test_invalid_dependency_value(self):
        """Test that dependency values not in block_order raise ValueError."""
        with pytest.raises(ValueError, match="not in block_order"):
            BlockTriangularStructure(
                block_order=['x', 'y'],
                dependencies={'y': ['z']},  # z not in block_order
                dims={'x': 1, 'y': 2}
            )
    
    def test_invalid_topological_order(self):
        """Test that non-topological ordering raises ValueError."""
        with pytest.raises(ValueError, match="Invalid topological order"):
            BlockTriangularStructure(
                block_order=['y', 'x'],  # y before x, but y depends on x
                dependencies={'x': [], 'y': ['x']},
                dims={'x': 1, 'y': 2}
            )
    
    def test_circular_dependency(self):
        """Test that circular dependencies are detected (via topological order check)."""
        with pytest.raises(ValueError, match="Invalid topological order"):
            BlockTriangularStructure(
                block_order=['x', 'y'],
                dependencies={'x': ['y'], 'y': ['x']},
                dims={'x': 1, 'y': 2}
            )
    
    def test_missing_dims(self):
        """Test that missing dims raise ValueError."""
        with pytest.raises(ValueError, match="missing dims"):
            BlockTriangularStructure(
                block_order=['x', 'y'],
                dependencies={'x': [], 'y': []},
                dims={'x': 1}  # Missing y
            )
    
    def test_negative_dim(self):
        """Test that non-positive dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            BlockTriangularStructure(
                block_order=['x'],
                dependencies={'x': []},
                dims={'x': 0}
            )
    
    def test_tractable_query_validation(self):
        """Test validation in is_tractable_query."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        # Invalid target
        with pytest.raises(ValueError, match="Unknown targets"):
            structure.is_tractable_query({'invalid'}, set())
        
        # Invalid condition
        with pytest.raises(ValueError, match="Unknown conditions"):
            structure.is_tractable_query(set(), {'invalid'})
        
        # Overlapping targets and conditions
        with pytest.raises(ValueError, match="must be disjoint"):
            structure.is_tractable_query({'x', 'y'}, {'y', 'z'})
    
    def test_is_ancestral_query(self):
        """Test is_ancestral_query (Descendant Rule only)."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        # p(z | x): ancestral (x is upstream of z) — even though y is a gap
        assert structure.is_ancestral_query({'z'}, {'x'}) is True
        
        # p(x | z): NOT ancestral (z is downstream of x)
        assert structure.is_ancestral_query({'x'}, {'z'}) is False
        
        # p(z | y): ancestral
        assert structure.is_ancestral_query({'z'}, {'y'}) is True
        
        # p(x, y, z): unconditional always ancestral
        assert structure.is_ancestral_query({'x', 'y', 'z'}, set()) is True
        
        # p(y | x): ancestral
        assert structure.is_ancestral_query({'y'}, {'x'}) is True
        
        # Validation works the same
        with pytest.raises(ValueError, match="Unknown targets"):
            structure.is_ancestral_query({'invalid'}, set())
    
    def test_query_requires_marginalisation(self):
        """Test query_requires_marginalisation (Parent Rule only)."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        # p(z | x): requires marginalisation (y is parent of z, missing)
        assert structure.query_requires_marginalisation({'z'}, {'x'}) is True
        
        # p(z | y): no marginalisation (y is parent of z, present)
        assert structure.query_requires_marginalisation({'z'}, {'y'}) is False
        
        # p(y, z | x): no marginalisation (y's parent x is in conditions, z's parent y is in targets)
        assert structure.query_requires_marginalisation({'y', 'z'}, {'x'}) is False
        
        # p(x): no marginalisation (root node, no parents)
        assert structure.query_requires_marginalisation({'x'}, set()) is False
        
        # p(x, y, z): no marginalisation (joint includes all parents)
        assert structure.query_requires_marginalisation({'x', 'y', 'z'}, set()) is False
        
        # p(z): requires marginalisation (y is parent of z, missing)
        assert structure.query_requires_marginalisation({'z'}, set()) is True
    
    def test_is_tractable_query_composition(self):
        """Test that is_tractable_query = is_ancestral AND NOT requires_marginalisation."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        # p(z | x): ancestral=True, marginalisation=True → tractable=False
        assert structure.is_ancestral_query({'z'}, {'x'}) is True
        assert structure.query_requires_marginalisation({'z'}, {'x'}) is True
        assert structure.is_tractable_query({'z'}, {'x'}) is False
        
        # p(z | y): ancestral=True, marginalisation=False → tractable=True
        assert structure.is_ancestral_query({'z'}, {'y'}) is True
        assert structure.query_requires_marginalisation({'z'}, {'y'}) is False
        assert structure.is_tractable_query({'z'}, {'y'}) is True
        
        # p(x | z): ancestral=False → tractable=False (regardless of marginalisation)
        assert structure.is_ancestral_query({'x'}, {'z'}) is False
        assert structure.is_tractable_query({'x'}, {'z'}) is False
    
    def test_get_all_descendants(self):
        """Test get_all_descendants method."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        assert structure.get_all_descendants('x') == {'y', 'z'}
        assert structure.get_all_descendants('y') == {'z'}
        assert structure.get_all_descendants('z') == set()
        
        # Test validation
        with pytest.raises(ValueError, match="not in structure"):
            structure.get_all_descendants('invalid')
    
    def test_get_all_ancestors(self):
        """Test get_all_ancestors method."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        assert structure.get_all_ancestors('z') == {'x', 'y'}
        assert structure.get_all_ancestors('y') == {'x'}
        assert structure.get_all_ancestors('x') == set()
        
        # Test with exclusion
        assert structure.get_all_ancestors('z', exclude={'y'}) == set()
        
        # Test validation
        with pytest.raises(ValueError, match="not in structure"):
            structure.get_all_ancestors('invalid')
    
    def test_get_required_simulation_blocks(self):
        """Test get_required_simulation_blocks method."""
        structure = BlockTriangularStructure(
            block_order=['x', 'y', 'z'],
            dependencies={'x': [], 'y': ['x'], 'z': ['y']},
            dims={'x': 1, 'y': 2, 'z': 1}
        )
        
        # Sample z with no conditions - need to simulate x, y, z
        assert structure.get_required_simulation_blocks({'z'}, set()) == ['x', 'y', 'z']
        
        # Sample z given y - only need to simulate z
        assert structure.get_required_simulation_blocks({'z'}, {'y'}) == ['z']
        
        # Sample y and z given x - need y, z
        assert structure.get_required_simulation_blocks({'y', 'z'}, {'x'}) == ['y', 'z']