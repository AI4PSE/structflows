"""
Tests for StructuralPGM and specific PGM implementations.

Tests organized by:
- StructuralPGM core functionality
- Different mechanism types
- Multi-dimensional nodes
- Error handling
"""

import jax
import jax.numpy as jnp
from jax import random
import pytest

from structflow.pgms.pgm import (
    make_chain_pgm,
    make_fork_pgm,
    make_collider_pgm,
    make_random_pgm,
    make_dense_pgm,
    make_chain_structure,
    make_structural_pgm,
    StructuralPGM,
    LinearMechanism,
)


# =============================================================================
# StructuralPGM Core Functionality Tests
# =============================================================================

class TestStructuralPGMCore:
    """Tests for StructuralPGM core methods."""
    
    def test_get_all_descendants(self):
        """Test get_all_descendants via structure for chain."""
        key = random.PRNGKey(0)
        pgm = make_chain_pgm(3, "linear", key)
        
        assert pgm.structure.get_all_descendants("x0") == {"x1", "x2"}
        assert pgm.structure.get_all_descendants("x1") == {"x2"}
        assert pgm.structure.get_all_descendants("x2") == set()
    
    def test_is_ancestral_query_chain(self):
        """Test is_ancestral_query via structure for chain."""
        key = random.PRNGKey(1)
        pgm = make_chain_pgm(4, "linear", key)
        
        # Ancestral queries: conditions don't depend on targets
        # x3 not dependent on x0, so x0 can be condition for x3
        assert pgm.structure.is_ancestral_query({"x3"}, {"x0"})
        
        # Sampling x0, x3 from scratch is always ancestral
        assert pgm.structure.is_ancestral_query({"x0", "x3"}, set())
        
        # Non-ancestral: x0 is ancestor of x3, so x3 depends on x0
        assert not pgm.structure.is_ancestral_query({"x0"}, {"x3"})
    
    def test_is_ancestral_query_fork(self):
        """Test is_ancestral_query via structure for fork."""
        key = random.PRNGKey(2)
        pgm = make_fork_pgm(2, "linear", key)
        
        # Sample x1, x2 given z is ancestral
        assert pgm.structure.is_ancestral_query({"x1", "x2"}, {"z"})
        
        # Sample z given x1 is NOT ancestral (x1 depends on z)
        assert not pgm.structure.is_ancestral_query({"z"}, {"x1"})
    
    def test_get_pgm_structure(self):
        """Test DAG structure retrieval."""
        key = random.PRNGKey(3)
        pgm = make_chain_pgm(3, "linear", key)
        
        structure = pgm.get_pgm_structure()
        
        assert structure["x0"] == []
        assert structure["x1"] == ["x0"]
        assert structure["x2"] == ["x1"]
    
    def test_get_inference_targets_and_conditions(self):
        """Test default inference task computation."""
        key = random.PRNGKey(4)
        pgm = make_chain_pgm(4, "linear", key)
        
        targets, conditions = pgm.get_inference_targets_and_conditions()
        
        # For chain: root is x0, leaf is x3
        assert "x0" in targets
        assert "x3" in conditions


# =============================================================================
# PGM Sample/LogProb/Counterfactual Tests
# =============================================================================

class TestPGMOperations:
    """Test core PGM operations: sample, log_prob, counterfactual."""
    
    @pytest.mark.parametrize("mechanism_type", ["linear", "nonlinear_parent", "nonlinear_noise", "universal"])
    def test_sample_basic(self, mechanism_type):
        """Test basic sampling from joint distribution."""
        key = random.PRNGKey(10)
        pgm = make_chain_pgm(3, mechanism_type, key)
        
        samples = pgm.sample(random.PRNGKey(100))
        
        assert set(samples.keys()) == {"x0", "x1", "x2"}
        for val in samples.values():
            assert val.shape == (1,)
        assert all(jnp.isfinite(v).all() for v in samples.values())
    
    @pytest.mark.parametrize("mechanism_type", ["linear", "nonlinear_parent", "nonlinear_noise", "universal"])
    def test_log_prob_joint(self, mechanism_type):
        """Test joint log_prob computation."""
        key = random.PRNGKey(11)
        pgm = make_chain_pgm(3, mechanism_type, key)
        
        samples = pgm.sample(random.PRNGKey(101))
        lp = pgm.log_prob(samples)
        
        assert jnp.isfinite(lp)
    
    @pytest.mark.parametrize("mechanism_type", ["linear", "nonlinear_parent", "nonlinear_noise", "universal"])
    def test_counterfactual_basic(self, mechanism_type):
        """Test counterfactual computation."""
        key = random.PRNGKey(12)
        pgm = make_chain_pgm(3, mechanism_type, key)
        
        data = pgm.sample(random.PRNGKey(102))
        interventions = {"x1": jnp.array([5.0])}
        cf, log_prob = pgm.counterfactual(data, interventions)
        
        assert jnp.allclose(cf["x0"], data["x0"], atol=1e-4)
        assert jnp.allclose(cf["x1"], interventions["x1"], atol=1e-4)
        assert jnp.isfinite(log_prob)
    
    def test_log_prob_inference_query_raises(self):
        """Test that log_prob raises for non-ancestral queries."""
        key = random.PRNGKey(13)
        pgm = make_chain_pgm(3, "linear", key)
        
        data = pgm.sample(random.PRNGKey(103))
        
        with pytest.raises(NotImplementedError):
            pgm.log_prob(data, targets={"x0"}, conditions={"x2"})


# =============================================================================
# Multi-Dimensional Node Tests
# =============================================================================

class TestMultiDimensionalNodes:
    """Test that PGMs correctly handle multi-dimensional nodes."""
    
    @pytest.mark.parametrize("node_dim,mechanism", [
        (2, "linear"),
        (3, "nonlinear_parent"),
        (2, "universal"),
    ])
    def test_multidim_chain_sample(self, node_dim, mechanism):
        """Test sampling from chain with multi-dimensional nodes."""
        key = random.PRNGKey(40)
        pgm = make_chain_pgm(3, mechanism, key, node_dim=node_dim)
        
        samples = pgm.sample(random.PRNGKey(400))
        
        assert samples["x0"].shape == (node_dim,)
        assert samples["x1"].shape == (node_dim,)
        assert samples["x2"].shape == (node_dim,)
    
    @pytest.mark.parametrize("node_dim", [2, 3, 4])
    def test_multidim_log_prob(self, node_dim):
        """Test log_prob with multi-dimensional nodes."""
        key = random.PRNGKey(41)
        pgm = make_chain_pgm(3, "linear", key, node_dim=node_dim)
        
        samples = pgm.sample(random.PRNGKey(401))
        lp = pgm.log_prob(samples)
        
        assert jnp.isfinite(lp)
    
    def test_multidim_counterfactual(self):
        """Test counterfactual with multi-dimensional nodes."""
        key = random.PRNGKey(42)
        pgm = make_chain_pgm(3, "linear", key, node_dim=2)
        
        data = pgm.sample(random.PRNGKey(402))
        interventions = {"x1": jnp.array([1.0, -1.0])}
        
        cf, log_prob = pgm.counterfactual(data, interventions)
        
        assert jnp.allclose(cf["x1"], interventions["x1"], atol=1e-4)
        assert jnp.isfinite(log_prob)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test that PGMs raise appropriate errors for invalid inputs."""
    
    def test_log_prob_missing_data(self):
        """Test error when data is missing required variables."""
        key = random.PRNGKey(51)
        pgm = make_chain_pgm(3, "linear", key)
        
        incomplete_data = {"x0": jnp.array([1.0]), "x1": jnp.array([2.0])}
        
        with pytest.raises(KeyError):
            pgm.log_prob(incomplete_data)
    
    def test_counterfactual_missing_data(self):
        """Test error when counterfactual data is missing variables."""
        key = random.PRNGKey(52)
        pgm = make_chain_pgm(3, "linear", key)
        
        incomplete_data = {"x0": jnp.array([1.0])}
        
        with pytest.raises(KeyError):
            pgm.counterfactual(incomplete_data, interventions={})


# =============================================================================
# Different Structure Tests
# =============================================================================

class TestDifferentStructures:
    """Test PGMs with different DAG structures."""
    
    def test_fork_structure(self):
        """Test fork structure z -> x1, z -> x2."""
        key = random.PRNGKey(60)
        pgm = make_fork_pgm(2, "linear", key)
        
        samples = pgm.sample(random.PRNGKey(600))
        
        assert "z" in samples
        assert "x1" in samples
        assert "x2" in samples
        
        lp = pgm.log_prob(samples)
        assert jnp.isfinite(lp)
    
    def test_collider_structure(self):
        """Test collider structure x1 -> y, x2 -> y."""
        key = random.PRNGKey(61)
        pgm = make_collider_pgm(2, "linear", key)
        
        samples = pgm.sample(random.PRNGKey(601))
        
        assert "x1" in samples
        assert "x2" in samples
        assert "y" in samples
        
        lp = pgm.log_prob(samples)
        assert jnp.isfinite(lp)
    
    def test_random_sparse_pgm(self):
        """Test random sparse PGM."""
        key = random.PRNGKey(62)
        pgm = make_random_pgm(5, "linear", key, sparsity=0.7)
        
        samples = pgm.sample(random.PRNGKey(602))
        lp = pgm.log_prob(samples)
        
        assert len(samples) == 5
        assert jnp.isfinite(lp)
    
    def test_dense_pgm(self):
        """Test fully connected (dense) PGM."""
        key = random.PRNGKey(63)
        pgm = make_dense_pgm(4, "nonlinear_parent", key)
        
        samples = pgm.sample(random.PRNGKey(603))
        lp = pgm.log_prob(samples)
        
        assert len(samples) == 4
        assert jnp.isfinite(lp)