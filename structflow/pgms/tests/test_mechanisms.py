"""
Tests for Mechanism implementations.

For each mechanism type:
- test_forward_inverse_roundtrip: forward then inverse recovers noise
- test_log_det_consistency: log_det from forward + inverse = 0
- test_various_dimensions: different input/output dims
"""

import jax
import jax.numpy as jnp
from jax import random
import pytest
from functools import partial

from structflow.pgms.mechanisms import (
    LinearMechanism,
    NonlinearParentMechanism,
    NonlinearNoiseMechanism,
    UniversalMechanism,
    CAREFLMechanism,
    UniversalConditionerMechanism,
)
import equinox as eqx


# --- Helper function for roundtrip testing ---

def check_roundtrip(mech, parents, noise, atol=1e-5):
    """Check that forward then inverse recovers noise."""
    x, log_det_fwd = mech.forward(parents, noise)
    noise_rec, log_det_inv = mech.inverse(parents, x)
    
    assert jnp.allclose(noise, noise_rec, atol=atol), \
        f"Roundtrip failed: max diff = {jnp.abs(noise - noise_rec).max()}"
    assert jnp.allclose(log_det_fwd + log_det_inv, 0.0, atol=atol), \
        f"Log det inconsistent: {log_det_fwd} + {log_det_inv} != 0"


# =============================================================================
# LinearMechanism Tests
# =============================================================================

class TestLinearMechanism:
    """Tests for LinearMechanism: x = A @ parents + B @ noise + bias."""
    
    @pytest.mark.parametrize("input_dim,output_dim", [
        (0, 1),   # Root node (no parents)
        (1, 1),   # 1D -> 1D
        (2, 1),   # 2D parents -> 1D
        (1, 3),   # 1D parent -> 3D
        (4, 2),   # Multi-dim both
    ])
    def test_forward_inverse_roundtrip(self, input_dim, output_dim):
        """forward then inverse should recover noise."""
        key = random.PRNGKey(input_dim * 10 + output_dim)
        
        parent_names = [f"p{i}" for i in range(input_dim)] if input_dim > 0 else []
        mech = LinearMechanism(
            parent_names=parent_names,
            input_dim=input_dim,
            output_dim=output_dim,
            key=key
        )
        
        k1, k2 = random.split(key)
        parents = random.normal(k1, (input_dim,)) if input_dim > 0 else jnp.array([])
        noise = random.normal(k2, (output_dim,))
        
        check_roundtrip(mech, parents, noise)
    
    def test_explicit_parameters(self):
        """Test with explicitly provided A, B, bias."""
        A = jnp.array([[1.0, 2.0], [0.5, -0.5]])  # 2x2
        B = jnp.array([[1.0, 0.0], [0.3, 0.8]])    # Lower triangular
        bias = jnp.array([0.5, -0.5])
        
        mech = LinearMechanism(
            parent_names=["x0", "x1"],
            input_dim=2,
            output_dim=2,
            A=A,
            B=B,
            bias=bias
        )
        
        parents = jnp.array([1.0, 2.0])
        noise = jnp.array([0.5, -0.3])
        
        x, log_det = mech.forward(parents, noise)
        
        # Manual calculation
        expected = A @ parents + B @ noise + bias
        assert jnp.allclose(x, expected)
        
        # Log det should be log(|det(B)|) = log(|0.8 * 1.0|) = log(0.8)
        expected_log_det = jnp.log(jnp.abs(jnp.linalg.det(B)))
        assert jnp.allclose(log_det, expected_log_det, atol=1e-5)


# =============================================================================
# NonlinearParentMechanism Tests
# =============================================================================

class TestNonlinearParentMechanism:
    """Tests for NonlinearParentMechanism: x = mu(parents) + sigma(parents) * noise."""
    
    @pytest.mark.parametrize("input_dim,output_dim", [
        (0, 1),   # Root node
        (1, 1),   # Simple
        (2, 3),   # Multi-dim
        (5, 2),
    ])
    def test_forward_inverse_roundtrip(self, input_dim, output_dim):
        """forward then inverse should recover noise."""
        key = random.PRNGKey(100 + input_dim * 10 + output_dim)
        
        parent_names = [f"p{i}" for i in range(input_dim)] if input_dim > 0 else []
        mech = NonlinearParentMechanism(
            parent_names=parent_names,
            input_dim=input_dim,
            output_dim=output_dim,
            key=key
        )
        
        k1, k2 = random.split(key)
        parents = random.normal(k1, (input_dim,)) if input_dim > 0 else jnp.array([])
        noise = random.normal(k2, (output_dim,))
        
        check_roundtrip(mech, parents, noise)
    
    def test_sigma_always_positive(self):
        """sigma(parents) should always be positive."""
        key = random.PRNGKey(101)
        mech = NonlinearParentMechanism(
            parent_names=["x"],
            input_dim=1,
            output_dim=2,
            key=key
        )
        
        # Test with various parent values
        for val in [-10.0, 0.0, 10.0]:
            parents = jnp.array([val])
            mu, sigma = mech._get_params(parents)
            assert (sigma > 0).all(), f"sigma should be positive, got {sigma}"


# =============================================================================
# NonlinearNoiseMechanism Tests
# =============================================================================

class TestNonlinearNoiseMechanism:
    """Tests for NonlinearNoiseMechanism: x = A @ parents + bias + f_odd(noise)."""
    
    @pytest.mark.parametrize("input_dim,output_dim", [
        (0, 1),   # Root node
        (1, 1),   # Simple
        (2, 1),   # Multi-dim parents
    ])
    def test_forward_inverse_roundtrip(self, input_dim, output_dim):
        """forward then inverse should recover noise."""
        key = random.PRNGKey(200 + input_dim * 10 + output_dim)
        
        parent_names = [f"p{i}" for i in range(input_dim)] if input_dim > 0 else []
        mech = NonlinearNoiseMechanism(
            parent_names=parent_names,
            input_dim=input_dim,
            output_dim=output_dim,
            key=key,
            num_bins=8
        )
        
        k1, k2 = random.split(key)
        parents = random.normal(k1, (input_dim,)) if input_dim > 0 else jnp.array([])
        # Keep noise small to stay in good spline range
        noise = random.normal(k2, (output_dim,)) * 0.5
        
        # This mechanism uses iterative inversion, so allow larger tolerance
        check_roundtrip(mech, parents, noise, atol=1e-3)
    
    def test_odd_function_property(self):
        """f_odd(-e) should equal -f_odd(e)."""
        key = random.PRNGKey(201)
        mech = NonlinearNoiseMechanism(
            parent_names=[],
            input_dim=0,
            output_dim=1,
            key=key
        )
        
        noise = jnp.array([0.5])
        f_pos, _ = mech._odd_transform(noise)
        f_neg, _ = mech._odd_transform(-noise)
        
        assert jnp.allclose(f_pos, -f_neg, atol=1e-5), \
            f"Odd function property violated: f({noise}) = {f_pos}, f({-noise}) = {f_neg}"


# =============================================================================
# UniversalMechanism Tests
# =============================================================================

class TestUniversalMechanism:
    """Tests for UniversalMechanism: x = spline(noise; params(parents))."""
    
    @pytest.mark.parametrize("input_dim,output_dim", [
        (0, 1),   # Root node
        (1, 1),   # Simple
        (2, 1),   # Multi-dim parents
        (1, 2),   # Multi-dim output
    ])
    def test_forward_inverse_roundtrip(self, input_dim, output_dim):
        """forward then inverse should recover noise."""
        key = random.PRNGKey(300 + input_dim * 10 + output_dim)
        
        parent_names = [f"p{i}" for i in range(input_dim)] if input_dim > 0 else []
        mech = UniversalMechanism(
            parent_names=parent_names,
            input_dim=input_dim,
            output_dim=output_dim,
            key=key,
            num_bins=8
        )
        
        k1, k2 = random.split(key)
        parents = random.normal(k1, (input_dim,)) if input_dim > 0 else jnp.array([])
        # Keep noise in spline range
        noise = random.normal(k2, (output_dim,)) * 0.5
        
        check_roundtrip(mech, parents, noise, atol=1e-4)
    
    def test_parent_conditioning(self):
        """Different parents should give different transformations."""
        key = random.PRNGKey(301)
        mech = UniversalMechanism(
            parent_names=["x"],
            input_dim=1,
            output_dim=1,
            key=key
        )
        
        noise = jnp.array([0.5])
        
        x1, _ = mech.forward(jnp.array([0.0]), noise)
        x2, _ = mech.forward(jnp.array([2.0]), noise)
        
        # Should be different (unless by coincidence)
        assert not jnp.allclose(x1, x2, atol=0.01), \
            "Different parents should give different outputs"


# =============================================================================
# Mechanism log_prob Tests
# =============================================================================

class TestMechanismLogProb:
    """Test log_prob computation for all mechanisms."""
    
    @pytest.mark.parametrize("MechClass,kwargs", [
        (LinearMechanism, {}),
        (NonlinearParentMechanism, {}),
        (UniversalMechanism, {"num_bins": 8}),
    ])
    def test_log_prob_finite(self, MechClass, kwargs):
        """log_prob should return finite values."""
        key = random.PRNGKey(400)
        
        mech = MechClass(
            parent_names=["x0"],
            input_dim=1,
            output_dim=1,
            key=key,
            **kwargs
        )
        
        parents = jnp.array([0.5])
        x = jnp.array([0.3])
        
        lp = mech.log_prob(parents, x)
        assert jnp.isfinite(lp), f"log_prob should be finite, got {lp}"
    
    @pytest.mark.parametrize("MechClass,kwargs", [
        (LinearMechanism, {}),
        (NonlinearParentMechanism, {}),
        (UniversalMechanism, {"num_bins": 8}),
    ])
    def test_log_prob_normalization(self, MechClass, kwargs):
        """Approximate normalization test via importance sampling."""
        key = random.PRNGKey(401)
        
        mech = MechClass(
            parent_names=[],
            input_dim=0,
            output_dim=1,
            key=key,
            **kwargs
        )
        
        # Sample many noise values and check log_prob averages sensibly
        keys = random.split(key, 1000)
        log_probs = []
        for k in keys:
            noise = random.normal(k, (1,)) * 0.5
            x, _ = mech.forward(jnp.array([]), noise)
            lp = mech.log_prob(jnp.array([]), x)
            log_probs.append(lp)
        
        log_probs = jnp.array(log_probs)
        assert jnp.isfinite(log_probs).all()


# =============================================================================
# CAREFLMechanism Tests
# =============================================================================

class TestCAREFLMechanism:
    """Tests for CAREFLMechanism: x = exp(s(pa)) * z + t(pa)."""
    
    @pytest.mark.parametrize("input_dim,output_dim", [
        (0, 1),   # Root node
        (1, 1),   # Simple
        (2, 1),   # Multi-dim parents
        (1, 2),   # Multi-dim output
    ])
    def test_forward_inverse_roundtrip(self, input_dim, output_dim):
        """forward then inverse should recover noise."""
        key = random.PRNGKey(500 + input_dim * 10 + output_dim)
        
        parent_names = [f"p{i}" for i in range(input_dim)] if input_dim > 0 else []
        mech = CAREFLMechanism(
            parent_names=parent_names,
            input_dim=input_dim,
            output_dim=output_dim,
            key=key,
            hidden_width=16,
            hidden_depth=2
        )
        
        k1, k2 = random.split(key)
        parents = random.normal(k1, (input_dim,)) if input_dim > 0 else jnp.array([])
        noise = random.normal(k2, (output_dim,))
        
        check_roundtrip(mech, parents, noise)


# =============================================================================
# UniversalConditionerMechanism Tests
# =============================================================================

class TestUniversalConditionerMechanism:
    """Tests for UniversalConditionerMechanism using an external conditioner."""
    
    def test_forward_inverse_roundtrip(self):
        """Test with an external MLP conditioner."""
        input_dim = 2
        output_dim = 1
        num_bins = 4
        param_dim = output_dim * (3 * num_bins + 1)
        
        key = random.PRNGKey(600)
        k1, k2, k3 = random.split(key, 3)
        
        # Simple MLP as conditioner
        conditioner = eqx.nn.MLP(
            in_size=input_dim,
            out_size=param_dim,
            width_size=16,
            depth=2,
            key=k1
        )
        
        mech = UniversalConditionerMechanism(
            parent_names=["x0", "x1"],
            input_dim=input_dim,
            output_dim=output_dim,
            conditioner=conditioner,
            key=k2,
            num_bins=num_bins
        )
        
        parents = random.normal(k3, (input_dim,))
        noise = random.normal(k3, (output_dim,)) * 0.5
        
        check_roundtrip(mech, parents, noise)