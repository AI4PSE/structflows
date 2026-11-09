"""Tests for ConvBlock and mixed conv/MLP structured velocity fields."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

from structflow.conditional_flows.conv import ConvBlock
from structflow.conditional_flows.static.net import StructuredVelocityFn, ConditionalBlock
from structflow.conditional_flows.flow_factory import create_static_flow
from structflow.pgms.structure import BlockTriangularStructure

H, W, C = 8, 8, 3
IMG_DIM = H * W * C


def make_structure():
    # colored-mnist-like: digit -> color, (digit, color) -> image
    return BlockTriangularStructure(
        block_order=['digit', 'color', 'image'],
        dependencies={'digit': [], 'color': ['digit'], 'image': ['digit', 'color']},
        dims={'digit': 10, 'color': 3, 'image': IMG_DIM},
    )


def test_conv_block_forward_shape():
    key = jax.random.PRNGKey(0)
    block = ConvBlock(IMG_DIM, [10, 3], key, shape=(H, W, C), base_channels=8)
    t = jnp.array([0.3])
    x = jax.random.normal(key, (IMG_DIM,))
    out = block(t, x, jnp.zeros(10), jnp.ones(3))
    assert out.shape == (IMG_DIM,)
    assert jnp.all(jnp.isfinite(out))


def test_conv_block_rejects_bad_shape():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        ConvBlock(IMG_DIM, [], key, shape=(H, W, C + 1), base_channels=8)
    with pytest.raises(ValueError):
        ConvBlock(6 * 6 * 3, [], key, shape=(6, 6, 3), base_channels=8)  # not /4


def test_mixed_structured_velocity_fn():
    key = jax.random.PRNGKey(1)
    structure = make_structure()

    def block_constructor(input_dim, context_dims, key, name=None):
        if name == 'image':
            return ConvBlock(input_dim, context_dims, key, shape=(H, W, C), base_channels=8)
        return ConditionalBlock(input_dim, context_dims, key, hidden_dim=16, depth=3)

    net = StructuredVelocityFn(structure, key, block_constructor)
    assert type(net.blocks['image']).__name__ == 'ConvBlock'
    assert type(net.blocks['digit']).__name__ == 'ConditionalBlock'

    x_all = {
        'digit': jnp.zeros(10),
        'color': jnp.ones(3),
        'image': jax.random.normal(key, (IMG_DIM,)),
    }
    v = net(x_all, 0.5)
    assert set(v.keys()) == {'digit', 'color', 'image'}
    for k, dim in structure.dims.items():
        assert v[k].shape == (dim,)
        assert jnp.all(jnp.isfinite(v[k]))

    # active_blocks subset still works with type-aware groups
    v_img = net(x_all, 0.5, active_blocks=['image'])
    assert set(v_img.keys()) == {'image'}


def test_static_flow_with_conv_trains():
    key = jax.random.PRNGKey(2)
    structure = make_structure()
    optimizer = optax.adam(1e-3)

    model, step_fn = create_static_flow(
        structure=structure, depth=3, width=16, model_key=key,
        optimizer=optimizer, use_coupling=False, normaliser=None,
        block_shapes={'image': (H, W, C)}, conv_channels=8,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    batch = {
        'digit': jax.random.normal(key, (4, 10)),
        'color': jax.random.normal(key, (4, 3)),
        'image': jax.random.normal(key, (4, IMG_DIM)),
    }
    model2, _, _, loss = step_fn(model, opt_state, None, batch, jax.random.PRNGKey(3))
    assert jnp.isfinite(loss)

    # Parameters of the conv block actually received gradient updates.
    before = eqx.filter(model, eqx.is_array)
    after = eqx.filter(model2, eqx.is_array)
    diffs = jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), before, after)
    max_diff = max(float(x) for x in jax.tree_util.tree_leaves(diffs))
    assert max_diff > 0.0
