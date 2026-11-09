import jax
import jax.numpy as jnp
import pytest
from structflow.pgms.image_benchmarks import DigitsPGM

def test_digits_pgm_init():
    pgm = DigitsPGM()
    assert pgm.images.shape == (1797, 64)
    assert pgm.labels.shape == (1797,)
    assert pgm.images.dtype == jnp.float32
    assert pgm.labels.dtype == jnp.int32

def test_digits_pgm_sample_joint():
    pgm = DigitsPGM()
    key = jax.random.PRNGKey(0)
    sample = pgm.sample(key)
    
    assert 'image' in sample
    assert 'label' in sample
    assert sample['image'].shape == (64,)
    assert sample['label'].shape == (10,)

def test_digits_pgm_sample_conditional():
    pgm = DigitsPGM()
    key = jax.random.PRNGKey(0)
    
    # Sample p(image | label=5)
    sample = pgm.sample(key, conditions={'label': 5}, targets={'image'})
    
    assert 'image' in sample
    assert 'label' not in sample
    assert sample['image'].shape == (64,)
    
    # Sample again to check label is indeed 5 in the dataset for that index
    # (The sample method returns the label if it's in the joint sample before filtering)
    full_sample = pgm.sample(key, conditions={'label': 5})
    assert jnp.argmax(full_sample['label']) == 5

def test_digits_pgm_structure():
    pgm = DigitsPGM()
    structure = pgm.get_pgm_structure()
    assert structure == {'label': [], 'image': ['label']}

def test_digits_pgm_defaults():
    pgm = DigitsPGM()
    targets, conditions = pgm.get_inference_targets_and_conditions()
    assert targets == {'label'}
    assert conditions == {'image'}

def test_digits_pgm_dims():
    pgm = DigitsPGM()
    dims = pgm.variable_dims
    assert dims['image'] == 64
    assert dims['label'] == 10
