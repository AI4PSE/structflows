import jax
import jax.numpy as jnp
from typing import Dict, List, Set, Optional, Union
from structflow.pgms.utilities import BasePGM
from sklearn.datasets import load_digits, fetch_openml

def load_mnist():
    """Load MNIST dataset via sklearn."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    return mnist.data.astype('float32'), mnist.target.astype('int32')

def one_hot_encode(labels: jnp.ndarray, n_classes: int) -> jnp.ndarray:
    """Convert integer labels to one-hot encoding."""
    return jax.nn.one_hot(labels, n_classes)

class DigitsPGM(BasePGM):
    """
    A PGM wrapper for the sklearn Digits dataset.
    The PGM structure is: label -> image
    
    Default targets/conditions: targets={image}, conditions={label}
    """
    images: jnp.ndarray
    labels: jnp.ndarray
    
    def __init__(self):
        digits = load_digits()
        # Flattened images (1797, 64)
        self.images = jnp.array(digits.data, dtype=jnp.float32).reshape(-1, 64)
        # Rescale images to [0, 1]
        # self.images = self.images / self.images.max()
        # normalization (1797, 64)
        # self.images = (self.images - jnp.mean(self.images)) / jnp.std(self.images)
        # Labels (1797,) as integers
        self.labels = jnp.array(digits.target, dtype=jnp.int32)

    def sample(self,
               key: jax.Array,
               conditions: Optional[Union[Dict[str, jnp.ndarray], Set[str]]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None,
               training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Sample from the dataset.
        If conditions={'label': L}, samples images with that label.
        If conditions is None, samples from the joint (image, label).
        """
        n_samples = self.images.shape[0]
        n_train = n_samples * 6 // 7
        
        if training:
            start_idx, end_idx = 0, n_train
        else:
            start_idx, end_idx = n_train, n_samples
        
        if conditions is not None and isinstance(conditions, dict) and 'label' in conditions:
            label_val = jnp.asarray(conditions['label'])
            # label_val might be integer index or one-hot.
            if label_val.ndim == 0 or (label_val.ndim == 1 and label_val.shape[0] == 1):
                target_label = label_val
            else:
                target_label = jnp.argmax(label_val, axis=-1)
            
            # Mask indices that don't match the label
            # Use categorical on logits to avoid jnp.where (which triggers concretization error)
            logits = jnp.where(self.labels == target_label, 0.0, -jnp.inf)
            
            # Mask for training/test split
            split_mask = (jnp.arange(n_samples) >= start_idx) & (jnp.arange(n_samples) < end_idx)
            logits = jnp.where(split_mask, logits, -jnp.inf)
            
            idx = jax.random.categorical(key, logits)
            
            sample = {'image': self.images[idx], 'label': one_hot_encode(self.labels[idx], 10)}
        else:
            # Joint sampling
            idx = jax.random.randint(key, (), start_idx, end_idx)
            sample = {'image': self.images[idx], 'label': one_hot_encode(self.labels[idx], 10)}

        if targets:
            return {k: v for k, v in sample.items() if k in targets}
        return sample

    def log_prob(self, 
                 data: Dict[str, jnp.ndarray], 
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Analytical log_prob not available for empirical Digits dataset.
        """
        raise NotImplementedError("Analytical log_prob not available for empirical Digits dataset.")

    def get_pgm_structure(self) -> Dict[str, List[str]]:
        return {
            'label': [],
            'image': ['label']
        }

    def get_inference_targets_and_conditions(self) -> (Set[str], Set[str]):
        return {'label'}, {'image'}

    def get_supported_distributions(self) -> List[Dict[str, frozenset[str]]]:
        dists = super().get_supported_distributions()
        # p(label)
        dists.append({"targets": frozenset(['label']), "conditions": frozenset()})
        # p(image | label)
        dists.append({"targets": frozenset(['image']), "conditions": frozenset(['label'])})
        return dists

    @property
    def variable_dims(self) -> Dict[str, int]:
        return {
            'image': 64,
            'label': 10
        }

import numpy as np

class MNISTPGM(BasePGM):
    """
    A PGM wrapper for the sklearn MNIST dataset.
    The PGM structure is: label -> image
    
    Default targets/conditions: targets={image}, conditions={label}
    """
    images: jnp.ndarray
    labels: jnp.ndarray
    
    def __init__(self):
        data, target = load_mnist()
        
        # Process on CPU using numpy to avoid OOM
        # Flattened images (70000, 784)
        data = np.array(data, dtype=np.float32).reshape(-1, 784)
        # normalized images (70000, 784)
        # data = (data - np.mean(data)) / (np.std(data) + 1e-6)
        
        # Labels (70000, 10)
        # target is int32 array. Convert to one-hot in numpy.
        # target is int32 array.
        target = np.array(target, dtype=np.int32).flatten()
        
        # Put on CPU explicitly
        cpu = jax.devices("cpu")[0]
        self.images = jax.device_put(jnp.array(data), cpu)
        self.labels = jax.device_put(jnp.array(target), cpu)

    def sample(self,
               key: jax.Array,
               conditions: Optional[Union[Dict[str, jnp.ndarray], Set[str]]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None,
               training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Sample from the dataset.
        If conditions={'label': L}, samples images with that label.
        If conditions is None, samples from the joint (image, label).
        """
        n_samples = self.images.shape[0]
        n_train = n_samples * 6 // 7
        
        if training:
            start_idx, end_idx = 0, n_train
        else:
            start_idx, end_idx = n_train, n_samples
        
        if conditions is not None and isinstance(conditions, dict) and 'label' in conditions:
            label_val = jnp.asarray(conditions['label'])
            
            # label_val might be integer index or one-hot.
            if label_val.ndim == 0 or (label_val.ndim == 1 and label_val.shape[0] == 1):
                target_label = label_val
            else:
                target_label = jnp.argmax(label_val, axis=-1)
            
            # Mask indices that don't match the label
            # Use categorical on logits to avoid jnp.where (which triggers concretization error)
            logits = jnp.where(self.labels == target_label, 0.0, -jnp.inf)
            
            # Mask for training/test split
            split_mask = (jnp.arange(n_samples) >= start_idx) & (jnp.arange(n_samples) < end_idx)
            logits = jnp.where(split_mask, logits, -jnp.inf)
            
            idx = jax.random.categorical(key, logits)
            
            sample = {'image': self.images[idx], 'label': one_hot_encode(self.labels[idx], 10)}
        else:
            # Joint sampling
            idx = jax.random.randint(key, (), start_idx, end_idx)
            sample = {'image': self.images[idx], 'label': one_hot_encode(self.labels[idx], 10)}

        if targets:
            return {k: v for k, v in sample.items() if k in targets}
        return sample

    def log_prob(self, 
                 data: Dict[str, jnp.ndarray], 
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Analytical log_prob not available for empirical MNIST dataset.
        """
        raise NotImplementedError("Analytical log_prob not available for empirical MNIST dataset.")

    def get_pgm_structure(self) -> Dict[str, List[str]]:
        return {
            'label': [],
            'image': ['label']
        }

    def get_inference_targets_and_conditions(self) -> (Set[str], Set[str]):
        return {'label'}, {'image'}

    def get_supported_distributions(self) -> List[Dict[str, frozenset[str]]]:
        dists = super().get_supported_distributions()
        # p(label)
        dists.append({"targets": frozenset(['label']), "conditions": frozenset()})
        # p(image | label)
        dists.append({"targets": frozenset(['image']), "conditions": frozenset(['label'])})
        return dists

    @property
    def variable_dims(self) -> Dict[str, int]:
        return {
            'image': 784,
            'label': 10
        }
