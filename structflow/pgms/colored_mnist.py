"""Colored-MNIST PGM with a confounded coloring mechanism.

Causal graph:

    digit -> color
    digit -> image <- color

The digit confounds the color: with probability `rho` the color is the
palette color of the digit, otherwise a uniformly random palette color;
Gaussian jitter of scale `color_noise` is added either way. The image is the
grayscale MNIST digit tinted by the color. Because the coloring mechanism is
authored here (rather than empirical), both observational conditionals
(via the exact digit posterior p(digit | color)) and interventional
distributions (do(color): digit from its marginal) admit ground-truth
sampling, so p(image | color) != p(image | do(color)) whenever rho > 0.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Set, Optional, Union

from structflow.pgms.utilities import BasePGM
from structflow.pgms.image_benchmarks import load_mnist, one_hot_encode

# One base RGB color per digit class (tab10-like, in [0, 1]).
DIGIT_PALETTE = np.array([
    [0.84, 0.15, 0.16],  # 0 red
    [1.00, 0.60, 0.00],  # 1 orange
    [0.90, 0.85, 0.10],  # 2 yellow
    [0.17, 0.63, 0.17],  # 3 green
    [0.10, 0.75, 0.70],  # 4 teal
    [0.12, 0.47, 0.71],  # 5 blue
    [0.35, 0.20, 0.80],  # 6 indigo
    [0.58, 0.40, 0.74],  # 7 purple
    [0.89, 0.47, 0.76],  # 8 pink
    [0.55, 0.34, 0.29],  # 9 brown
], dtype=np.float32)


class ColoredMNISTPGM(BasePGM):
    """
    Three-node PGM on MNIST: digit -> color, (digit, color) -> image.

    Variables:
        digit: 10-d one-hot.
        color: 3-d RGB in [0, 1] (palette color + Gaussian jitter).
        image: 2352-d (28x28x3 tinted grayscale digit, values in [0, 1]).
    """
    images: jnp.ndarray      # (N, 784) grayscale in [0, 1]
    labels: jnp.ndarray      # (N,) int
    palette: jnp.ndarray     # (10, 3)
    log_pd_train: jnp.ndarray  # (10,) empirical log p(digit) on the train split
    log_pd_test: jnp.ndarray   # (10,) on the test split
    rho: float
    color_noise: float

    def __init__(self, rho: float = 0.8, color_noise: float = 0.05):
        data, target = load_mnist()
        data = np.array(data, dtype=np.float32).reshape(-1, 784) / 255.0
        target = np.array(target, dtype=np.int32).flatten()

        n = data.shape[0]
        n_train = n * 6 // 7
        counts_train = np.bincount(target[:n_train], minlength=10).astype(np.float32)
        counts_test = np.bincount(target[n_train:], minlength=10).astype(np.float32)

        cpu = jax.devices("cpu")[0]
        self.images = jax.device_put(jnp.array(data), cpu)
        self.labels = jax.device_put(jnp.array(target), cpu)
        self.palette = jnp.array(DIGIT_PALETTE)
        self.log_pd_train = jnp.log(counts_train / counts_train.sum())
        self.log_pd_test = jnp.log(counts_test / counts_test.sum())
        self.rho = rho
        self.color_noise = color_noise

    # --- mechanisms -------------------------------------------------------

    def _split_bounds(self, training: bool):
        n = self.images.shape[0]
        n_train = n * 6 // 7
        return (0, n_train) if training else (n_train, n)

    def _log_mix_weights(self) -> jnp.ndarray:
        """(10, 10) matrix W[d, k] = log p(palette component k | digit d)."""
        w = (1.0 - self.rho) / 10.0 * jnp.ones((10, 10)) + self.rho * jnp.eye(10)
        return jnp.log(w)

    def _sample_color_given_digit(self, key: jax.Array, digit_idx: jnp.ndarray) -> jnp.ndarray:
        k_comp, k_noise = jax.random.split(key)
        comp = jax.random.categorical(k_comp, self._log_mix_weights()[digit_idx])
        noise = self.color_noise * jax.random.normal(k_noise, (3,))
        return self.palette[comp] + noise

    def _digit_posterior_logits(self, color: jnp.ndarray, training: bool) -> jnp.ndarray:
        """log p(digit | color) up to a constant, exact under the mixture DGP."""
        sq = jnp.sum((color[None, :] - self.palette) ** 2, axis=-1)  # (10,)
        log_norm = -0.5 * sq / (self.color_noise ** 2) \
            - 1.5 * jnp.log(2 * jnp.pi * self.color_noise ** 2)      # (10,) per component
        log_mix = jax.scipy.special.logsumexp(
            self._log_mix_weights() + log_norm[None, :], axis=-1)     # (10,) per digit
        log_pd = self.log_pd_train if training else self.log_pd_test
        return log_pd + log_mix

    def _sample_image_given(self, key: jax.Array, digit_idx: jnp.ndarray,
                            color: jnp.ndarray, training: bool) -> jnp.ndarray:
        start_idx, end_idx = self._split_bounds(training)
        n = self.images.shape[0]
        logits = jnp.where(self.labels == digit_idx, 0.0, -jnp.inf)
        split_mask = (jnp.arange(n) >= start_idx) & (jnp.arange(n) < end_idx)
        logits = jnp.where(split_mask, logits, -jnp.inf)
        idx = jax.random.categorical(key, logits)
        gray = self.images[idx]                             # (784,)
        return (gray[:, None] * color[None, :]).reshape(-1)  # (2352,)

    @staticmethod
    def _digit_index(value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value)
        if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
            return value.astype(jnp.int32).reshape(())
        return jnp.argmax(value, axis=-1)

    # --- sampling ---------------------------------------------------------

    def sample(self,
               key: jax.Array,
               conditions: Optional[Union[Dict[str, jnp.ndarray], Set[str]]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None,
               training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Supported conditioning sets: {}, {digit}, {color}, {digit, color}.
        Conditioning on the image is not supported (requires inversion).
        """
        conditions = conditions if isinstance(conditions, dict) else {}
        if 'image' in conditions:
            raise NotImplementedError("Conditioning on the image is not supported.")

        k_digit, k_color, k_image = jax.random.split(key, 3)

        if 'digit' in conditions:
            digit_idx = self._digit_index(conditions['digit'])
        elif 'color' in conditions:
            logits = self._digit_posterior_logits(jnp.asarray(conditions['color']), training)
            digit_idx = jax.random.categorical(k_digit, logits)
        else:
            start_idx, end_idx = self._split_bounds(training)
            idx = jax.random.randint(k_digit, (), start_idx, end_idx)
            digit_idx = self.labels[idx]

        if 'color' in conditions:
            color = jnp.asarray(conditions['color'])
        else:
            color = self._sample_color_given_digit(k_color, digit_idx)

        image = self._sample_image_given(k_image, digit_idx, color, training)

        sample = {
            'digit': one_hot_encode(digit_idx, 10),
            'color': color,
            'image': image,
        }
        if targets:
            return {k: v for k, v in sample.items() if k in targets}
        return sample

    def sample_interventional(self,
                              key: jax.Array,
                              interventions: Dict[str, jnp.ndarray],
                              targets: Optional[Union[Set[str], List[str]]] = None,
                              training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Sample the mutilated graph: intervened nodes are clamped, all other
        nodes follow their mechanisms. E.g. do(color): digit from its marginal
        (unchanged), image tinted with the forced color.
        """
        if 'image' in interventions:
            raise NotImplementedError("Intervening on the image is not supported.")

        k_digit, k_color, k_image = jax.random.split(key, 3)

        if 'digit' in interventions:
            digit_idx = self._digit_index(interventions['digit'])
        else:
            start_idx, end_idx = self._split_bounds(training)
            idx = jax.random.randint(k_digit, (), start_idx, end_idx)
            digit_idx = self.labels[idx]

        if 'color' in interventions:
            color = jnp.asarray(interventions['color'])
        else:
            color = self._sample_color_given_digit(k_color, digit_idx)

        image = self._sample_image_given(k_image, digit_idx, color, training)

        sample = {
            'digit': one_hot_encode(digit_idx, 10),
            'color': color,
            'image': image,
        }
        if targets:
            return {k: v for k, v in sample.items() if k in targets}
        return sample

    def b_sample_interventional(self, key, n_samples, interventions,
                                targets=None, training: bool = True):
        keys = jax.random.split(key, n_samples)
        return jax.vmap(
            lambda k: self.sample_interventional(k, interventions, targets, training)
        )(keys)

    def counterfactual(self,
                       data: Dict[str, jnp.ndarray],
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Set[str]] = None):
        """
        Ground-truth counterfactual for color interventions: the exogenous
        noise of the image mechanism is the grayscale drawing, which is exactly
        recoverable from the factual image (image = gray x color), so
        do(color=c') re-tints the same drawing. Counterfactuals over the digit
        are not well defined for the empirical image mechanism (no
        cross-class correspondence of drawings) and raise NotImplementedError.

        Returns (samples, log_prob) matching the StructuralPGM contract;
        log_prob is NaN (empirical image density unavailable).
        """
        unsupported = set(interventions) - {'color'}
        if unsupported:
            raise NotImplementedError(
                f"Ground-truth counterfactuals only support do(color); got {unsupported}.")

        factual_color = jnp.asarray(data['color'])
        cf_color = jnp.asarray(interventions['color'])

        img = jnp.asarray(data['image']).reshape(784, 3)
        # Least-squares inversion of image = gray x color (exact for the DGP).
        gray = img @ factual_color / (factual_color @ factual_color + 1e-12)
        cf_image = (gray[:, None] * cf_color[None, :]).reshape(-1)

        sample = {
            'digit': jnp.asarray(data['digit']),
            'color': cf_color,
            'image': cf_image,
        }
        if targets:
            sample = {k: v for k, v in sample.items() if k in targets}
        return sample, jnp.array(jnp.nan)

    # --- interface --------------------------------------------------------

    def log_prob(self, data, targets=None, conditions=None) -> jnp.ndarray:
        raise NotImplementedError("Analytical log_prob not available for empirical MNIST images.")

    def get_pgm_structure(self) -> Dict[str, List[str]]:
        return {
            'digit': [],
            'color': ['digit'],
            'image': ['digit', 'color'],
        }

    def get_inference_targets_and_conditions(self) -> (Set[str], Set[str]):
        # Observational inference against the causal direction: given a color,
        # infer the digit belief and the image distribution. Compiling with
        # conditions={color} also makes p(digit | color, image) tractable.
        return {'digit', 'image'}, {'color'}

    def get_supported_distributions(self) -> List[Dict[str, frozenset[str]]]:
        dists = super().get_supported_distributions()
        supported = [
            ({'digit'}, set()),
            ({'color'}, set()),
            ({'digit', 'color'}, set()),
            ({'color'}, {'digit'}),
            ({'color', 'image'}, {'digit'}),
            ({'image'}, {'digit', 'color'}),
            ({'digit'}, {'color'}),
            ({'image'}, {'color'}),
        ]
        for t, c in supported:
            dists.append({"targets": frozenset(t), "conditions": frozenset(c)})
        return dists

    @property
    def variable_dims(self) -> Dict[str, int]:
        return {
            'digit': 10,
            'color': 3,
            'image': 2352,
        }

    @property
    def variable_shapes(self) -> Dict[str, tuple]:
        """Spatial shapes of image-like variables, in (H, W, C) flat layout.
        Consumed by shape-aware flow factories (conv blocks)."""
        return {'image': (28, 28, 3)}
