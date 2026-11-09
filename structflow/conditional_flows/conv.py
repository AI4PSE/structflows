"""Convolutional velocity/denoiser blocks for image-shaped variables.

`ConvBlock` is a drop-in alternative to `ConditionalBlock` for block-triangular
flows whose variable is an image stored flat in (pixel, channel) layout. It is
a small two-scale U-Net with FiLM conditioning: the time embedding and all
(vector-valued) parent variables are embedded once and modulate every residual
stage. The interface matches `Block`: `__call__(t, x_flat, *parents) -> flat`.
"""
import math
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from structflow.conditional_flows.static.net import Block
from flow.net import FourierFeatures


def _groups_for(channels: int) -> int:
    return math.gcd(8, channels)


class FiLM(eqx.Module):
    """Feature-wise linear modulation of a (C, H, W) map from an embedding."""
    linear: eqx.nn.Linear

    def __init__(self, emb_dim: int, channels: int, key: jax.Array):
        self.linear = eqx.nn.Linear(emb_dim, 2 * channels, key=key)

    def __call__(self, h: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        scale, shift = jnp.split(self.linear(emb), 2)
        return h * (1.0 + scale[:, None, None]) + shift[:, None, None]


class ResConvBlock(eqx.Module):
    """GroupNorm -> SiLU -> Conv -> FiLM -> GroupNorm -> SiLU -> Conv, residual."""
    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    film: FiLM
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    skip: Optional[eqx.nn.Conv2d]

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.GroupNorm(_groups_for(in_ch), in_ch)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=k1)
        self.film = FiLM(emb_dim, out_ch, k2)
        self.norm2 = eqx.nn.GroupNorm(_groups_for(out_ch), out_ch)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, 3, padding=1, key=k3)
        self.skip = None if in_ch == out_ch else eqx.nn.Conv2d(in_ch, out_ch, 1, key=k4)

    def __call__(self, h: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        r = self.conv1(jax.nn.silu(self.norm1(h)))
        r = self.film(r, emb)
        r = self.conv2(jax.nn.silu(self.norm2(r)))
        base = h if self.skip is None else self.skip(h)
        return base + r


class ConvBlock(Block):
    """
    Two-scale U-Net block for an image variable of spatial shape (H, W, C),
    stored flat in (pixel, channel) layout, i.e. x.reshape(H, W, C).

    Conditioning: the Fourier time embedding and all parents (flat vectors)
    are concatenated, passed through a 2-layer MLP, and injected into every
    residual stage via FiLM. H and W must be divisible by 4.
    """
    shape: Tuple[int, int, int] = eqx.field(static=True)  # (H, W, C)
    time_embed: FourierFeatures
    emb_in: eqx.nn.Linear
    emb_out: eqx.nn.Linear
    stem: eqx.nn.Conv2d
    res_down1: ResConvBlock
    down1: eqx.nn.Conv2d
    res_down2: ResConvBlock
    down2: eqx.nn.Conv2d
    res_mid: ResConvBlock
    up1: eqx.nn.ConvTranspose2d
    res_up1: ResConvBlock
    up2: eqx.nn.ConvTranspose2d
    res_up2: ResConvBlock
    head_norm: eqx.nn.GroupNorm
    head: eqx.nn.Conv2d

    def __init__(self,
                 input_dim: int,
                 context_dims: List[int],
                 key: jax.Array,
                 shape: Tuple[int, int, int],
                 base_channels: int = 64,
                 emb_dim: int = 128,
                 time_embed_dim: int = 16,
                 **kwargs):
        super().__init__(input_dim, list(context_dims))

        h, w, c = shape
        if h * w * c != input_dim:
            raise ValueError(f"shape {shape} does not match input_dim {input_dim}")
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError(f"H and W must be divisible by 4, got {shape}")
        object.__setattr__(self, "shape", (h, w, c))

        ch = base_channels
        keys = jax.random.split(key, 14)

        self.time_embed = FourierFeatures(output_features=time_embed_dim, key=keys[0])
        cond_dim = time_embed_dim + sum(context_dims)
        self.emb_in = eqx.nn.Linear(cond_dim, emb_dim, key=keys[1])
        self.emb_out = eqx.nn.Linear(emb_dim, emb_dim, key=keys[2])

        self.stem = eqx.nn.Conv2d(c, ch, 3, padding=1, key=keys[3])
        self.res_down1 = ResConvBlock(ch, ch, emb_dim, keys[4])
        self.down1 = eqx.nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1, key=keys[5])
        self.res_down2 = ResConvBlock(2 * ch, 2 * ch, emb_dim, keys[6])
        self.down2 = eqx.nn.Conv2d(2 * ch, 4 * ch, 3, stride=2, padding=1, key=keys[7])
        self.res_mid = ResConvBlock(4 * ch, 4 * ch, emb_dim, keys[8])
        self.up1 = eqx.nn.ConvTranspose2d(4 * ch, 2 * ch, 4, stride=2, padding=1, key=keys[9])
        self.res_up1 = ResConvBlock(4 * ch, 2 * ch, emb_dim, keys[10])
        self.up2 = eqx.nn.ConvTranspose2d(2 * ch, ch, 4, stride=2, padding=1, key=keys[11])
        self.res_up2 = ResConvBlock(2 * ch, ch, emb_dim, keys[12])
        self.head_norm = eqx.nn.GroupNorm(_groups_for(ch), ch)
        self.head = eqx.nn.Conv2d(ch, c, 3, padding=1, key=keys[13])

    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, *parents: jnp.ndarray) -> jnp.ndarray:
        self.validate_input(t, x, *parents)
        h_dim, w_dim, c_dim = self.shape

        t_embed = self.time_embed(t).squeeze()
        cond = jnp.concatenate([t_embed] + list(parents)) if parents else t_embed
        emb = jax.nn.silu(self.emb_in(cond))
        emb = self.emb_out(emb)

        img = x.reshape(h_dim, w_dim, c_dim).transpose(2, 0, 1)  # (C, H, W)

        h0 = self.stem(img)
        h1 = self.res_down1(h0, emb)              # (ch,   H,   W)
        h2 = self.res_down2(self.down1(h1), emb)  # (2ch,  H/2, W/2)
        h3 = self.res_mid(self.down2(h2), emb)    # (4ch,  H/4, W/4)
        u = self.up1(h3)                          # (2ch,  H/2, W/2)
        u = self.res_up1(jnp.concatenate([u, h2], axis=0), emb)
        u = self.up2(u)                           # (ch,   H,   W)
        u = self.res_up2(jnp.concatenate([u, h1], axis=0), emb)
        out = self.head(jax.nn.silu(self.head_norm(u)))  # (C, H, W)

        return out.transpose(1, 2, 0).reshape(-1)
