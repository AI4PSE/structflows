"""ε-prediction networks for the DCM (per-node diffusion) backbone.

The whole point of the backbone ablation is that the *structure handling* is
shared with the CNF backbone: ``DiffusionBlock`` reuses ``ConditionalBlock``
(same ``(input_dim, context_dims, width, depth)`` parameterization, same
``(t, x, *parents)`` call signature, same parameter count at equal width), and
the per-graph container is the untouched ``StructuredVelocityFn`` consuming
``structure.dependencies[block]`` for each block's conditioning set. Only the
*interpretation* of the output head changes: it predicts the noise ε̂ that
corrupted ``x_t``, not a flow velocity.
"""

import jax
from typing import Callable

from structflow.pgms.structure import Structure
from structflow.conditional_flows.static.net import ConditionalBlock, StructuredVelocityFn


class DiffusionBlock(ConditionalBlock):
    """ε-prediction MLP ``(x_t, pa, t) -> ε̂``.

    Architecturally identical to ``ConditionalBlock`` (matched parameter
    budget with the CNF backbone by construction); the output is interpreted
    as the predicted corruption noise under the VP cosine schedule in
    :mod:`conditional_flows.dcm.schedule`.
    """
    pass


def build_structured_eps_fn(structure: Structure,
                            key: jax.Array,
                            width: int,
                            depth: int) -> StructuredVelocityFn:
    """Build a structured ε-prediction network: one DiffusionBlock per graph
    block, wired by ``structure.dependencies`` exactly as the CNF backbone."""

    def block_constructor(input_dim, context_dims, key):
        return DiffusionBlock(input_dim, context_dims, key,
                              width=width, depth=depth)

    return StructuredVelocityFn(structure, key, block_constructor)
