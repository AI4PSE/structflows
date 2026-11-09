"""
DynamicV2: Improved parallel flow inference via weight clipping and trajectory interpolation.

This module implements an alternative inference strategy for ParallelStructuredFn:
- Tail clipping: Remove unused child variables by slicing weight matrices
- Head conditioning: Interpolate parent variable trajectories instead of integrating them

This provides exact condition fidelity and integrates only the target subset.
"""

from structflow.conditional_flows.dynamicv2.subset_velocity import (
    clip_tail,
    ClippedParallelFn,
    SubsetVelocityFn,
    interp_trajectory,
)
from structflow.conditional_flows.dynamicv2.flow import DynamicV2Flow
from structflow.conditional_flows.dynamicv2.dist import DynamicV2Distribution

__all__ = [
    "clip_tail",
    "ClippedParallelFn",
    "SubsetVelocityFn",
    "interp_trajectory",
    "DynamicV2Flow",
    "DynamicV2Distribution",
]
