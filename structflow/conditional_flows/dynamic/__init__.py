"""Dynamic conditioning: all blocks share one ODE solve, and each block sees its parents'
co-evolving states rather than their final values. See README.md in this folder for the
two inference implementations and how they differ.

Network:  net.ParallelStructuredFn (masked, block-triangular MLP; layers.py)
Flow:     parallel_flow.ParallelFlow (integrates every block jointly)
Default:  flow.DynamicFlow / dist.DynamicDistribution (``mode="dynamic"``)
Other:    parallel_dist.ParallelBlockTriangularDistribution (full-system re-integration)
"""

from structflow.conditional_flows.dynamic.subset_velocity import (
    clip_tail,
    ClippedParallelFn,
    SubsetVelocityFn,
    interp_trajectory,
)
from structflow.conditional_flows.dynamic.flow import DynamicFlow
from structflow.conditional_flows.dynamic.dist import DynamicDistribution
from structflow.conditional_flows.dynamic.parallel_flow import ParallelFlow
from structflow.conditional_flows.dynamic.parallel_dist import ParallelBlockTriangularDistribution

__all__ = [
    "clip_tail",
    "ClippedParallelFn",
    "SubsetVelocityFn",
    "interp_trajectory",
    "DynamicFlow",
    "DynamicDistribution",
    "ParallelFlow",
    "ParallelBlockTriangularDistribution",
]
