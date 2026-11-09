"""Distribution objects, posteriors, MAP estimation, and comparison metrics.

Built on top of ``flow``: ``FlowDistribution`` / ``ProcessDistribution`` wrap a
trained flow as a :class:`Distribution` exposing ``sample`` and ``log_prob``.
The module also provides closed-form distributions (``Gaussian``,
``GaussianMixture``, ...), MCMC-based posteriors, and metrics (SWD, Sinkhorn,
KL) for comparing distributions. See ``bayes/README.md``.
"""

from structflow.bayes.distribution import (
    Distribution,
    Gaussian,
    GaussianMixture,
    FlowDistribution,
    ProcessDistribution,
    DictDistribution,
    IsotropicLaplace,
    gmm_log_p,
)
from structflow.bayes.sampling import run_mcmc, SimulationBasedPosterior, MCMCPosterior
from structflow.bayes.metrics import (
    compare_distributions,
    compute_swd_jax,
    compute_sinkhorn_jax,
    compute_kl_jax,
    analyze_flow_dynamics,
    compute_counterfactual,
)

__version__ = "0.1.0"

__all__ = [
    "Distribution",
    "Gaussian",
    "GaussianMixture",
    "FlowDistribution",
    "ProcessDistribution",
    "DictDistribution",
    "IsotropicLaplace",
    "gmm_log_p",
    "run_mcmc",
    "SimulationBasedPosterior",
    "MCMCPosterior",
    "compare_distributions",
    "compute_swd_jax",
    "compute_sinkhorn_jax",
    "compute_kl_jax",
    "analyze_flow_dynamics",
    "compute_counterfactual",
]
