# bayes

Use neural flows (and closed-form baselines) to power computations with
probabilities and distributions. Everything is expressed through the
[`Distribution`](distribution.py) interface — `sample(key)`, `log_prob(x)`, and
their batched/JIT variants (`b_sample`, `b_log_prob`, ...).

## Modules

| Module | Contents |
| :--- | :--- |
| [`distribution.py`](distribution.py) | The `Distribution` ABC and concrete distributions: `Gaussian`, `GaussianMixture`, `IsotropicLaplace`, `DictDistribution`, and the flow wrappers `FlowDistribution` / `ProcessDistribution` (wrap a trained `flow.Flow` / `flow.Process`). |
| [`sampling.py`](sampling.py) | `run_mcmc` (a thin BlackJAX/NUTS wrapper) and posterior distributions: `MCMCPosterior` and `SimulationBasedPosterior`. |
| [`metrics.py`](metrics.py) | Distribution-comparison metrics — sliced-Wasserstein (`compute_swd_jax`), Sinkhorn (`compute_sinkhorn_jax`), KL (`compute_kl_jax`), the convenience `compare_distributions`, and flow-dynamics diagnostics (`analyze_flow_dynamics`). |

What this enables:

- **Modelling a distribution** (`distribution.py`): sample $x \sim p(x)$ and
  evaluate $p(x)$ / $\log p(x)$.
- **Bayesian reasoning** (`sampling.py`): given a likelihood and prior, obtain a
  posterior $p(\theta \mid D) = \frac{p(D \mid \theta)\,p(\theta)}{p(D)}$ via
  MCMC, or wrap simulation-based samples as a `Distribution`.
- **Expectations / entropy**: because a `Distribution` exposes both samples and
  densities, expectations such as the mean $\mathbb{E}_{p}[x]$ and entropy
  $-\mathbb{E}_{p}[\log p(x)]$ are Monte-Carlo estimable (`Distribution.entropy`).
- **Comparing distributions** (`metrics.py`): SWD, Sinkhorn, and KL.

## Notes

- Because flows provide access to $x \sim p(x)$, Monte-Carlo estimates of
  expectations converge at the usual $\mathcal{O}(1/\sqrt{n})$ rate.
- A posterior can be folded into the flow non-parametrically via
  $\nabla \log p(x_t \mid D) = \nabla \log p(D \mid x_t) + \nabla \log p(x_t)$ —
  i.e. *guidance*, derived directly from Bayes' rule. See the
  `conditional_flows.guidance` package.
