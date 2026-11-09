Here we use flows to build conditional distributions.

The common definition for a conditional distribution is;

- sample(key, conditions: Dict[str, jnp.ndarray], targets: Set[str])
- log_prob(data: Dict[str, jnp.ndarray], conditions: Dict[str, jnp.ndarray], targets: Set[str])

We implement 3 types of conditional flows:

- `StaticConditionalDistribution`: 
- `BlockTriangularDistribution`: A block-triangular flow to transform simple base distributions (typically Gaussians) into complex structured distributions via a velocity field defined on a dependency graph. 
- `GuidanceBasedDistribution`: Uses reconstruction guidance to support arbitrary conditional queries p(targets | conditions) from a joint distribution.