"""StructFlow: compile probabilistic graphical models into structured continuous normalizing flows.

Subpackages:
  structflow.pgms               PGM benchmarks and the PGM-to-flow compiler
  structflow.conditional_flows  static / dynamic / guided conditional flows (+ CAREFL, DCM baselines)
  structflow.bayes              distribution objects, NUTS reference sampling, metrics

Built on `sinterp` (interpolants, couplings) and `flow` (ODE/SDE integration).
"""
__version__ = "0.1.0"
