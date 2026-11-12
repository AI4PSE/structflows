"""Conditional distributions ``p(targets | conditions)`` via structured flows.

Three conditioning paradigms are provided:

* ``static`` — autoregressive block-triangular conditioning (conditions are
  integrated to completion, then fed as fixed context);
* ``dynamic`` — all variables integrated jointly in a single ODE solve,
  conditioning on the co-evolving trajectory (see ``dynamic/README.md``);
* ``guidance`` — reconstruction-guided sampling supporting arbitrary queries.

The ``create_*`` factory functions in :mod:`conditional_flows.flow_factory`
build trainable flows, and those in :mod:`conditional_flows.dist_factory` wrap a
trained model as a queryable distribution. See ``conditional_flows/README.md``.

The public names are imported lazily (PEP 562): the factory modules depend on
:mod:`pgms`, which in turn depends on a submodule of this package, so importing
them eagerly here would create an import cycle. Accessing e.g.
``conditional_flows.create_distribution`` triggers the import on first use.
"""

import importlib

__version__ = "0.1.0"

# Public name -> submodule that defines it.
_EXPORTS = {
    "create_joint_flow": "structflow.conditional_flows.flow_factory",
    "create_static_flow": "structflow.conditional_flows.flow_factory",
    "create_dynamic_flow": "structflow.conditional_flows.flow_factory",
    "create_carefl_flow": "structflow.conditional_flows.flow_factory",
    "create_universal_flow": "structflow.conditional_flows.flow_factory",
    "create_dcm_flow": "structflow.conditional_flows.flow_factory",
    "create_distribution": "structflow.conditional_flows.dist_factory",
    "create_static_distribution": "structflow.conditional_flows.dist_factory",
    "create_guided_distribution": "structflow.conditional_flows.dist_factory",
    "create_parallel_distribution": "structflow.conditional_flows.dist_factory",
    "create_dynamic_distribution": "structflow.conditional_flows.dist_factory",
    "create_dcm_distribution": "structflow.conditional_flows.dist_factory",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Lazily import public factory functions on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
