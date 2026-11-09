import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Optional, Union, Set, Iterator
from abc import ABC, abstractmethod

class Structure(ABC):
    """
    Abstract base class for defining block-structured dependency graphs.
    
    A Structure defines how variables (blocks) depend on each other in a directed
    acyclic graph (DAG). This structure is used by flows and distributions to
    determine which conditional distributions are tractable and which variables
    need to be simulated for a given query.
    
    Attributes:
        block_order: List of block names in topological order. Each block represents
            a variable or group of variables in the model.
        dependencies: Dictionary mapping each block to its parents (direct dependencies).
            Example: {'z': ['x', 'y'], 'y': ['x'], 'x': []}
        dims: Dictionary mapping each block name to its dimensionality.
            Example: {'x': 1, 'y': 2, 'z': 3}
    
    Example:
        >>> # Define a simple chain structure: x -> y -> z
        >>> structure = BlockTriangularStructure(
        ...     block_order=['x', 'y', 'z'],
        ...     dependencies={'x': [], 'y': ['x'], 'z': ['y']},
        ...     dims={'x': 1, 'y': 2, 'z': 1}
        ... )
        >>> structure.is_tractable_query(targets={'z'}, conditions={'y'})  
        True  # p(z|y) is tractable
        >>> structure.is_tractable_query(targets={'x'}, conditions={'z'})
        False  # p(x|z) requires inverting the flow (intractable)
    """
    def __init__(self, 
        block_order: List[str],
        dependencies: Dict[str, List[str]],
        dims: Dict[str, int],
    ):
        self.block_order = block_order
        self.dependencies = dependencies
        self.dims = dims

    @abstractmethod
    def get_all_descendants(self, block: str) -> Set[str]:
        """
        Get all descendants of the given block (recursively).
        
        Args:
            block: Name of the block to find descendants for.
            
        Returns:
            Set of block names that depend (directly or indirectly) on the given block.
            
        Example:
            For chain x -> y -> z, get_all_descendants('x') returns {'y', 'z'}
        """
        pass

    @abstractmethod
    def is_ancestral_query(self, targets: Set[str], conditions: Set[str]) -> bool:
        """
        Determines if p(targets | conditions) respects causal (ancestral) direction.
        
        A query is ancestral if no condition is a descendant of any target.
        This means we can generate targets via forward (ancestral) simulation,
        potentially sampling intermediate nodes and discarding them.
        
        This is sufficient for sampling but NOT for log_prob evaluation
        (which also requires no marginalisation — see is_tractable_query).
        
        Args:
            targets: Set of blocks to sample.
            conditions: Set of blocks to condition on.
            
        Returns:
            True if conditions are upstream of (or unrelated to) targets,
            False if any condition is a descendant of a target.
        """
        pass

    @abstractmethod
    def query_requires_marginalisation(self, targets: Set[str], conditions: Set[str]) -> bool:
        """
        Checks whether evaluating log p(targets | conditions) would require
        marginalising out intermediate variables.
        
        This is the case when some parent of a target is neither in targets
        nor in conditions — i.e. there is a "gap" node whose value is unknown.
        
        Example:
            For chain x0 -> x1 -> x2, query p(x2 | x0):
            x1 is a parent of x2 but is in neither {x2} nor {x0}, so
            marginalisation over x1 is required → returns True.
        
        Args:
            targets: Set of blocks to compute probabilities for.
            conditions: Set of blocks to condition on.
            
        Returns:
            True if the query requires marginalising out hidden variables,
            False if all parents of targets are known (in targets or conditions).
        """
        pass

    @abstractmethod
    def is_tractable_query(self, targets: Set[str], conditions: Set[str]) -> bool:
        """
        Determines if log p(targets | conditions) is efficiently computable by the flow.
        
        A query is tractable if it satisfies BOTH:
        1. Ancestral direction: no condition is a descendant of any target
           (checked by is_ancestral_query).
        2. No marginalisation: all parents of targets are known
           (checked by NOT query_requires_marginalisation).
        
        Equivalently: is_ancestral_query AND NOT query_requires_marginalisation.
        
        Args:
            targets: Set of blocks to compute probabilities for.
            conditions: Set of blocks to condition on.
            
        Returns:
            True if the query can be efficiently computed using forward flow simulation,
            False if it requires flow inversion or marginalization.
        """
        pass

    @abstractmethod
    def create_dummy_input(self):
        """
        Creates a dummy input dictionary for testing and initialization.
        
        Returns:
            Dictionary mapping block names to zero-initialized arrays of the correct shape.
        """
        pass

    @abstractmethod
    def get_all_ancestors(self, block: str, exclude: Set[str] = None) -> Set[str]:
        """
        Get all ancestor blocks of the given block (recursively), excluding specified blocks.
        
        Args:
            block: Name of the block to find ancestors for.
            exclude: Optional set of blocks to exclude from the ancestry search.
            
        Returns:
            Set of block names that the given block depends on (directly or indirectly),
            excluding any blocks in the exclude set.
        """
        pass

    @abstractmethod
    def get_ancestral_closure(self, blocks: Set[str], exclude: Set[str] = None) -> Set[str]:
        """
        Returns the set of blocks plus all their ancestors (recursively), excluding specified blocks.
        
        Args:
            blocks: Set of blocks to compute closure for.
            exclude: Optional set of blocks to exclude.
            
        Returns:
            The input blocks plus all their ancestors, excluding any in the exclude set.
            
        Note:
            This is useful for determining which variables must be available to compute
            the given blocks.
        """
        pass

    @abstractmethod
    def get_required_simulation_blocks(self, targets: Set[str], conditions: Set[str]) -> List[str]:
        """
        Determines the minimal set of blocks that must be simulated to obtain 'targets'.
        
        Args:
            targets: Set of blocks we want to sample/compute.
            conditions: Set of blocks that are already known/observed.
            
        Returns:
            List of block names in topological order that need to be simulated.
            This includes targets and any of their ancestors not in conditions.
            
        Example:
            For chain x -> y -> z with conditions={'x'} and targets={'z'}:
            Returns ['y', 'z'] (x is already known, y is needed to compute z)
        """
        pass


class BlockTriangularStructure(Structure):
    """
    Defines the structure of a block-triangular flow.
    
    A block-triangular structure represents a directed acyclic graph (DAG) where
    each block (variable) can depend only on blocks that appear earlier in the
    topological ordering. This enables efficient forward sampling and tractable
    conditional probability computation.
    
    The "triangular" property comes from the dependency structure: if we arrange
    the blocks in topological order and represent dependencies as a matrix, it
    has a triangular pattern (no backward dependencies).
    
    Attributes:
        block_order: List of block names in topological order. Later blocks can
            depend on earlier blocks, but not vice versa.
        dependencies: Dictionary mapping each block to its list of parent blocks.
            All dependencies must appear earlier in block_order.
        dims: Dictionary mapping each block name to its dimensionality (int).
    
    Note:
        This uses a custom graph implementation rather than an existing library
        to maintain JAX compatibility, minimize dependencies, and provide a
        simple, focused API tailored to flow-based probabilistic inference.
    
    Example:
        >>> # Chain structure: x -> y -> z
        >>> structure = BlockTriangularStructure(
        ...     block_order=['x', 'y', 'z'],
        ...     dependencies={'x': [], 'y': ['x'], 'z': ['y']},
        ...     dims={'x': 1, 'y': 2, 'z': 1}
        ... )
        >>> structure.is_tractable_query({'z'}, {'y'})
        True
    """
    def __init__(self, 
        block_order: List[str],
        dependencies: Dict[str, List[str]],
        dims: Dict[str, int],
    ):
        """
        Initialize a block-triangular structure.
        
        Args:
            block_order: List of block names in topological order.
            dependencies: Dict mapping each block to its list of parent blocks.
            dims: Dict mapping each block name to its dimensionality.
            
        Raises:
            ValueError: If inputs are invalid (empty blocks, duplicate names,
                invalid dependencies, non-topological order, circular deps, etc.)
        """
        # Validate block_order
        if not block_order:
            raise ValueError("block_order cannot be empty")
        if not isinstance(block_order, list):
            raise ValueError(f"block_order must be a list, got {type(block_order)}")
        if len(block_order) != len(set(block_order)):
            duplicates = [b for b in set(block_order) if block_order.count(b) > 1]
            raise ValueError(f"block_order contains duplicate names: {duplicates}")
        if not all(isinstance(b, str) for b in block_order):
            raise ValueError("All block names must be strings")
            
        # Validate dependencies
        if not isinstance(dependencies, dict):
            raise ValueError(f"dependencies must be a dict, got {type(dependencies)}")
        
        block_set = set(block_order)
        for block, parents in dependencies.items():
            if block not in block_set:
                raise ValueError(f"Dependency key '{block}' not in block_order")
            if not isinstance(parents, list):
                raise ValueError(f"Dependencies for '{block}' must be a list, got {type(parents)}")
            for parent in parents:
                if parent not in block_set:
                    raise ValueError(f"Parent '{parent}' of block '{block}' not in block_order")
                    
        # Ensure all blocks have dependency entries (even if empty)
        for block in block_order:
            if block not in dependencies:
                dependencies[block] = []
                
        # Validate topological ordering
        block_index = {b: i for i, b in enumerate(block_order)}
        for block, parents in dependencies.items():
            block_idx = block_index[block]
            for parent in parents:
                parent_idx = block_index[parent]
                if parent_idx >= block_idx:
                    raise ValueError(
                        f"Invalid topological order: block '{block}' at index {block_idx} "
                        f"depends on '{parent}' at index {parent_idx}. "
                        f"Parents must appear before their children in block_order."
                    )
        
        # Check for circular dependencies (redundant with topo check, but explicit)
        visited = set()
        rec_stack = set()
        
        def has_cycle(block):
            visited.add(block)
            rec_stack.add(block)
            for parent in dependencies.get(block, []):
                if parent not in visited:
                    if has_cycle(parent):
                        return True
                elif parent in rec_stack:
                    return True
            rec_stack.remove(block)
            return False
        
        for block in block_order:
            if block not in visited:
                if has_cycle(block):
                    raise ValueError("Circular dependency detected in block structure")
        
        # Validate dims
        if not isinstance(dims, dict):
            raise ValueError(f"dims must be a dict, got {type(dims)}")
        if set(dims.keys()) != block_set:
            missing = block_set - set(dims.keys())
            extra = set(dims.keys()) - block_set
            msg = []
            if missing:
                msg.append(f"missing dims for blocks: {missing}")
            if extra:
                msg.append(f"extra dims for unknown blocks: {extra}")
            raise ValueError("dims keys must match block_order. " + ", ".join(msg))
        for block, dim in dims.items():
            if not isinstance(dim, int):
                raise ValueError(f"Dimension for '{block}' must be an int, got {type(dim)}")
            if dim <= 0:
                raise ValueError(f"Dimension for '{block}' must be positive, got {dim}")
        
        self.block_order = block_order
        self.dependencies = dependencies
        self.dims = dims

    def get_all_descendants(self, block: str) -> Set[str]:
        """
        Get all descendants of the given block (recursively).
        
        Args:
            block: Name of the block to find descendants for.
            
        Returns:
            Set of block names that depend (directly or indirectly) on the given block.
            
        Raises:
            ValueError: If block is not in the structure.
        """
        if block not in self.block_order:
            raise ValueError(f"Block '{block}' not in structure")
            
        descendants = set()
        
        # Invert dependencies to get children
        children_map = {b: [] for b in self.block_order}
        for b, parents in self.dependencies.items():
            for p in parents:
                children_map[p].append(b)
                
        def collect_descendants(b):
            for child in children_map[b]:
                if child not in descendants:
                    descendants.add(child)
                    collect_descendants(child)
                    
        collect_descendants(block)
        return descendants

    def _validate_query_inputs(self, targets: Set[str], conditions: Set[str]) -> None:
        """
        Validate targets and conditions for query methods.
        
        Raises:
            ValueError: If targets or conditions contain invalid block names,
                or if they overlap.
        """
        block_set = set(self.block_order)
        invalid_targets = targets - block_set
        if invalid_targets:
            raise ValueError(f"Unknown targets: {invalid_targets}")
        invalid_conditions = conditions - block_set
        if invalid_conditions:
            raise ValueError(f"Unknown conditions: {invalid_conditions}")
        
        overlap = targets & conditions
        if overlap:
            raise ValueError(
                f"Targets and conditions must be disjoint, but both contain: {overlap}"
            )

    def is_ancestral_query(self, targets: Set[str], conditions: Set[str]) -> bool:
        """
        Determines if p(targets | conditions) respects causal (ancestral) direction.
        
        A query is ancestral if no condition is a descendant of any target.
        This is sufficient for sampling (we can simulate intermediate nodes
        and discard them), but NOT sufficient for log_prob evaluation.
        
        Args:
            targets: Set of blocks to sample.
            conditions: Set of blocks to condition on.
            
        Returns:
            True if conditions are upstream of (or unrelated to) targets,
            False if any condition is a descendant of a target.
            
        Raises:
            ValueError: If targets or conditions contain invalid block names,
                or if they overlap.
        """
        self._validate_query_inputs(targets, conditions)
        
        # Descendant Rule: No condition is a descendant of any target
        for t in targets:
            descendants = self.get_all_descendants(t)
            if not conditions.isdisjoint(descendants):
                return False
                
        return True

    def query_requires_marginalisation(self, targets: Set[str], conditions: Set[str]) -> bool:
        """
        Checks whether evaluating log p(targets | conditions) would require
        marginalising out intermediate variables.
        
        This is the case when some parent of a target is neither in targets
        nor in conditions — i.e. there is a "gap" node whose value is unknown.
        
        Example:
            For chain x -> y -> z, query p(z | x):
            y is a parent of z but is in neither {z} nor {x}, so
            marginalisation over y is required → returns True.
        
        Args:
            targets: Set of blocks to compute probabilities for.
            conditions: Set of blocks to condition on.
            
        Returns:
            True if the query requires marginalising out hidden variables,
            False if all parents of targets are known (in targets or conditions).
            
        Raises:
            ValueError: If targets or conditions contain invalid block names,
                or if they overlap.
        """
        self._validate_query_inputs(targets, conditions)
        
        known_vars = targets.union(conditions)
        for t in targets:
            parents = set(self.dependencies.get(t, []))
            if not parents.issubset(known_vars):
                return True
        return False

    def is_tractable_query(self, targets: Set[str], conditions: Set[str]) -> bool:
        """
        Determines if log p(targets | conditions) is efficiently computable.
        
        A query is tractable if it satisfies BOTH:
        1. Ancestral direction: no condition is a descendant of any target
           (is_ancestral_query returns True).
        2. No marginalisation: all parents of targets are known
           (query_requires_marginalisation returns False).
        
        Args:
            targets: Set of blocks to compute probabilities for.
            conditions: Set of blocks to condition on.
            
        Returns:
            True if the query can be efficiently computed using forward flow simulation,
            False if it requires flow inversion or marginalization.
            
        Raises:
            ValueError: If targets or conditions contain invalid block names,
                or if they overlap.
        """
        self._validate_query_inputs(targets, conditions)
        return (self.is_ancestral_query(targets, conditions)
                and not self.query_requires_marginalisation(targets, conditions))

    def create_dummy_input(self):
        return {name: jnp.zeros((self.dims[name],)) for name in self.block_order}

    def get_all_ancestors(self, block: str, exclude: Set[str] = None) -> Set[str]:
        """
        Get all ancestor blocks of the given block (recursively), excluding specified blocks.
        
        Args:
            block: Name of the block to find ancestors for.
            exclude: Optional set of blocks to exclude from the ancestry search.
            
        Returns:
            Set of block names that the given block depends on (directly or indirectly),
            excluding any blocks in the exclude set.
            
        Raises:
            ValueError: If block or any exclude blocks are not in the structure.
        """
        if block not in self.block_order:
            raise ValueError(f"Block '{block}' not in structure")
        if exclude is not None:
            block_set = set(self.block_order)
            invalid_exclude = exclude - block_set
            if invalid_exclude:
                raise ValueError(f"Unknown blocks in exclude: {invalid_exclude}")
        
        if exclude is None: exclude = set()
        ancestors = set()
        
        def collect_ancestors(b):
            for parent in self.dependencies.get(b, []):
                if parent not in exclude and parent not in ancestors:
                    ancestors.add(parent)
                    collect_ancestors(parent)
        
        collect_ancestors(block)
        return ancestors

    def get_ancestral_closure(self, blocks: Set[str], exclude: Set[str] = None) -> Set[str]:
        """
        Returns the set of blocks plus all their ancestors (recursively), excluding specified blocks.
        
        Args:
            blocks: Set of blocks to compute closure for.
            exclude: Optional set of blocks to exclude.
            
        Returns:
            The input blocks plus all their ancestors, excluding any in the exclude set.
            
        Raises:
            ValueError: If any blocks or exclude blocks are not in the structure.
        """
        block_set = set(self.block_order)
        invalid_blocks = blocks - block_set
        if invalid_blocks:
            raise ValueError(f"Unknown blocks: {invalid_blocks}")
        if exclude is not None:
            invalid_exclude = exclude - block_set
            if invalid_exclude:
                raise ValueError(f"Unknown blocks in exclude: {invalid_exclude}")
        
        if exclude is None: exclude = set()
        closure = set(blocks)
        for block in list(closure):
            ancestors = self.get_all_ancestors(block, exclude)
            closure.update(ancestors)
        return closure

    def mutilated(self, blocks: Set[str]) -> "BlockTriangularStructure":
        """A copy of this structure with the incoming edges of ``blocks`` deleted.

        This is Pearl's graph surgery. Intervening on a block does not merely
        observe it -- it removes the mechanism that produced it, so the block's
        parents stop being relevant to it while the block itself stays relevant
        to its children. Every other conditional is unchanged, which is exactly
        why one fitted model can answer both observational and interventional
        queries: the surgery deletes blocks, it does not alter the surviving
        ones.

        The topological order is preserved, since removing edges cannot break it.
        """
        unknown = set(blocks) - set(self.block_order)
        if unknown:
            raise ValueError(f"Unknown blocks to intervene on: {sorted(unknown)}")
        deps = {v: ([] if v in blocks else list(ps))
                for v, ps in self.dependencies.items()}
        return BlockTriangularStructure(block_order=list(self.block_order),
                                        dependencies=deps,
                                        dims=dict(self.dims))

    def get_required_simulation_blocks(self, targets: Set[str], conditions: Set[str]) -> List[str]:
        """
        Determines the minimal set of blocks that must be simulated to obtain 'targets'.
        
        Args:
            targets: Set of blocks we want to sample/compute.
            conditions: Set of blocks that are already known/observed.
            
        Returns:
            List of block names in topological order that need to be simulated.
            This includes targets and any of their ancestors not in conditions.
            
        Raises:
            ValueError: If targets or conditions contain invalid block names.
        """
        block_set = set(self.block_order)
        invalid_targets = targets - block_set
        if invalid_targets:
            raise ValueError(f"Unknown targets: {invalid_targets}")
        invalid_conditions = conditions - block_set
        if invalid_conditions:
            raise ValueError(f"Unknown conditions: {invalid_conditions}")
        
        # We assume block_order is topological.
        required = set(targets)
        
        # We repeatedly add parents of required blocks until we hit conditions or roots.
        # Simple fixed-point iteration or just checking dependencies.
        
        # Helper: Get all ancestors (inclusive)
        def get_ancestors(block):
            parents = self.dependencies.get(block, [])
            for p in parents:
                if p not in conditions and p not in required:
                    required.add(p)
                    get_ancestors(p)
        
        # Populate required set
        # Copy to avoid modifying iteration set issues
        initial_targets = list(required)
        for t in initial_targets:
            get_ancestors(t)
            
        # Now filter the global block order to get the correct simulation sequence
        simulation_list = []
        for block in self.block_order:
            if block in required and block not in conditions:
                simulation_list.append(block)
                
        return simulation_list

    def __str__(self):
        return f"Structure({self.block_order})" + "\n" + "\n".join([f"  {block}: {self.dependencies.get(block, [])}" for block in self.block_order])

    def get_tractable_distributions(self) -> List[Dict[str, frozenset[str]]]:
        """
        Returns a list of dictionaries describing all tractable distributions.
        Each dictionary has 'targets' and 'conditions' keys with frozenset values.
        """
        return list(get_all_tractable_distributions(self))

def get_base_tractable_distributions(structure: Structure) -> List[Dict[str, frozenset[str]]]:
    """
    Returns the base conditional distributions directly implemented by the structure.
    
    For a structure where each block depends on some parents, these are the 
    p(block | parents) distributions.
    
    Args:
        structure: The dependency structure.
        
    Returns:
        List of dictionaries for each base distribution.
    """
    base_tractable = []
    for block in structure.block_order:
        parents = structure.dependencies.get(block, [])
        base_tractable.append({
            "targets": frozenset([block]),
            "conditions": frozenset(parents)
        })
    return base_tractable

def get_all_tractable_distributions(structure: Structure) -> Iterator[Dict[str, frozenset[str]]]:
    """
    Returns a generator yielding dictionaries describing all tractable distributions.
    This includes prefix joints, standard triangular map conditionals,
    and minimal local conditionals for all contiguous slices.
    
    A query is tractable if it satisfies the Parent Rule and Descendant Rule.
    
    Args:
        structure: The dependency structure.
        
    Yields:
        Dictionaries with 'targets' and 'conditions' keys.
    """
    seen = set()
    nodes = structure.block_order

    # Iterate over all contiguous slices [i:j]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes) + 1):
            targets = nodes[i:j]
            targets_set = frozenset(targets)
            
            if i == 0:
                # Prefix Joints: p(x0, ..., xj-1)
                key = (targets_set, frozenset())
                if key not in seen:
                    yield {"targets": targets_set, "conditions": frozenset()}
                    seen.add(key)
            else:
                # 1. Prefix Conditionals: p(xi, ..., xj-1 | x0, ..., x{i-1})
                prefix_conditions = frozenset(nodes[:i])
                key = (targets_set, prefix_conditions)
                if key not in seen:
                    yield {"targets": targets_set, "conditions": prefix_conditions}
                    seen.add(key)
                
                # 2. Minimal Local Conditionals: p(xi, ..., xj-1 | parents(targets) \ targets)
                all_parents = set()
                for t in targets:
                    all_parents.update(structure.dependencies.get(t, []))
                
                minimal_conditions_set = frozenset(all_parents - set(targets))
                
                if structure.is_tractable_query(set(targets), set(minimal_conditions_set)):
                    key = (targets_set, minimal_conditions_set)
                    if key not in seen:
                        yield {"targets": targets_set, "conditions": minimal_conditions_set}
                        seen.add(key)

def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
    """
    Computes a stable topological sort of a directed acyclic graph.
    
    The sort is stable and prioritizes nodes with lower original in-degree and
    alphabetical order for tie-breaking. This ensures a consistent ordering 
    between different components (e.g., PGMCompiler and CAREFL).
    
    Args:
        dependencies: Dictionary mapping each node to its list of parent nodes.
        
    Returns:
        List of node names in topological order.
        
    Raises:
        ValueError: If the graph contains a cycle or refers to unknown nodes.
    """
    # 1. Initialize data structures
    nodes = sorted(list(dependencies.keys()))
    in_degree = {u: len(parents) for u, parents in dependencies.items()}
    
    # Adjacency list (children)
    children = {u: [] for u in nodes}
    for u, parents in dependencies.items():
        for p in parents:
            if p not in children:
                raise ValueError(f"Parent '{p}' of node '{u}' not found in dependency keys.")
            children[p].append(u)
            
    # 2. Priority for stable sort (original in-degree, then name)
    priority = {u: len(parents) for u, parents in dependencies.items()}
    
    # 3. Kahn's algorithm
    # Initialize queue with root nodes (in-degree 0)
    queue = [u for u in nodes if in_degree[u] == 0]
    queue.sort(key=lambda n: (priority[n], n))
    
    result = []
    while queue:
        u = queue.pop(0)
        result.append(u)
        
        # Sort children to maintain stability
        for v in sorted(children[u]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                # Re-sort to maintain priority
                queue.sort(key=lambda n: (priority[n], n))
                
    if len(result) != len(nodes):
        missing = set(nodes) - set(result)
        raise ValueError(f"Graph contains a cycle or missing nodes: {missing}")
        
    return result
