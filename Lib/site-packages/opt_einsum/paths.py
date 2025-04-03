"""Contains the path technology behind opt_einsum in addition to several path helpers."""

import bisect
import functools
import heapq
import itertools
import operator
import random
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, FrozenSet, Generator, List, Optional, Sequence, Set, Tuple, Union
from typing import Counter as CounterType

from opt_einsum.helpers import compute_size_by_dict, flop_count
from opt_einsum.typing import ArrayIndexType, PathSearchFunctionType, PathType, TensorShapeType

__all__ = [
    "optimal",
    "BranchBound",
    "branch",
    "greedy",
    "auto",
    "auto_hq",
    "get_path_fn",
    "DynamicProgramming",
    "dynamic_programming",
]

_UNLIMITED_MEM = {-1, None, float("inf")}


class PathOptimizer:
    r"""Base class for different path optimizers to inherit from.

    Subclassed optimizers should define a call method with signature:

    ```python
    def __call__(self, inputs: List[ArrayIndexType], output: ArrayIndexType, size_dict: dict[str, int], memory_limit: int | None = None) -> list[tuple[int, ...]]:
        \"\"\"
        Parameters:
            inputs: The indices of each input array.
            outputs: The output indices
            size_dict: The size of each index
            memory_limit: If given, the maximum allowed memory.
        \"\"\"
        # ... compute path here ...
        return path
    ```

    where `path` is a list of int-tuples specifying a contraction order.
    """

    def _check_args_against_first_call(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
    ) -> None:
        """Utility that stateful optimizers can use to ensure they are not
        called with different contractions across separate runs.
        """
        args = (inputs, output, size_dict)
        if not hasattr(self, "_first_call_args"):
            # simply set the attribute as currently there is no global PathOptimizer init
            self._first_call_args = args
        elif args != self._first_call_args:
            raise ValueError(
                "The arguments specifying the contraction that this path optimizer "
                "instance was called with have changed - try creating a new instance."
            )

    def __call__(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
        memory_limit: Optional[int] = None,
    ) -> PathType:
        raise NotImplementedError


def ssa_to_linear(ssa_path: PathType) -> PathType:
    """Convert a path with static single assignment ids to a path with recycled
    linear ids.

    Example:
        ```python
        ssa_to_linear([(0, 3), (2, 4), (1, 5)])
        #> [(0, 3), (1, 2), (0, 1)]
        ```
    """
    # ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)  # type: ignore
    # path = []
    # for ssa_ids in ssa_path:
    #     path.append(tuple(int(ids[ssa_id]) for ssa_id in ssa_ids))
    #     for ssa_id in ssa_ids:
    #         ids[ssa_id:] -= 1
    # return path

    n = sum(map(len, ssa_path)) - len(ssa_path) + 1
    ids = list(range(n))
    path = []
    ssa = n
    for scon in ssa_path:
        con = sorted([bisect.bisect_left(ids, s) for s in scon])
        for j in reversed(con):
            ids.pop(j)
        ids.append(ssa)
        path.append(con)
        ssa += 1
    return [tuple(x) for x in path]

    # N = sum(map(len, ssa_path)) - len(ssa_path) + 1
    # ids = list(range(N))
    # ids = np.arange(1 + max(map(max, ssa_path)), dtype=np.int32)
    # path = []
    # ssa = N
    # for scon in ssa_path:
    #     con = sorted(map(ids.index, scon))
    #     for j in reversed(con):
    #         ids.pop(j)
    #     ids.append(ssa)
    #     path.append(con)
    #     ssa += 1
    # return path


def linear_to_ssa(path: PathType) -> PathType:
    """Convert a path with recycled linear ids to a path with static single
    assignment ids.

    Exmaple:
        ```python
        linear_to_ssa([(0, 3), (1, 2), (0, 1)])
        #> [(0, 3), (2, 4), (1, 5)]
        ```
    """
    num_inputs = sum(map(len, path)) - len(path) + 1
    linear_to_ssa = list(range(num_inputs))
    new_ids = itertools.count(num_inputs)
    ssa_path = []
    for ids in path:
        ssa_path.append(tuple(linear_to_ssa[id_] for id_ in ids))
        for id_ in sorted(ids, reverse=True):
            del linear_to_ssa[id_]
        linear_to_ssa.append(next(new_ids))
    return ssa_path


def calc_k12_flops(
    inputs: Tuple[FrozenSet[str]],
    output: FrozenSet[str],
    remaining: FrozenSet[int],
    i: int,
    j: int,
    size_dict: Dict[str, int],
) -> Tuple[FrozenSet[str], int]:
    """Calculate the resulting indices and flops for a potential pairwise
    contraction - used in the recursive (optimal/branch) algorithms.

    Parameters:
        inputs: The indices of each tensor in this contraction, note this includes
            tensors unavailable to contract as static single assignment is used:>
            contracted tensors are not removed from the list.
        output: The set of output indices for the whole contraction.
        remaining: *The set of indices (corresponding to ``inputs``) of tensors still available to contract.
        i: Index of potential tensor to contract.
        j: Index of potential tensor to contract.
        size_dict: Size mapping of all the indices.

    Returns:
        k12: The resulting indices of the potential tensor.
        cost: Estimated flop count of operation.
    """
    k1, k2 = inputs[i], inputs[j]
    either = k1 | k2
    shared = k1 & k2
    keep = frozenset.union(output, *map(inputs.__getitem__, remaining - {i, j}))

    k12 = either & keep
    cost = flop_count(either, bool(shared - keep), 2, size_dict)

    return k12, cost


def _compute_oversize_flops(
    inputs: Tuple[FrozenSet[str]],
    remaining: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
) -> int:
    """Compute the flop count for a contraction of all remaining arguments. This
    is used when a memory limit means that no pairwise contractions can be made.
    """
    idx_contraction = frozenset.union(*map(inputs.__getitem__, remaining))  # type: ignore
    inner = idx_contraction - output
    num_terms = len(remaining)
    return flop_count(idx_contraction, bool(inner), num_terms, size_dict)


def optimal(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
) -> PathType:
    """Computes all possible pair contractions in a depth-first recursive manner,
    sieving results based on `memory_limit` and the best path found so far.

    Parameters:
        inputs: List of sets that represent the lhs side of the einsum subscript.
        output: Set that represents the rhs side of the overall einsum subscript.
        size_dict: Dictionary of index sizes.
        memory_limit: The maximum number of elements in a temporary array.

    Returns:
        path: The optimal contraction order within the memory limit constraint.

    Examples:
    ```python
    isets = [set('abd'), set('ac'), set('bdc')]
    oset = set('')
    idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    optimal(isets, oset, idx_sizes, 5000)
    #> [(0, 2), (0, 1)]
    ```
    """
    inputs_set = tuple(map(frozenset, inputs))
    output_set = frozenset(output)

    best_flops = {"flops": float("inf")}
    best_ssa_path = {"ssa_path": (tuple(range(len(inputs))),)}
    size_cache: Dict[FrozenSet[str], int] = {}
    result_cache: Dict[Tuple[ArrayIndexType, ArrayIndexType], Tuple[FrozenSet[str], int]] = {}

    def _optimal_iterate(path, remaining, inputs, flops):
        # reached end of path (only ever get here if flops is best found so far)
        if len(remaining) == 1:
            best_flops["flops"] = flops
            best_ssa_path["ssa_path"] = path
            return

        # check all possible remaining paths
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i
            key = (inputs[i], inputs[j])
            try:
                k12, flops12 = result_cache[key]
            except KeyError:
                k12, flops12 = result_cache[key] = calc_k12_flops(inputs, output_set, remaining, i, j, size_dict)

            # sieve based on current best flops
            new_flops = flops + flops12
            if new_flops >= best_flops["flops"]:
                continue

            # sieve based on memory limit
            if memory_limit not in _UNLIMITED_MEM:
                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = compute_size_by_dict(k12, size_dict)

                # possibly terminate this path with an all-terms einsum
                if size12 > memory_limit:
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output_set, size_dict)
                    if new_flops < best_flops["flops"]:
                        best_flops["flops"] = new_flops
                        best_ssa_path["ssa_path"] = path + (tuple(remaining),)
                    continue

            # add contraction and recurse into all remaining
            _optimal_iterate(
                path=path + ((i, j),),
                inputs=inputs + (k12,),
                remaining=remaining - {i, j} | {len(inputs)},
                flops=new_flops,
            )

    _optimal_iterate(path=(), inputs=inputs_set, remaining=set(range(len(inputs))), flops=0)

    return ssa_to_linear(best_ssa_path["ssa_path"])


# functions for comparing which of two paths is 'better'


def better_flops_first(flops: int, size: int, best_flops: int, best_size: int) -> bool:
    return (flops, size) < (best_flops, best_size)


def better_size_first(flops: int, size: int, best_flops: int, best_size: int) -> bool:
    return (size, flops) < (best_size, best_flops)


_BETTER_FNS = {
    "flops": better_flops_first,
    "size": better_size_first,
}


def get_better_fn(key: str) -> Callable[[int, int, int, int], bool]:
    return _BETTER_FNS[key]


# functions for assigning a heuristic 'cost' to a potential contraction


def cost_memory_removed(size12: int, size1: int, size2: int, k12: int, k1: int, k2: int) -> float:
    """The default heuristic cost, corresponding to the total reduction in
    memory of performing a contraction.
    """
    return size12 - size1 - size2


def cost_memory_removed_jitter(size12: int, size1: int, size2: int, k12: int, k1: int, k2: int) -> float:
    """Like memory-removed, but with a slight amount of noise that breaks ties
    and thus jumbles the contractions a bit.
    """
    return random.gauss(1.0, 0.01) * (size12 - size1 - size2)


_COST_FNS = {
    "memory-removed": cost_memory_removed,
    "memory-removed-jitter": cost_memory_removed_jitter,
}


class BranchBound(PathOptimizer):
    def __init__(
        self,
        nbranch: Optional[int] = None,
        cutoff_flops_factor: int = 4,
        minimize: str = "flops",
        cost_fn: str = "memory-removed",
    ):
        """Explores possible pair contractions in a depth-first recursive manner like
        the `optimal` approach, but with extra heuristic early pruning of branches
        as well sieving by `memory_limit` and the best path found so far.


        Parameters:
            nbranch: How many branches to explore at each contraction step. If None, explore
                all possible branches. If an integer, branch into this many paths at
                each step. Defaults to None.
            cutoff_flops_factor: If at any point, a path is doing this much worse than the best path
                found so far was, terminate it. The larger this is made, the more paths
                will be fully explored and the slower the algorithm. Defaults to 4.
            minimize: Whether to optimize the path with regard primarily to the total
                estimated flop-count, or the size of the largest intermediate. The
                option not chosen will still be used as a secondary criterion.
            cost_fn: A function that returns a heuristic 'cost' of a potential contraction
                with which to sort candidates. Should have signature
                `cost_fn(size12, size1, size2, k12, k1, k2)`.
        """
        if (nbranch is not None) and nbranch < 1:
            raise ValueError(f"The number of branches must be at least one, `nbranch={nbranch}`.")

        self.nbranch = nbranch
        self.cutoff_flops_factor = cutoff_flops_factor
        self.minimize = minimize
        self.cost_fn: Any = _COST_FNS.get(cost_fn, cost_fn)

        self.better = get_better_fn(minimize)
        self.best: Dict[str, Any] = {"flops": float("inf"), "size": float("inf")}
        self.best_progress: Dict[int, float] = defaultdict(lambda: float("inf"))

    @property
    def path(self) -> PathType:
        return ssa_to_linear(self.best["ssa_path"])

    def __call__(
        self,
        inputs_: List[ArrayIndexType],
        output_: ArrayIndexType,
        size_dict: Dict[str, int],
        memory_limit: Optional[int] = None,
    ) -> PathType:
        """Parameters:
            inputs_: List of sets that represent the lhs side of the einsum subscript
            output_: Set that represents the rhs side of the overall einsum subscript
            size_dict: Dictionary of index sizes
            memory_limit: The maximum number of elements in a temporary array.

        Returns:
            path: The contraction order within the memory limit constraint.

        Examples:
        ```python
        isets = [set('abd'), set('ac'), set('bdc')]
        oset = set('')
        idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        optimal(isets, oset, idx_sizes, 5000)
        #> [(0, 2), (0, 1)]
        """
        self._check_args_against_first_call(inputs_, output_, size_dict)

        inputs: Tuple[FrozenSet[str]] = tuple(map(frozenset, inputs_))  # type: ignore
        output: FrozenSet[str] = frozenset(output_)

        size_cache = {k: compute_size_by_dict(k, size_dict) for k in inputs}
        result_cache: Dict[Tuple[FrozenSet[str], FrozenSet[str]], Tuple[FrozenSet[str], int]] = {}

        def _branch_iterate(path, inputs, remaining, flops, size):
            # reached end of path (only ever get here if flops is best found so far)
            if len(remaining) == 1:
                self.best["size"] = size
                self.best["flops"] = flops
                self.best["ssa_path"] = path
                return

            def _assess_candidate(k1: FrozenSet[str], k2: FrozenSet[str], i: int, j: int) -> Any:
                # find resulting indices and flops
                try:
                    k12, flops12 = result_cache[k1, k2]
                except KeyError:
                    k12, flops12 = result_cache[k1, k2] = calc_k12_flops(inputs, output, remaining, i, j, size_dict)

                try:
                    size12 = size_cache[k12]
                except KeyError:
                    size12 = size_cache[k12] = compute_size_by_dict(k12, size_dict)

                new_flops = flops + flops12
                new_size = max(size, size12)

                # sieve based on current best i.e. check flops and size still better
                if not self.better(new_flops, new_size, self.best["flops"], self.best["size"]):
                    return None

                # compare to how the best method was doing as this point
                if new_flops < self.best_progress[len(inputs)]:
                    self.best_progress[len(inputs)] = new_flops
                # sieve based on current progress relative to best
                elif new_flops > self.cutoff_flops_factor * self.best_progress[len(inputs)]:
                    return None

                # sieve based on memory limit
                if (memory_limit not in _UNLIMITED_MEM) and (size12 > memory_limit):  # type: ignore
                    # terminate path here, but check all-terms contract first
                    new_flops = flops + _compute_oversize_flops(inputs, remaining, output_, size_dict)
                    if new_flops < self.best["flops"]:
                        self.best["flops"] = new_flops
                        self.best["ssa_path"] = path + (tuple(remaining),)
                    return None

                # set cost heuristic in order to locally sort possible contractions
                size1, size2 = size_cache[inputs[i]], size_cache[inputs[j]]
                cost = self.cost_fn(size12, size1, size2, k12, k1, k2)

                return cost, flops12, new_flops, new_size, (i, j), k12

            # check all possible remaining paths
            candidates = []
            for i, j in itertools.combinations(remaining, 2):
                if i > j:
                    i, j = j, i
                k1, k2 = inputs[i], inputs[j]

                # initially ignore outer products
                if k1.isdisjoint(k2):
                    continue

                candidate = _assess_candidate(k1, k2, i, j)
                if candidate:
                    heapq.heappush(candidates, candidate)

            # assess outer products if nothing left
            if not candidates:
                for i, j in itertools.combinations(remaining, 2):
                    if i > j:
                        i, j = j, i
                    k1, k2 = inputs[i], inputs[j]
                    candidate = _assess_candidate(k1, k2, i, j)
                    if candidate:
                        heapq.heappush(candidates, candidate)

            # recurse into all or some of the best candidate contractions
            bi = 0
            while (self.nbranch is None or bi < self.nbranch) and candidates:
                _, _, new_flops, new_size, (i, j), k12 = heapq.heappop(candidates)
                _branch_iterate(
                    path=path + ((i, j),),
                    inputs=inputs + (k12,),
                    remaining=(remaining - {i, j}) | {len(inputs)},
                    flops=new_flops,
                    size=new_size,
                )
                bi += 1

        _branch_iterate(path=(), inputs=inputs, remaining=set(range(len(inputs))), flops=0, size=0)

        return self.path


def branch(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    nbranch: Optional[int] = None,
    cutoff_flops_factor: int = 4,
    minimize: str = "flops",
    cost_fn: str = "memory-removed",
) -> PathType:
    optimizer = BranchBound(
        nbranch=nbranch, cutoff_flops_factor=cutoff_flops_factor, minimize=minimize, cost_fn=cost_fn
    )
    return optimizer(inputs, output, size_dict, memory_limit)


branch_all = functools.partial(branch, nbranch=None)
branch_2 = functools.partial(branch, nbranch=2)
branch_1 = functools.partial(branch, nbranch=1)

GreedyCostType = Tuple[int, int, int]
GreedyContractionType = Tuple[GreedyCostType, ArrayIndexType, ArrayIndexType, ArrayIndexType]  # Cost, t1,t2->t3


def _get_candidate(
    output: ArrayIndexType,
    sizes: Dict[str, int],
    remaining: Dict[ArrayIndexType, int],
    footprints: Dict[ArrayIndexType, int],
    dim_ref_counts: Dict[int, Set[str]],
    k1: ArrayIndexType,
    k2: ArrayIndexType,
    cost_fn: Any,
) -> GreedyContractionType:
    either = k1 | k2
    two = k1 & k2
    one = either - two
    k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    cost = cost_fn(
        compute_size_by_dict(k12, sizes),
        footprints[k1],
        footprints[k2],
        k12,
        k1,
        k2,
    )
    id1 = remaining[k1]
    id2 = remaining[k2]
    if id1 > id2:
        k1, id1, k2, id2 = k2, id2, k1, id1
    cost = cost, id2, id1  # break ties to ensure determinism
    return cost, k1, k2, k12


def _push_candidate(
    output: ArrayIndexType,
    sizes: Dict[str, Any],
    remaining: Dict[ArrayIndexType, int],
    footprints: Dict[ArrayIndexType, int],
    dim_ref_counts: Dict[int, Set[str]],
    k1: ArrayIndexType,
    k2s: List[ArrayIndexType],
    queue: List[GreedyContractionType],
    push_all: bool,
    cost_fn: Any,
) -> None:
    candidates = (_get_candidate(output, sizes, remaining, footprints, dim_ref_counts, k1, k2, cost_fn) for k2 in k2s)
    if push_all:
        # want to do this if we e.g. are using a custom 'choose_fn'
        for candidate in candidates:
            heapq.heappush(queue, candidate)
    else:
        heapq.heappush(queue, min(candidates))


def _update_ref_counts(
    dim_to_keys: Dict[str, Set[ArrayIndexType]],
    dim_ref_counts: Dict[int, Set[str]],
    dims: ArrayIndexType,
) -> None:
    for dim in dims:
        count = len(dim_to_keys[dim])
        if count <= 1:
            dim_ref_counts[2].discard(dim)
            dim_ref_counts[3].discard(dim)
        elif count == 2:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].discard(dim)
        else:
            dim_ref_counts[2].add(dim)
            dim_ref_counts[3].add(dim)


def _simple_chooser(queue, remaining):
    """Default contraction chooser that simply takes the minimum cost option."""
    cost, k1, k2, k12 = heapq.heappop(queue)
    if k1 not in remaining or k2 not in remaining:
        return None  # candidate is obsolete
    return cost, k1, k2, k12


def ssa_greedy_optimize(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    sizes: Dict[str, int],
    choose_fn: Any = None,
    cost_fn: Any = "memory-removed",
) -> PathType:
    """This is the core function for :func:`greedy` but produces a path with
    static single assignment ids rather than recycled linear ids.
    SSA ids are cheaper to work with and easier to reason about.
    """
    if len(inputs) == 1:
        # Perform a single contraction to match output shape.
        return [(0,)]

    # set the function that assigns a heuristic cost to a possible contraction
    cost_fn = _COST_FNS.get(cost_fn, cost_fn)

    # set the function that chooses which contraction to take
    if choose_fn is None:
        choose_fn = _simple_chooser
        push_all = False
    else:
        # assume chooser wants access to all possible contractions
        push_all = True

    # A dim that is common to all tensors might as well be an output dim, since it
    # cannot be contracted until the final step. This avoids an expensive all-pairs
    # comparison to search for possible contractions at each step, leading to speedup
    # in many practical problems where all tensors share a common batch dimension.
    fs_inputs = [frozenset(x) for x in inputs]
    output = frozenset(output) | frozenset.intersection(*fs_inputs)

    # Deduplicate shapes by eagerly computing Hadamard products.
    remaining: Dict[ArrayIndexType, int] = {}  # key -> ssa_id
    ssa_ids = itertools.count(len(fs_inputs))
    ssa_path: List[TensorShapeType] = []
    for ssa_id, key in enumerate(fs_inputs):
        if key in remaining:
            ssa_path.append((remaining[key], ssa_id))
            remaining[key] = next(ssa_ids)
        else:
            remaining[key] = ssa_id

    # Keep track of possible contraction dims.
    dim_to_keys = defaultdict(set)
    for key in remaining:
        for dim in key - output:
            dim_to_keys[dim].add(key)

    # Keep track of the number of tensors using each dim; when the dim is no longer
    # used it can be contracted. Since we specialize to binary ops, we only care about
    # ref counts of >=2 or >=3.
    dim_ref_counts = {
        count: {dim for dim, keys in dim_to_keys.items() if len(keys) >= count} - output for count in [2, 3]
    }

    # Compute separable part of the objective function for contractions.
    footprints = {key: compute_size_by_dict(key, sizes) for key in remaining}

    # Find initial candidate contractions.
    queue: List[GreedyContractionType] = []
    for dim, dim_keys in dim_to_keys.items():
        dim_keys_list = sorted(dim_keys, key=remaining.__getitem__)
        for i, k1 in enumerate(dim_keys_list[:-1]):
            k2s_guess = dim_keys_list[1 + i :]
            _push_candidate(
                output,
                sizes,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                k2s_guess,
                queue,
                push_all,
                cost_fn,
            )

    # Greedily contract pairs of tensors.
    while queue:
        con = choose_fn(queue, remaining)
        if con is None:
            continue  # allow choose_fn to flag all candidates obsolete
        cost, k1, k2, k12 = con

        ssa_id1 = remaining.pop(k1)
        ssa_id2 = remaining.pop(k2)
        for dim in k1 - output:
            dim_to_keys[dim].remove(k1)
        for dim in k2 - output:
            dim_to_keys[dim].remove(k2)
        ssa_path.append((ssa_id1, ssa_id2))
        if k12 in remaining:
            ssa_path.append((remaining[k12], next(ssa_ids)))
        else:
            for dim in k12 - output:
                dim_to_keys[dim].add(k12)
        remaining[k12] = next(ssa_ids)
        _update_ref_counts(dim_to_keys, dim_ref_counts, k1 | k2 - output)
        footprints[k12] = compute_size_by_dict(k12, sizes)

        # Find new candidate contractions.
        k1 = k12
        k2s = {k2 for dim in k1 for k2 in dim_to_keys[dim]}
        k2s.discard(k1)
        if k2s:
            _push_candidate(
                output,
                sizes,
                remaining,
                footprints,
                dim_ref_counts,
                k1,
                list(k2s),
                queue,
                push_all,
                cost_fn,
            )

    # Greedily compute pairwise outer products.
    final_queue = [(compute_size_by_dict(key & output, sizes), ssa_id, key) for key, ssa_id in remaining.items()]
    heapq.heapify(final_queue)
    _, ssa_id1, k1 = heapq.heappop(final_queue)
    while final_queue:
        _, ssa_id2, k2 = heapq.heappop(final_queue)
        ssa_path.append((min(ssa_id1, ssa_id2), max(ssa_id1, ssa_id2)))
        k12 = (k1 | k2) & output
        cost = compute_size_by_dict(k12, sizes)
        ssa_id12 = next(ssa_ids)
        _, ssa_id1, k1 = heapq.heappushpop(final_queue, (cost, ssa_id12, k12))

    return ssa_path


def greedy(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    choose_fn: Any = None,
    cost_fn: str = "memory-removed",
) -> PathType:
    """Finds the path by a three stage algorithm:

    1. Eagerly compute Hadamard products.
    2. Greedily compute contractions to maximize `removed_size`
    3. Greedily compute outer products.

    This algorithm scales quadratically with respect to the
    maximum number of elements sharing a common dim.

    Parameters:
        inputs: List of sets that represent the lhs side of the einsum subscript
        output: Set that represents the rhs side of the overall einsum subscript
        size_dict: Dictionary of index sizes
        memory_limit: The maximum number of elements in a temporary array
        choose_fn: A function that chooses which contraction to perform from the queue
        cost_fn: A function that assigns a potential contraction a cost.

    Returns:
        path: The contraction order (a list of tuples of ints).

    Examples:
        ```python
        isets = [set('abd'), set('ac'), set('bdc')]
        oset = set('')
        idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
        greedy(isets, oset, idx_sizes)
        #> [(0, 2), (0, 1)]
        ```
    """
    if memory_limit not in _UNLIMITED_MEM:
        return branch(inputs, output, size_dict, memory_limit, nbranch=1, cost_fn=cost_fn)  # type: ignore

    ssa_path = ssa_greedy_optimize(inputs, output, size_dict, cost_fn=cost_fn, choose_fn=choose_fn)
    return ssa_to_linear(ssa_path)


def _tree_to_sequence(tree: Tuple[Any, ...]) -> PathType:
    """Converts a contraction tree to a contraction path as it has to be
    returned by path optimizers. A contraction tree can either be an int
    (=no contraction) or a tuple containing the terms to be contracted. An
    arbitrary number (>= 1) of terms can be contracted at once. Note that
    contractions are commutative, e.g. (j, k, l) = (k, l, j). Note that in
    general, solutions are not unique.

    Parameters:
        c: Contraction tree

    Returns:
        path: Contraction path

    Examples:
        ```python
        _tree_to_sequence(((1,2),(0,(4,5,3))))
        #> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
        ```
    """
    # ((1,2),(0,(4,5,3))) --> [(1, 2), (1, 2, 3), (0, 2), (0, 1)]
    #
    # 0     0         0           (1,2)       --> ((1,2),(0,(3,4,5)))
    # 1     3         (1,2)   --> (0,(3,4,5))
    # 2 --> 4     --> (3,4,5)
    # 3     5
    # 4     (1,2)
    # 5
    #
    # this function iterates through the table shown above from right to left;

    if type(tree) == int:  # noqa: E721
        return []

    c: List[Tuple[Any, ...]] = [tree]  # list of remaining contractions (lower part of columns shown above)
    t: List[int] = []  # list of elementary tensors (upper part of columns)
    s: List[Tuple[int, ...]] = []  # resulting contraction sequence

    while len(c) > 0:
        j = c.pop(-1)
        s.insert(0, ())

        for i in sorted([i for i in j if type(i) == int]):  # noqa: E721
            s[0] += (sum(1 for q in t if q < i),)
            t.insert(s[0][-1], i)

        for i_tup in [i_tup for i_tup in j if type(i_tup) != int]:  # noqa: E721
            s[0] += (len(t) + len(c),)
            c.append(i_tup)

    return s


def _find_disconnected_subgraphs(inputs: List[FrozenSet[int]], output: FrozenSet[int]) -> List[FrozenSet[int]]:
    """Finds disconnected subgraphs in the given list of inputs. Inputs are
    connected if they share summation indices. Note: Disconnected subgraphs
    can be contracted independently before forming outer products.

    Parameters:
        inputs: List of sets that represent the lhs side of the einsum subscript
        output: Set that represents the rhs side of the overall einsum subscript

    Returns:
        subgraphs: List containing sets of indices for each subgraph

    Examples:
        ```python
        _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("bd"))
        #> [{0, 2}, {1}]

        _find_disconnected_subgraphs([set("ab"), set("c"), set("ad")], set("abd"))
        #> [{0}, {1}, {2}]
        ```
    """
    subgraphs = []
    unused_inputs = set(range(len(inputs)))

    i_sum = frozenset.union(*inputs) - output  # all summation indices

    while len(unused_inputs) > 0:
        g = set()
        q = [unused_inputs.pop()]
        while len(q) > 0:
            j = q.pop()
            g.add(j)
            i_tmp = i_sum & inputs[j]
            n = {k for k in unused_inputs if len(i_tmp & inputs[k]) > 0}
            q.extend(n)
            unused_inputs.difference_update(n)

        subgraphs.append(g)

    return [frozenset(x) for x in subgraphs]


def _bitmap_select(s: int, seq: List[FrozenSet[int]]) -> Generator[FrozenSet[int], None, None]:
    """Select elements of ``seq`` which are marked by the bitmap set ``s``.

    E.g.:

        >>> list(_bitmap_select(0b11010, ['A', 'B', 'C', 'D', 'E']))
        ['B', 'D', 'E']
    """
    return (x for x, b in zip(seq, bin(s)[:1:-1]) if b == "1")


def _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2):
    """Calculates the effective outer indices of the intermediate tensor
    corresponding to the subgraph ``s``.
    """
    # set of remaining tensors (=g-s)
    r = g & (all_tensors ^ s)
    # indices of remaining indices:
    if r:
        i_r = frozenset.union(*_bitmap_select(r, inputs))
    else:
        i_r = frozenset()
    # contraction indices:
    i_contract = i1_cut_i2_wo_output - i_r
    return i1_union_i2 - i_contract


def _dp_compare_flops(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
) -> None:
    """Performs the inner comparison of whether the two subgraphs (the bitmaps
    `s1` and `s2`) should be merged and added to the dynamic programming
    search. Will skip for a number of reasons:

    1. If the number of operations to form `s = s1 | s2` including previous
       contractions is above the cost-cap.
    2. If we've already found a better way of making `s`.
    3. If the intermediate tensor corresponding to `s` is going to break the
       memory limit.
    """
    # TODO: Odd usage with an Iterable[int] to map a dict of type List[int]
    cost = cost1 + cost2 + compute_size_by_dict(i1_union_i2, size_dict)
    if cost <= cost_cap:
        s = s1 | s2
        if s not in xn or cost < xn[s][1]:
            i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
            mem = compute_size_by_dict(i, size_dict)
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_size(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
) -> None:
    """Like `_dp_compare_flops` but sieves the potential contraction based
    on the size of the intermediate tensor created, rather than the number of
    operations, and so calculates that first.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    cost = max(cost1, cost2, mem)
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


def _dp_compare_write(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
) -> None:
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on the total size of memory created, rather than the number of
    operations, and so calculates that first.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    cost = cost1 + cost2 + mem
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


DEFAULT_COMBO_FACTOR = 64


def _dp_compare_combo(
    cost1: int,
    cost2: int,
    i1_union_i2: Set[int],
    size_dict: List[int],
    cost_cap: int,
    s1: int,
    s2: int,
    xn: Dict[int, Any],
    g: int,
    all_tensors: int,
    inputs: List[FrozenSet[int]],
    i1_cut_i2_wo_output: Set[int],
    memory_limit: Optional[int],
    contract1: Union[int, Tuple[int]],
    contract2: Union[int, Tuple[int]],
    factor: Union[int, float] = DEFAULT_COMBO_FACTOR,
    combine: Callable = sum,
) -> None:
    """Like ``_dp_compare_flops`` but sieves the potential contraction based
    on some combination of both the flops and size,.
    """
    s = s1 | s2
    i = _dp_calc_legs(g, all_tensors, s, inputs, i1_cut_i2_wo_output, i1_union_i2)
    mem = compute_size_by_dict(i, size_dict)
    f = compute_size_by_dict(i1_union_i2, size_dict)
    cost = cost1 + cost2 + combine((f, factor * mem))
    if cost <= cost_cap:
        if s not in xn or cost < xn[s][1]:
            if memory_limit is None or mem <= memory_limit:
                xn[s] = (i, cost, (contract1, contract2))


minimize_finder = re.compile(r"(flops|size|write|combo|limit)-*(\d*)")


@functools.lru_cache(128)
def _parse_minimize(minimize: Union[str, Callable]) -> Tuple[Callable, Union[int, float]]:
    """This works out what local scoring function to use for the dp algorithm
    as well as a `naive_scale` to account for the memory_limit checks.
    """
    if minimize == "flops":
        return _dp_compare_flops, 1
    elif minimize == "size":
        return _dp_compare_size, 1
    elif minimize == "write":
        return _dp_compare_write, 1
    elif callable(minimize):
        # default to naive_scale=inf for this and remaining options
        # as otherwise memory_limit check can cause problems
        return minimize, float("inf")

    # parse out a customized value for the combination factor
    match = minimize_finder.fullmatch(minimize)
    if match is None:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")

    minimize, custom_factor = match.groups()
    factor = float(custom_factor) if custom_factor else DEFAULT_COMBO_FACTOR
    if minimize == "combo":
        return functools.partial(_dp_compare_combo, factor=factor, combine=sum), float("inf")
    elif minimize == "limit":
        return functools.partial(_dp_compare_combo, factor=factor, combine=max), float("inf")
    else:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")


def simple_tree_tuple(seq: Sequence[Tuple[int, ...]]) -> Tuple[Any, ...]:
    """Make a simple left to right binary tree out of iterable `seq`.

    ```python
    tuple_nest([1, 2, 3, 4])
    #> (((1, 2), 3), 4)
    ```

    """
    return functools.reduce(lambda x, y: (x, y), seq)


def _dp_parse_out_single_term_ops(
    inputs: List[FrozenSet[int]], all_inds: Tuple[str, ...], ind_counts: CounterType[str]
) -> Tuple[List[FrozenSet[int]], List[Tuple[int]], List[Union[int, Tuple[int]]]]:
    """Take `inputs` and parse for single term index operations, i.e. where
    an index appears on one tensor and nowhere else.

    If a term is completely reduced to a scalar in this way it can be removed
    to `inputs_done`. If only some indices can be summed then add a 'single
    term contraction' that will perform this summation.
    """
    i_single = frozenset(i for i, c in enumerate(all_inds) if ind_counts[c] == 1)
    inputs_parsed: List[FrozenSet[int]] = []
    inputs_done: List[Tuple[int]] = []
    inputs_contractions: List[Union[int, Tuple[int]]] = []
    for j, i in enumerate(inputs):
        i_reduced = i - i_single
        if (not i_reduced) and (len(i) > 0):
            # input reduced to scalar already - remove
            inputs_done.append((j,))
        else:
            # if the input has any index reductions, add single contraction
            inputs_parsed.append(i_reduced)
            inputs_contractions.append((j,) if i_reduced != i else j)

    return inputs_parsed, inputs_done, inputs_contractions


class DynamicProgramming(PathOptimizer):
    """Finds the optimal path of pairwise contractions without intermediate outer
    products based a dynamic programming approach presented in
    Phys. Rev. E 90, 033315 (2014) (the corresponding preprint is publicly
    available at https://arxiv.org/abs/1304.6112). This method is especially
    well-suited in the area of tensor network states, where it usually
    outperforms all the other optimization strategies.

    This algorithm shows exponential scaling with the number of inputs
    in the worst case scenario (see example below). If the graph to be
    contracted consists of disconnected subgraphs, the algorithm scales
    linearly in the number of disconnected subgraphs and only exponentially
    with the number of inputs per subgraph.

    Parameters:
        minimize: What to minimize:
            - 'flops' - minimize the number of flops
            - 'size' - minimize the size of the largest intermediate
            - 'write' - minimize the size of all intermediate tensors
            - 'combo' - minimize `flops + alpha * write` summed over intermediates, a default ratio of alpha=64
            is used, or it can be customized with `f'combo-{alpha}'`
            - 'limit' - minimize `max(flops, alpha * write)` summed over intermediates, a default ratio of alpha=64
            is used, or it can be customized with `f'limit-{alpha}'`
            - callable - a custom local cost function

        cost_cap: How to implement cost-capping:
            - True - iteratively increase the cost-cap
            - False - implement no cost-cap at all
            - int - use explicit cost cap

        search_outer: In rare circumstances the optimal contraction may involve an outer
            product, this option allows searching such contractions but may well
            slow down the path finding considerably on all but very small graphs.
    """

    def __init__(self, minimize: str = "flops", cost_cap: Union[bool, int] = True, search_outer: bool = False) -> None:
        self.minimize = minimize
        self.search_outer = search_outer
        self.cost_cap = cost_cap

    def __call__(
        self,
        inputs_: List[ArrayIndexType],
        output_: ArrayIndexType,
        size_dict_: Dict[str, int],
        memory_limit_: Optional[int] = None,
    ) -> PathType:
        """Parameters:
            inputs_: List of sets that represent the lhs side of the einsum subscript
            output_: Set that represents the rhs side of the overall einsum subscript
            size_dict_: Dictionary of index sizes
            memory_limit_: The maximum number of elements in a temporary array.

        Returns:
            path: The contraction order (a list of tuples of ints).

        Examples:
            ```python
            n_in = 3  # exponential scaling
            n_out = 2 # linear scaling
            s = dict()
            i_all = []
            for _ in range(n_out):
                i = [set() for _ in range(n_in)]
                for j in range(n_in):
                    for k in range(j+1, n_in):
                        c = oe.get_symbol(len(s))
                        i[j].add(c)
                        i[k].add(c)
                        s[c] = 2
                i_all.extend(i)
            o = DynamicProgramming()
            o(i_all, set(), s)
            #> [(1, 2), (0, 4), (1, 2), (0, 2), (0, 1)]
            ```
        """
        _check_contraction, naive_scale = _parse_minimize(self.minimize)
        _check_outer = (lambda x: True) if self.search_outer else (lambda x: x)

        ind_counts = Counter(itertools.chain(*inputs_, output_))
        all_inds = tuple(ind_counts)

        # convert all indices to integers (makes set operations ~10 % faster)
        symbol2int = {c: j for j, c in enumerate(all_inds)}
        inputs = [frozenset(symbol2int[c] for c in i) for i in inputs_]
        output = frozenset(symbol2int[c] for c in output_)
        size_dict_canonical = {symbol2int[c]: v for c, v in size_dict_.items() if c in symbol2int}
        size_dict = [size_dict_canonical[j] for j in range(len(size_dict_canonical))]
        naive_cost = naive_scale * len(inputs) * functools.reduce(operator.mul, size_dict, 1)

        inputs, inputs_done, inputs_contractions = _dp_parse_out_single_term_ops(inputs, all_inds, ind_counts)

        if not inputs:
            # nothing left to do after single axis reductions!
            return _tree_to_sequence(simple_tree_tuple(inputs_done))

        # a list of all necessary contraction expressions for each of the
        # disconnected subgraphs and their size
        subgraph_contractions = inputs_done
        subgraph_contractions_size = [1] * len(inputs_done)

        if self.search_outer:
            # optimize everything together if we are considering outer products
            subgraphs = [frozenset(range(len(inputs)))]
        else:
            subgraphs = _find_disconnected_subgraphs(inputs, output)

        # the bitmap set of all tensors is computed as it is needed to
        # compute set differences: s1 - s2 transforms into
        # s1 & (all_tensors ^ s2)
        all_tensors = (1 << len(inputs)) - 1

        for g in subgraphs:
            # dynamic programming approach to compute x[n] for subgraph g;
            # x[n][set of n tensors] = (indices, cost, contraction)
            # the set of n tensors is represented by a bitmap: if bit j is 1,
            # tensor j is in the set, e.g. 0b100101 = {0,2,5}; set unions
            # (intersections) can then be computed by bitwise or (and);
            x: List[Any] = [None] * 2 + [{} for j in range(len(g) - 1)]
            x[1] = {1 << j: (inputs[j], 0, inputs_contractions[j]) for j in g}

            # convert set of tensors g to a bitmap set:
            bitmap_g = functools.reduce(lambda x, y: x | y, (1 << j for j in g))

            # try to find contraction with cost <= cost_cap and increase
            # cost_cap successively if no such contraction is found;
            # this is a major performance improvement; start with product of
            # output index dimensions as initial cost_cap
            subgraph_inds = frozenset.union(*_bitmap_select(bitmap_g, inputs))
            if self.cost_cap is True:
                cost_cap = compute_size_by_dict(subgraph_inds & output, size_dict)
            elif self.cost_cap is False:
                cost_cap = float("inf")  # type: ignore
            else:
                cost_cap = self.cost_cap
            # set the factor to increase the cost by each iteration (ensure > 1)
            if len(subgraph_inds) == 0:
                cost_increment = 2
            else:
                cost_increment = max(min(map(size_dict.__getitem__, subgraph_inds)), 2)

            while len(x[-1]) == 0:
                for n in range(2, len(x[1]) + 1):
                    xn = x[n]

                    # try to combine solutions from x[m] and x[n-m]
                    for m in range(1, n // 2 + 1):
                        for s1, (i1, cost1, contract1) in x[m].items():
                            for s2, (i2, cost2, contract2) in x[n - m].items():
                                # can only merge if s1 and s2 are disjoint
                                # and avoid e.g. s1={0}, s2={1} and s1={1}, s2={0}
                                if (not s1 & s2) and (m != n - m or s1 < s2):
                                    i1_cut_i2_wo_output = (i1 & i2) - output

                                    # maybe ignore outer products:
                                    if _check_outer(i1_cut_i2_wo_output):
                                        i1_union_i2 = i1 | i2
                                        _check_contraction(
                                            cost1,
                                            cost2,
                                            i1_union_i2,
                                            size_dict,
                                            cost_cap,
                                            s1,
                                            s2,
                                            xn,
                                            bitmap_g,
                                            all_tensors,
                                            inputs,
                                            i1_cut_i2_wo_output,
                                            memory_limit_,
                                            contract1,
                                            contract2,
                                        )

                if (cost_cap > naive_cost) and (len(x[-1]) == 0):
                    raise RuntimeError("No contraction found for given `memory_limit`.")

                # increase cost cap for next iteration:
                cost_cap = cost_increment * cost_cap

            i, cost, contraction = list(x[-1].values())[0]
            subgraph_contractions.append(contraction)
            subgraph_contractions_size.append(compute_size_by_dict(i, size_dict))

        # sort the subgraph contractions by the size of the subgraphs in
        # ascending order (will give the cheapest contractions); note that
        # outer products should be performed pairwise (to use BLAS functions)
        subgraph_contractions = [
            subgraph_contractions[j]
            for j in sorted(
                range(len(subgraph_contractions_size)),
                key=subgraph_contractions_size.__getitem__,
            )
        ]

        # build the final contraction tree
        tree = simple_tree_tuple(subgraph_contractions)
        return _tree_to_sequence(tree)


def dynamic_programming(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    **kwargs: Any,
) -> PathType:
    optimizer = DynamicProgramming(**kwargs)
    return optimizer(inputs, output, size_dict, memory_limit)


_AUTO_CHOICES = {}
for i in range(1, 5):
    _AUTO_CHOICES[i] = optimal
for i in range(5, 7):
    _AUTO_CHOICES[i] = branch_all
for i in range(7, 9):
    _AUTO_CHOICES[i] = branch_2
for i in range(9, 15):
    _AUTO_CHOICES[i] = branch_1


def auto(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
) -> PathType:
    """Finds the contraction path by automatically choosing the method based on
    how many input arguments there are.
    """
    return _AUTO_CHOICES.get(len(inputs), greedy)(inputs, output, size_dict, memory_limit)


_AUTO_HQ_CHOICES = {}
for i in range(1, 6):
    _AUTO_HQ_CHOICES[i] = optimal
for i in range(6, 17):
    _AUTO_HQ_CHOICES[i] = dynamic_programming


def auto_hq(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
) -> PathType:
    """Finds the contraction path by automatically choosing the method based on
    how many input arguments there are, but targeting a more generous
    amount of search time than ``'auto'``.
    """
    from opt_einsum.path_random import random_greedy_128

    return _AUTO_HQ_CHOICES.get(len(inputs), random_greedy_128)(inputs, output, size_dict, memory_limit)


_PATH_OPTIONS: Dict[str, PathSearchFunctionType] = {
    "auto": auto,
    "auto-hq": auto_hq,
    "optimal": optimal,
    "branch-all": branch_all,
    "branch-2": branch_2,
    "branch-1": branch_1,
    "greedy": greedy,
    "eager": greedy,
    "opportunistic": greedy,
    "dp": dynamic_programming,
    "dynamic-programming": dynamic_programming,
}


def register_path_fn(name: str, fn: PathSearchFunctionType) -> None:
    """Add path finding function ``fn`` as an option with ``name``."""
    if name in _PATH_OPTIONS:
        raise KeyError(f"Path optimizer '{name}' already exists.")

    _PATH_OPTIONS[name.lower()] = fn


def get_path_fn(path_type: str) -> PathSearchFunctionType:
    """Get the correct path finding function from str ``path_type``."""
    path_type = path_type.lower()
    if path_type not in _PATH_OPTIONS:
        raise KeyError(f"Path optimizer '{path_type}' not found, valid options are {set(_PATH_OPTIONS.keys())}.")

    return _PATH_OPTIONS[path_type]
