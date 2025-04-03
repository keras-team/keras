"""Support for random optimizers, including the random-greedy path."""

import functools
import heapq
import math
import time
from collections import deque
from decimal import Decimal
from random import choices as random_choices
from random import seed as random_seed
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from opt_einsum import helpers, paths
from opt_einsum.typing import ArrayIndexType, ArrayType, PathType

__all__ = ["RandomGreedy", "random_greedy", "random_greedy_128"]


class RandomOptimizer(paths.PathOptimizer):
    """Base class for running any random path finder that benefits
    from repeated calling, possibly in a parallel fashion. Custom random
    optimizers should subclass this, and the `setup` method should be
    implemented with the following signature:

    ```python
    def setup(self, inputs, output, size_dict):
        # custom preparation here ...
        return trial_fn, trial_args
    ```

    Where `trial_fn` itself should have the signature::

    ```python
    def trial_fn(r, *trial_args):
        # custom computation of path here
        return ssa_path, cost, size
    ```

    Where `r` is the run number and could for example be used to seed a
    random number generator. See `RandomGreedy` for an example.


    Parameters:
        max_repeats: The maximum number of repeat trials to have.
        max_time: The maximum amount of time to run the algorithm for.
        minimize:  Whether to favour paths that minimize the total estimated flop-count or
            the size of the largest intermediate created.
        parallel: Whether to parallelize the random trials, by default `False`. If
            `True`, use a `concurrent.futures.ProcessPoolExecutor` with the same
            number of processes as cores. If an integer is specified, use that many
            processes instead. Finally, you can supply a custom executor-pool which
            should have an API matching that of the python 3 standard library
            module `concurrent.futures`. Namely, a `submit` method that returns
            `Future` objects, themselves with `result` and `cancel` methods.
        pre_dispatch: If running in parallel, how many jobs to pre-dispatch so as to avoid
            submitting all jobs at once. Should also be more than twice the number
            of workers to avoid under-subscription. Default: 128.

    Attributes:
        path: The best path found so far.
        costs: The list of each trial's costs found so far.
        sizes: The list of each trial's largest intermediate size so far.
    """

    def __init__(
        self,
        max_repeats: int = 32,
        max_time: Optional[float] = None,
        minimize: str = "flops",
        parallel: Union[bool, Decimal, int] = False,
        pre_dispatch: int = 128,
    ):
        if minimize not in ("flops", "size"):
            raise ValueError("`minimize` should be one of {'flops', 'size'}.")

        self.max_repeats = max_repeats
        self.max_time = max_time
        self.minimize = minimize
        self.better = paths.get_better_fn(minimize)
        self._parallel: Union[bool, Decimal, int] = False
        self.parallel = parallel
        self.pre_dispatch = pre_dispatch

        self.costs: List[int] = []
        self.sizes: List[int] = []
        self.best: Dict[str, Any] = {"flops": float("inf"), "size": float("inf")}

        self._repeats_start = 0
        self._executor: Any
        self._futures: Any

    @property
    def path(self) -> PathType:
        """The best path found so far."""
        return paths.ssa_to_linear(self.best["ssa_path"])

    @property
    def parallel(self) -> Union[bool, Decimal, int]:
        return self._parallel

    @parallel.setter
    def parallel(self, parallel: Union[bool, Decimal, int]) -> None:
        # shutdown any previous executor if we are managing it
        if getattr(self, "_managing_executor", False):
            self._executor.shutdown()

        self._parallel = parallel
        self._managing_executor = False

        if parallel is False:
            self._executor = None
            return

        if parallel is True:
            from concurrent.futures import ProcessPoolExecutor

            self._executor = ProcessPoolExecutor()
            self._managing_executor = True
            return

        if isinstance(parallel, (int, Decimal)):
            from concurrent.futures import ProcessPoolExecutor

            self._executor = ProcessPoolExecutor(int(parallel))
            self._managing_executor = True
            return

        # assume a pool-executor has been supplied
        self._executor = parallel

    def _gen_results_parallel(self, repeats: Iterable[int], trial_fn: Any, args: Any) -> Generator[Any, None, None]:
        """Lazily generate results from an executor without submitting all jobs at once."""
        self._futures = deque()

        # the idea here is to submit at least ``pre_dispatch`` jobs *before* we
        # yield any results, then do both in tandem, before draining the queue
        for r in repeats:
            if len(self._futures) < self.pre_dispatch:
                self._futures.append(self._executor.submit(trial_fn, r, *args))
                continue
            yield self._futures.popleft().result()

        while self._futures:
            yield self._futures.popleft().result()

    def _cancel_futures(self) -> None:
        if self._executor is not None:
            for f in self._futures:
                f.cancel()

    def setup(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def __call__(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
        memory_limit: Optional[int] = None,
    ) -> PathType:
        self._check_args_against_first_call(inputs, output, size_dict)

        # start a timer?
        if self.max_time is not None:
            t0 = time.time()

        trial_fn, trial_args = self.setup(inputs, output, size_dict)

        r_start = self._repeats_start + len(self.costs)
        r_stop = r_start + self.max_repeats
        repeats = range(r_start, r_stop)

        # create the trials lazily
        if self._executor is not None:
            trials = self._gen_results_parallel(repeats, trial_fn, trial_args)
        else:
            trials = (trial_fn(r, *trial_args) for r in repeats)

        # assess the trials
        for ssa_path, cost, size in trials:
            # keep track of all costs and sizes
            self.costs.append(cost)
            self.sizes.append(size)

            # check if we have found a new best
            found_new_best = self.better(cost, size, self.best["flops"], self.best["size"])

            if found_new_best:
                self.best["flops"] = cost
                self.best["size"] = size
                self.best["ssa_path"] = ssa_path

            # check if we have run out of time
            if (self.max_time is not None) and (time.time() > t0 + self.max_time):
                break

        self._cancel_futures()
        return self.path

    def __del__(self):
        # if we created the parallel pool-executor, shut it down
        if getattr(self, "_managing_executor", False):
            self._executor.shutdown()


def thermal_chooser(queue, remaining, nbranch=8, temperature=1, rel_temperature=True):
    """A contraction 'chooser' that weights possible contractions using a
    Boltzmann distribution. Explicitly, given costs `c_i` (with `c_0` the
    smallest), the relative weights, `w_i`, are computed as:

        $$w_i = exp( -(c_i - c_0) / temperature)$$

    Additionally, if `rel_temperature` is set, scale `temperature` by
    `abs(c_0)` to account for likely fluctuating cost magnitudes during the
    course of a contraction.

    Parameters:
        queue: The heapified list of candidate contractions.
        remaining: Mapping of remaining inputs' indices to the ssa id.
        temperature: When choosing a possible contraction, its relative probability will be
            proportional to `exp(-cost / temperature)`. Thus the larger
            `temperature` is, the further random paths will stray from the normal
            'greedy' path. Conversely, if set to zero, only paths with exactly the
            same cost as the best at each step will be explored.
        rel_temperature: Whether to normalize the `temperature` at each step to the scale of
            the best cost. This is generally beneficial as the magnitude of costs
            can vary significantly throughout a contraction.
        nbranch: How many potential paths to calculate probability for and choose from at each step.

    Returns:
        cost
        k1
        k2
        k3
    """
    n = 0
    choices = []
    while queue and n < nbranch:
        cost, k1, k2, k12 = heapq.heappop(queue)
        if k1 not in remaining or k2 not in remaining:
            continue  # candidate is obsolete
        choices.append((cost, k1, k2, k12))
        n += 1

    if n == 0:
        return None
    if n == 1:
        return choices[0]

    costs = [choice[0][0] for choice in choices]
    cmin = costs[0]

    # adjust by the overall scale to account for fluctuating absolute costs
    if rel_temperature:
        temperature *= max(1, abs(cmin))

    # compute relative probability for each potential contraction
    if temperature == 0.0:
        energies = [1 if c == cmin else 0 for c in costs]
    else:
        # shift by cmin for numerical reasons
        energies = [math.exp(-(c - cmin) / temperature) for c in costs]

    # randomly choose a contraction based on energies
    (chosen,) = random_choices(range(n), weights=energies)
    cost, k1, k2, k12 = choices.pop(chosen)

    # put the other choice back in the heap
    for other in choices:
        heapq.heappush(queue, other)

    return cost, k1, k2, k12


def ssa_path_compute_cost(
    ssa_path: PathType,
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
) -> Tuple[int, int]:
    """Compute the flops and max size of an ssa path."""
    inputs = list(map(frozenset, inputs))
    output = frozenset(output)
    remaining = set(range(len(inputs)))
    total_cost = 0
    max_size = 0

    for i, j in ssa_path:
        k12, flops12 = paths.calc_k12_flops(inputs, output, remaining, i, j, size_dict)  # type: ignore
        remaining.discard(i)
        remaining.discard(j)
        remaining.add(len(inputs))
        inputs.append(k12)
        total_cost += flops12
        max_size = max(max_size, helpers.compute_size_by_dict(k12, size_dict))

    return total_cost, max_size


def _trial_greedy_ssa_path_and_cost(
    r: int,
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    size_dict: Dict[str, int],
    choose_fn: Any,
    cost_fn: Any,
) -> Tuple[PathType, int, int]:
    """A single, repeatable, greedy trial run. **Returns:** ``ssa_path`` and cost."""
    if r == 0:
        # always start with the standard greedy approach
        choose_fn = None

    random_seed(r)

    ssa_path = paths.ssa_greedy_optimize(inputs, output, size_dict, choose_fn, cost_fn)
    cost, size = ssa_path_compute_cost(ssa_path, inputs, output, size_dict)

    return ssa_path, cost, size


class RandomGreedy(RandomOptimizer):
    def __init__(
        self,
        cost_fn: str = "memory-removed-jitter",
        temperature: float = 1.0,
        rel_temperature: bool = True,
        nbranch: int = 8,
        **kwargs: Any,
    ):
        """Parameters:
        cost_fn: A function that returns a heuristic 'cost' of a potential contraction
                with which to sort candidates. Should have signature
                `cost_fn(size12, size1, size2, k12, k1, k2)`.
        temperature: When choosing a possible contraction, its relative probability will be
                proportional to `exp(-cost / temperature)`. Thus the larger
                `temperature` is, the further random paths will stray from the normal
                'greedy' path. Conversely, if set to zero, only paths with exactly the
                same cost as the best at each step will be explored.
        rel_temperature: Whether to normalize the ``temperature`` at each step to the scale of
                the best cost. This is generally beneficial as the magnitude of costs
                can vary significantly throughout a contraction. If False, the
                algorithm will end up branching when the absolute cost is low, but
                stick to the 'greedy' path when the cost is high - this can also be
                beneficial.
        nbranch: How many potential paths to calculate probability for and choose from at each step.
        kwargs: Supplied to RandomOptimizer.
        """
        self.cost_fn = cost_fn
        self.temperature = temperature
        self.rel_temperature = rel_temperature
        self.nbranch = nbranch
        super().__init__(**kwargs)

    @property
    def choose_fn(self) -> Any:
        """The function that chooses which contraction to take - make this a
        property so that ``temperature`` and ``nbranch`` etc. can be updated
        between runs.
        """
        if self.nbranch == 1:
            return None

        return functools.partial(
            thermal_chooser,
            temperature=self.temperature,
            nbranch=self.nbranch,
            rel_temperature=self.rel_temperature,
        )

    def setup(
        self,
        inputs: List[ArrayIndexType],
        output: ArrayIndexType,
        size_dict: Dict[str, int],
    ) -> Tuple[Any, Any]:
        fn = _trial_greedy_ssa_path_and_cost
        args = (inputs, output, size_dict, self.choose_fn, self.cost_fn)
        return fn, args


def random_greedy(
    inputs: List[ArrayIndexType],
    output: ArrayIndexType,
    idx_dict: Dict[str, int],
    memory_limit: Optional[int] = None,
    **optimizer_kwargs: Any,
) -> ArrayType:
    """A simple wrapper around the `RandomGreedy` optimizer."""
    optimizer = RandomGreedy(**optimizer_kwargs)
    return optimizer(inputs, output, idx_dict, memory_limit)


random_greedy_128 = functools.partial(random_greedy, max_repeats=128)
