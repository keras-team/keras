"""
Tests the accuracy of the opt_einsum paths in addition to unit tests for
the various path helper functions.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import pytest

import opt_einsum as oe
from opt_einsum.testing import build_shapes, rand_equation
from opt_einsum.typing import ArrayIndexType, OptimizeKind, PathType, TensorShapeType

explicit_path_tests = {
    "GEMM1": (
        [set("abd"), set("ac"), set("bdc")],
        set(""),
        {"a": 1, "b": 2, "c": 3, "d": 4},
    ),
    "Inner1": (
        [set("abcd"), set("abc"), set("bc")],
        set(""),
        {"a": 5, "b": 2, "c": 3, "d": 4},
    ),
}

# note that these tests have no unique solution due to the chosen dimensions
path_edge_tests = [
    ["greedy", "eb,cb,fb->cef", ((0, 2), (0, 1))],
    ["branch-all", "eb,cb,fb->cef", ((0, 2), (0, 1))],
    ["branch-2", "eb,cb,fb->cef", ((0, 2), (0, 1))],
    ["optimal", "eb,cb,fb->cef", ((0, 2), (0, 1))],
    ["dp", "eb,cb,fb->cef", ((1, 2), (0, 1))],
    ["greedy", "dd,fb,be,cdb->cef", ((0, 3), (0, 1), (0, 1))],
    ["branch-all", "dd,fb,be,cdb->cef", ((0, 3), (0, 1), (0, 1))],
    ["branch-2", "dd,fb,be,cdb->cef", ((0, 3), (0, 1), (0, 1))],
    ["optimal", "dd,fb,be,cdb->cef", ((0, 3), (0, 1), (0, 1))],
    ["optimal", "dd,fb,be,cdb->cef", ((0, 3), (0, 1), (0, 1))],
    ["dp", "dd,fb,be,cdb->cef", ((0, 3), (0, 2), (0, 1))],
    ["greedy", "bca,cdb,dbf,afc->", ((1, 2), (0, 2), (0, 1))],
    ["branch-all", "bca,cdb,dbf,afc->", ((1, 2), (0, 2), (0, 1))],
    ["branch-2", "bca,cdb,dbf,afc->", ((1, 2), (0, 2), (0, 1))],
    ["optimal", "bca,cdb,dbf,afc->", ((1, 2), (0, 2), (0, 1))],
    ["dp", "bca,cdb,dbf,afc->", ((1, 2), (1, 2), (0, 1))],
    ["greedy", "dcc,fce,ea,dbf->ab", ((1, 2), (0, 1), (0, 1))],
    ["branch-all", "dcc,fce,ea,dbf->ab", ((1, 2), (0, 2), (0, 1))],
    ["branch-2", "dcc,fce,ea,dbf->ab", ((1, 2), (0, 2), (0, 1))],
    ["optimal", "dcc,fce,ea,dbf->ab", ((1, 2), (0, 2), (0, 1))],
    ["dp", "dcc,fce,ea,dbf->ab", ((1, 2), (0, 2), (0, 1))],
]

# note that these tests have no unique solution due to the chosen dimensions
path_scalar_tests = [
    [
        "a,->a",
        1,
    ],
    ["ab,->ab", 1],
    [",a,->a", 2],
    [",,a,->a", 3],
    [",,->", 2],
]


def check_path(test_output: PathType, benchmark: PathType, bypass: bool = False) -> bool:
    if not isinstance(test_output, list):
        return False

    if len(test_output) != len(benchmark):
        return False

    ret = True
    for pos in range(len(test_output)):
        ret &= isinstance(test_output[pos], tuple)
        ret &= test_output[pos] == list(benchmark)[pos]
    return ret


def assert_contract_order(func: Any, test_data: Any, max_size: int, benchmark: PathType) -> None:
    test_output = func(test_data[0], test_data[1], test_data[2], max_size)
    assert check_path(test_output, benchmark)


def test_size_by_dict() -> None:
    sizes_dict = {}
    for ind, val in zip("abcdez", [2, 5, 9, 11, 13, 0]):
        sizes_dict[ind] = val

    path_func = oe.helpers.compute_size_by_dict

    assert 1 == path_func("", sizes_dict)
    assert 2 == path_func("a", sizes_dict)
    assert 5 == path_func("b", sizes_dict)

    assert 0 == path_func("z", sizes_dict)
    assert 0 == path_func("az", sizes_dict)
    assert 0 == path_func("zbc", sizes_dict)

    assert 104 == path_func("aaae", sizes_dict)
    assert 12870 == path_func("abcde", sizes_dict)


def test_flop_cost() -> None:
    size_dict = {v: 10 for v in "abcdef"}

    # Loop over an array
    assert 10 == oe.helpers.flop_count("a", False, 1, size_dict)

    # Hadamard product (*)
    assert 10 == oe.helpers.flop_count("a", False, 2, size_dict)
    assert 100 == oe.helpers.flop_count("ab", False, 2, size_dict)

    # Inner product (+, *)
    assert 20 == oe.helpers.flop_count("a", True, 2, size_dict)
    assert 200 == oe.helpers.flop_count("ab", True, 2, size_dict)

    # Inner product x3 (+, *, *)
    assert 30 == oe.helpers.flop_count("a", True, 3, size_dict)

    # GEMM
    assert 2000 == oe.helpers.flop_count("abc", True, 2, size_dict)


def test_bad_path_option() -> None:
    with pytest.raises(KeyError):
        oe.contract("a,b,c", [1], [2], [3], optimize="optimall", shapes=True)  # type: ignore


def test_explicit_path() -> None:
    pytest.importorskip("numpy")
    x = oe.contract("a,b,c", [1], [2], [3], optimize=[(1, 2), (0, 1)])
    assert x.item() == 6


def test_path_optimal() -> None:
    test_func = oe.paths.optimal

    test_data = explicit_path_tests["GEMM1"]
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])


def test_path_greedy() -> None:
    test_func = oe.paths.greedy

    test_data = explicit_path_tests["GEMM1"]
    assert_contract_order(test_func, test_data, 5000, [(0, 2), (0, 1)])
    assert_contract_order(test_func, test_data, 0, [(0, 1, 2)])


def test_memory_paths() -> None:
    expression = "abc,bdef,fghj,cem,mhk,ljk->adgl"

    views = build_shapes(expression)

    # Test tiny memory limit
    path_ret = oe.contract_path(expression, *views, optimize="optimal", memory_limit=5, shapes=True)
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])

    path_ret = oe.contract_path(expression, *views, optimize="greedy", memory_limit=5, shapes=True)
    assert check_path(path_ret[0], [(0, 1, 2, 3, 4, 5)])

    # Check the possibilities, greedy is capped
    path_ret = oe.contract_path(expression, *views, optimize="optimal", memory_limit=-1, shapes=True)
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])

    path_ret = oe.contract_path(expression, *views, optimize="greedy", memory_limit=-1, shapes=True)
    assert check_path(path_ret[0], [(0, 3), (0, 4), (0, 2), (0, 2), (0, 1)])


@pytest.mark.parametrize("alg,expression,order", path_edge_tests)
def test_path_edge_cases(alg: OptimizeKind, expression: str, order: PathType) -> None:
    views = build_shapes(expression)

    # Test tiny memory limit
    path_ret = oe.contract_path(expression, *views, optimize=alg, shapes=True)
    assert check_path(path_ret[0], order)


@pytest.mark.parametrize("expression,order", path_scalar_tests)
@pytest.mark.parametrize("alg", oe.paths._PATH_OPTIONS)
def test_path_scalar_cases(alg: OptimizeKind, expression: str, order: PathType) -> None:
    views = build_shapes(expression)

    # Test tiny memory limit
    path_ret = oe.contract_path(expression, *views, optimize=alg, shapes=True)
    # print(path_ret[0])
    assert len(path_ret[0]) == order


def test_optimal_edge_cases() -> None:
    # Edge test5
    expression = "a,ac,ab,ad,cd,bd,bc->"
    edge_test4 = build_shapes(expression, dimension_dict={"a": 20, "b": 20, "c": 20, "d": 20})
    path, _ = oe.contract_path(expression, *edge_test4, optimize="greedy", memory_limit="max_input", shapes=True)
    assert check_path(path, [(0, 1), (0, 1, 2, 3, 4, 5)])

    path, _ = oe.contract_path(expression, *edge_test4, optimize="optimal", memory_limit="max_input", shapes=True)
    assert check_path(path, [(0, 1), (0, 1, 2, 3, 4, 5)])


def test_greedy_edge_cases() -> None:
    expression = "abc,cfd,dbe,efa"
    dim_dict = {k: 20 for k in expression.replace(",", "")}
    tensors = build_shapes(expression, dimension_dict=dim_dict)

    path, _ = oe.contract_path(expression, *tensors, optimize="greedy", memory_limit="max_input", shapes=True)
    assert check_path(path, [(0, 1, 2, 3)])

    path, _ = oe.contract_path(expression, *tensors, optimize="greedy", memory_limit=-1, shapes=True)
    assert check_path(path, [(0, 1), (0, 2), (0, 1)])


def test_dp_edge_cases_dimension_1() -> None:
    eq = "nlp,nlq,pl->n"
    shapes = [(1, 1, 1), (1, 1, 1), (1, 1)]
    info = oe.contract_path(eq, *shapes, shapes=True, optimize="dp")[1]
    assert max(info.scale_list) == 3


def test_dp_edge_cases_all_singlet_indices() -> None:
    eq = "a,bcd,efg->"
    shapes = [(2,), (2, 2, 2), (2, 2, 2)]
    info = oe.contract_path(eq, *shapes, shapes=True, optimize="dp")[1]
    assert max(info.scale_list) == 3


def test_custom_dp_can_optimize_for_outer_products() -> None:
    eq = "a,b,abc->c"

    da, db, dc = 2, 2, 3
    shapes = [(da,), (db,), (da, db, dc)]

    opt1 = oe.DynamicProgramming(search_outer=False)
    opt2 = oe.DynamicProgramming(search_outer=True)

    info1 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt1)[1]
    info2 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt2)[1]

    assert info2.opt_cost < info1.opt_cost


def test_custom_dp_can_optimize_for_size() -> None:
    eq, shapes = rand_equation(10, 4, seed=43)

    opt1 = oe.DynamicProgramming(minimize="flops")
    opt2 = oe.DynamicProgramming(minimize="size")

    info1 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt1)[1]
    info2 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt2)[1]

    assert info1.opt_cost < info2.opt_cost
    assert info1.largest_intermediate > info2.largest_intermediate


def test_custom_dp_can_set_cost_cap() -> None:
    eq, shapes = rand_equation(5, 3, seed=42)
    opt1 = oe.DynamicProgramming(cost_cap=True)
    opt2 = oe.DynamicProgramming(cost_cap=False)
    opt3 = oe.DynamicProgramming(cost_cap=100)
    info1 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt1)[1]
    info2 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt2)[1]
    info3 = oe.contract_path(eq, *shapes, shapes=True, optimize=opt3)[1]
    assert info1.opt_cost == info2.opt_cost == info3.opt_cost


@pytest.mark.parametrize(
    "minimize,cost,width,path",
    [
        ("flops", 663054, 18900, [(4, 5), (2, 5), (2, 7), (5, 6), (1, 5), (1, 4), (0, 3), (0, 2), (0, 1)]),
        ("size", 1114440, 2016, [(2, 7), (3, 8), (3, 7), (2, 6), (1, 5), (1, 4), (1, 3), (1, 2), (0, 1)]),
        ("write", 983790, 2016, [(0, 8), (3, 4), (1, 4), (5, 6), (1, 5), (0, 4), (0, 3), (1, 2), (0, 1)]),
        ("combo", 973518, 2016, [(4, 5), (2, 5), (6, 7), (2, 6), (1, 5), (1, 4), (0, 3), (0, 2), (0, 1)]),
        ("limit", 983832, 2016, [(2, 7), (3, 4), (0, 4), (3, 6), (2, 5), (0, 4), (0, 3), (1, 2), (0, 1)]),
        ("combo-256", 983790, 2016, [(0, 8), (3, 4), (1, 4), (5, 6), (1, 5), (0, 4), (0, 3), (1, 2), (0, 1)]),
        ("limit-256", 983832, 2016, [(2, 7), (3, 4), (0, 4), (3, 6), (2, 5), (0, 4), (0, 3), (1, 2), (0, 1)]),
    ],
)
def test_custom_dp_can_set_minimize(minimize: str, cost: int, width: int, path: PathType) -> None:
    eq, shapes = rand_equation(10, 4, seed=43)
    opt = oe.DynamicProgramming(minimize=minimize)
    info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)[1]
    assert info.path == path
    assert info.opt_cost == cost
    assert info.largest_intermediate == width


def test_dp_errors_when_no_contractions_found() -> None:
    eq, shapes = rand_equation(10, 3, seed=42)

    # first get the actual minimum cost
    opt = oe.DynamicProgramming(minimize="size")
    _, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
    mincost = info.largest_intermediate

    # check we can still find it without minimizing size explicitly
    oe.contract_path(eq, *shapes, shapes=True, memory_limit=mincost, optimize="dp")

    # but check just below this threshold raises
    with pytest.raises(RuntimeError):
        oe.contract_path(eq, *shapes, shapes=True, memory_limit=mincost - 1, optimize="dp")


@pytest.mark.parametrize("optimize", ["greedy", "branch-2", "branch-all", "optimal", "dp"])
def test_can_optimize_outer_products(optimize: OptimizeKind) -> None:
    a, b, c = ((10, 10) for _ in range(3))
    d = (10, 2)

    assert oe.contract_path("ab,cd,ef,fg", a, b, c, d, optimize=optimize, shapes=True)[0] == [
        (2, 3),
        (0, 2),
        (0, 1),
    ]


@pytest.mark.parametrize("num_symbols", [2, 3, 26, 26 + 26, 256 - 140, 300])
def test_large_path(num_symbols: int) -> None:
    symbols = "".join(oe.get_symbol(i) for i in range(num_symbols))
    dimension_dict = dict(zip(symbols, itertools.cycle([2, 3, 4])))
    expression = ",".join(symbols[t : t + 2] for t in range(num_symbols - 1))
    tensors = build_shapes(expression, dimension_dict=dimension_dict)

    # Check that path construction does not crash
    oe.contract_path(expression, *tensors, optimize="greedy", shapes=True)


def test_custom_random_greedy() -> None:
    np = pytest.importorskip("numpy")

    eq, shapes = rand_equation(10, 4, seed=42)
    views = list(map(np.ones, shapes))

    with pytest.raises(ValueError):
        oe.RandomGreedy(minimize="something")

    optimizer = oe.RandomGreedy(max_repeats=10, minimize="flops")
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)

    assert len(optimizer.costs) == 10
    assert len(optimizer.sizes) == 10

    assert path == optimizer.path
    assert optimizer.best["flops"] == min(optimizer.costs)
    assert path_info.largest_intermediate == optimizer.best["size"]
    assert path_info.opt_cost == optimizer.best["flops"]

    # check can change settings and run again
    optimizer.temperature = 0.0
    optimizer.max_repeats = 6
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)

    assert len(optimizer.costs) == 16
    assert len(optimizer.sizes) == 16

    assert path == optimizer.path
    assert optimizer.best["size"] == min(optimizer.sizes)
    assert path_info.largest_intermediate == optimizer.best["size"]
    assert path_info.opt_cost == optimizer.best["flops"]

    # check error if we try and reuse the optimizer on a different expression
    eq, shapes = rand_equation(10, 4, seed=41)
    views = list(map(np.ones, shapes))
    with pytest.raises(ValueError):
        path, path_info = oe.contract_path(eq, *views, optimize=optimizer)


def test_custom_branchbound() -> None:
    np = pytest.importorskip("numpy")

    eq, shapes = rand_equation(8, 4, seed=42)
    views = list(map(np.ones, shapes))
    optimizer = oe.BranchBound(nbranch=2, cutoff_flops_factor=10, minimize="size")

    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)

    assert path == optimizer.path
    assert path_info.largest_intermediate == optimizer.best["size"]
    assert path_info.opt_cost == optimizer.best["flops"]

    # tweak settings and run again
    optimizer.nbranch = 3
    optimizer.cutoff_flops_factor = 4
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)

    assert path == optimizer.path
    assert path_info.largest_intermediate == optimizer.best["size"]
    assert path_info.opt_cost == optimizer.best["flops"]

    # check error if we try and reuse the optimizer on a different expression
    eq, shapes = rand_equation(8, 4, seed=41)
    views = list(map(np.ones, shapes))
    with pytest.raises(ValueError):
        path, path_info = oe.contract_path(eq, *views, optimize=optimizer)


def test_branchbound_validation() -> None:
    with pytest.raises(ValueError):
        oe.BranchBound(nbranch=0)


def test_parallel_random_greedy() -> None:
    np = pytest.importorskip("numpy")

    pool = ProcessPoolExecutor(2)

    eq, shapes = rand_equation(10, 4, seed=42)
    views = list(map(np.ones, shapes))

    optimizer = oe.RandomGreedy(max_repeats=10, parallel=pool)
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)

    assert len(optimizer.costs) == 10
    assert len(optimizer.sizes) == 10

    assert path == optimizer.path
    assert optimizer.parallel is pool
    assert optimizer._executor is pool
    assert optimizer.best["flops"] == min(optimizer.costs)
    assert path_info.largest_intermediate == optimizer.best["size"]
    assert path_info.opt_cost == optimizer.best["flops"]

    # now switch to max time algorithm
    optimizer.max_repeats = int(1e6)
    optimizer.max_time = 0.2
    optimizer.parallel = 2

    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)

    assert len(optimizer.costs) > 10
    assert len(optimizer.sizes) > 10

    assert path == optimizer.path
    assert optimizer.best["flops"] == min(optimizer.costs)
    assert path_info.largest_intermediate == optimizer.best["size"]
    assert path_info.opt_cost == optimizer.best["flops"]

    optimizer.parallel = True
    assert optimizer._executor is not None
    assert optimizer._executor is not pool

    are_done = [f.running() or f.done() for f in optimizer._futures]
    assert all(are_done)


def test_custom_path_optimizer() -> None:
    np = pytest.importorskip("numpy")

    class NaiveOptimizer(oe.paths.PathOptimizer):
        def __call__(
            self,
            inputs: List[ArrayIndexType],
            output: ArrayIndexType,
            size_dict: Dict[str, int],
            memory_limit: Optional[int] = None,
        ) -> PathType:
            self.was_used = True
            return [(0, 1)] * (len(inputs) - 1)

    eq, shapes = rand_equation(5, 3, seed=42, d_max=3)
    views = list(map(np.ones, shapes))

    exp = oe.contract(eq, *views, optimize=False)

    optimizer = NaiveOptimizer()
    out = oe.contract(eq, *views, optimize=optimizer)
    assert exp == out
    assert optimizer.was_used


def test_custom_random_optimizer() -> None:
    np = pytest.importorskip("numpy")

    class NaiveRandomOptimizer(oe.path_random.RandomOptimizer):
        @staticmethod
        def random_path(
            r: int, n: int, inputs: List[ArrayIndexType], output: ArrayIndexType, size_dict: Dict[str, int]
        ) -> Any:
            """Picks a completely random contraction order."""
            np.random.seed(r)
            ssa_path: List[TensorShapeType] = []
            remaining = set(range(n))
            while len(remaining) > 1:
                i, j = np.random.choice(list(remaining), size=2, replace=False)
                remaining.add(n + len(ssa_path))
                remaining.remove(i)
                remaining.remove(j)
                ssa_path.append((i, j))
            cost, size = oe.path_random.ssa_path_compute_cost(ssa_path, inputs, output, size_dict)
            return ssa_path, cost, size

        def setup(self, inputs: Any, output: Any, size_dict: Any) -> Any:
            self.was_used = True
            n = len(inputs)
            trial_fn = self.random_path
            trial_args = (n, inputs, output, size_dict)
            return trial_fn, trial_args

    eq, shapes = rand_equation(5, 3, seed=42, d_max=3)
    views = list(map(np.ones, shapes))

    exp = oe.contract(eq, *views, optimize=False)

    optimizer = NaiveRandomOptimizer(max_repeats=16)
    out = oe.contract(eq, *views, optimize=optimizer)
    assert exp == out
    assert optimizer.was_used

    assert len(optimizer.costs) == 16


def test_optimizer_registration() -> None:
    def custom_optimizer(
        inputs: List[ArrayIndexType], output: ArrayIndexType, size_dict: Dict[str, int], memory_limit: Optional[int]
    ) -> PathType:
        return [(0, 1)] * (len(inputs) - 1)

    with pytest.raises(KeyError):
        oe.paths.register_path_fn("optimal", custom_optimizer)

    oe.paths.register_path_fn("custom", custom_optimizer)
    assert "custom" in oe.paths._PATH_OPTIONS

    eq = "ab,bc,cd"
    shapes = [(2, 3), (3, 4), (4, 5)]
    path, _ = oe.contract_path(eq, *shapes, shapes=True, optimize="custom")  # type: ignore
    assert path == [(0, 1), (0, 1)]
    del oe.paths._PATH_OPTIONS["custom"]


def test_path_with_assumed_shapes() -> None:
    path, _ = oe.contract_path("ab,bc,cd", [[5, 3]], [[2], [4]], [[3, 2]])
    assert path == [(0, 1), (0, 1)]
