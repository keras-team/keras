import functools

from keras import backend
from keras.api_export import keras_export
from keras.backend.common.variables import ALLOWED_DTYPES

# Default dtypes corresponding to Python scalars.
PYTHON_SCALAR_DTYPES = {
    bool: "bool",
    int: "int64",
    float: "float64",
    # TODO: support complex dtypes
    # complex: "complex128",
}
BIT64_TO_BIT16_DTYPE = {
    "int32": "int16",
    "int64": "int16",
    "uint32": "uint16",
    "uint64": "uint16",
    "float32": "float16",
    "float64": "float16",
    # TODO: support complex dtypes
    # "complex64": "complex32",
    # "complex128": "complex32",
}
BIT64_TO_BIT32_DTYPE = {
    "int64": "int32",
    "uint64": "uint32",
    "float64": "float32",
    # TODO: support complex dtypes
    # "complex128": "complex64",
}


@functools.lru_cache(maxsize=None)
def canonicalize_dtype(float_type, dtype):
    if float_type == "float16":
        return BIT64_TO_BIT16_DTYPE.get(dtype, dtype)
    elif float_type == "float32":
        return BIT64_TO_BIT32_DTYPE.get(dtype, dtype)
    elif float_type == "float64":
        return dtype
    else:
        raise ValueError(
            f"Invalid float_type: {float_type}, `float_type` must be one of"
            f"('float16', 'float32', 'float64')."
        )


def dtype(x, *, canonicalize=False):
    """Return the dtype object for a tensor or type."""
    dtype = None
    if x is None:
        raise ValueError(f"Invalid argument to dtype: {x}.")
    # when inputing python scalar or tensor
    elif type(x) in PYTHON_SCALAR_DTYPES:
        dtype = PYTHON_SCALAR_DTYPES[type(x)]
    elif backend.is_tensor(x):
        dtype = backend.standardize_dtype(x.dtype)
    # when inputing python dtype or tensor dtype
    elif isinstance(x, type) and x in PYTHON_SCALAR_DTYPES:
        dtype = PYTHON_SCALAR_DTYPES[x]
    elif hasattr(x, "name"):
        dtype = x.name
    elif hasattr(x, "__str__") and "torch" in str(dtype):
        dtype = str(x).split(".")[-1]
    # when inputing dtype string
    elif x in ALLOWED_DTYPES:
        dtype = x
    elif x in ["int", "float"]:
        if x == "int":
            dtype = "int64"
        else:
            dtype = "float64"

    if dtype is None or dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Invalid dtype: {x}")
    return (
        canonicalize_dtype(backend.floatx(), dtype) if canonicalize else dtype
    )


_bool_types = ["bool"]
_int_types = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
]
_float_types = ["bfloat16", "float16", "float32", "float64"]
_complex_types = ["complex64", "complex128"]
_weak_types = ["int", "float", "complex"]
_weak_types_python = [int, float, complex]


def _default_types(dtype):
    if not isinstance(dtype, str):
        raise TypeError(f"Invalid type: {type(dtype)}. Must be a str.")
    if dtype == "bfloat16":  # special case for bfloat16
        dtype_indicator = "f"
    else:
        dtype_indicator = dtype[:1]

    # get backend.floatx precision (16, 32, 64)
    precision = backend.floatx()[-2:]

    if dtype_indicator == "b":
        return "bool"
    elif dtype_indicator == "i":
        return "int" + precision
    elif dtype_indicator == "u":
        return "uint" + precision
    elif dtype_indicator == "f":
        return "float" + precision
    # TODO: support complex types
    else:
        raise ValueError(f"Invalid dtype: {dtype} that is failed to be parsed")


def _keras_core_type(dtype, weak_type):
    """Return the keras core type for a dtype and weak type."""
    if weak_type:
        if dtype == "bool":
            return dtype
        if "float" in dtype:
            return "float"
        if "int" in dtype:
            return "int"
        # TODO: support complex types
        # if "complex" in dtype:
        #     return "complex"
    return dtype


def is_weakly_typed(x):
    return type(x) in _weak_types_python


def _dtype_and_weaktype(value):
    """Return a (dtype, weak_type) tuple for the given input."""
    return dtype(value), any(
        value is typ for typ in _weak_types_python
    ) or is_weakly_typed(value)


def _type_promotion_lattice():
    """
    Return the type promotion lattice in the form of a DAG.
    This DAG maps each type to its immediately higher type on the lattice.
    """
    (b1,) = _bool_types
    (u1, u2, u4, u8, i1, i2, i4, i8) = _int_types
    bf, f2, f4, f8 = _float_types
    c4, c8 = _complex_types
    i_, f_, c_ = _weak_types
    out = {
        b1: [i_],
        u1: [i2, u2],
        u2: [i4, u4],
        u4: [i8, u8],
        u8: [f_],
        i_: [u1, i1],
        i1: [i2],
        i2: [i4],
        i4: [i8],
        i8: [f_],
        f_: [bf, f2, c_],
        bf: [f4],
        f2: [f4],
        f4: [f8, c4],
        f8: [c8],
        c_: [c4],
        c4: [c8],
        c8: [],
    }
    return out


def _make_lattice_upper_bounds():
    lattice = _type_promotion_lattice()
    upper_bounds = {node: {node} for node in lattice}
    for n in lattice:
        while True:
            new_upper_bounds = set().union(
                *(lattice[b] for b in upper_bounds[n])
            )
            if n in new_upper_bounds:
                raise ValueError(
                    f"cycle detected in type promotion lattice for node {n}"
                )
            if new_upper_bounds.issubset(upper_bounds[n]):
                break
            upper_bounds[n] |= new_upper_bounds
    return upper_bounds


_lattice_upper_bounds = _make_lattice_upper_bounds()


@functools.lru_cache(512)
def _least_upper_bound(*nodes):
    """Compute the least upper bound of a set of nodes.

    Args:
        nodes: sequence of entries from _jax_types + _weak_types

    Returns:
        The type representing the least upper bound of the input nodes on the
        promotion lattice.
    """
    # This function computes the least upper bound of a set of nodes N within a
    # partially ordered set defined by the lattice generated above.
    # Given a partially ordered set S, let the set of upper bounds of n ∈ S be
    #   UB(n) ≡ {m ∈ S | n ≤ m}
    # Further, for a set of nodes N ⊆ S, let the set of common upper bounds be
    # given by
    #   CUB(N) ≡ {a ∈ S | ∀ b ∈ N: a ∈ UB(b)}
    # Then the least upper bound of N is defined as
    #   LUB(N) ≡ {c ∈ CUB(N) | ∀ d ∈ CUB(N), c ≤ d}
    # The definition of an upper bound implies that
    #   c ≤ d if and only if d ∈ UB(c),
    # so the LUB can be expressed:
    #   LUB(N) = {c ∈ CUB(N) | ∀ d ∈ CUB(N): d ∈ UB(c)}
    # or, equivalently:
    #   LUB(N) = {c ∈ CUB(N) | CUB(N) ⊆ UB(c)}
    # By definition, LUB(N) has a cardinality of 1 for a partially ordered set.
    # Note a potential algorithmic shortcut: from the definition of CUB(N),
    # we have
    #   ∀ c ∈ N: CUB(N) ⊆ UB(c)
    # So if N ∩ CUB(N) is nonempty, if follows that LUB(N) = N ∩ CUB(N).
    N = set(nodes)
    UB = _lattice_upper_bounds
    try:
        bounds = [UB[n] for n in N]
    except KeyError:
        dtype = next(n for n in N if n not in UB)
        raise ValueError(
            f"{dtype=} is not a valid dtype for Keras Core type promotion."
        )
    CUB = set.intersection(*bounds)
    LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
    if len(LUB) == 1:
        return LUB.pop()
    elif len(LUB) == 0:
        msg = (
            f"Input dtypes {tuple(str(n) for n in nodes)} have no available "
            "implicit dtype promotion path. Try explicitly casting inputs to "
            "the desired output type."
        )
        raise ValueError(msg)
    else:
        # If we get here, it means the lattice is ill-formed.
        raise ValueError(
            f"Internal Type Promotion error: {nodes} do not have a unique "
            f"least upper bound on the specified lattice; options are {LUB}. "
            "This is an unexpected error in Keras Core's internal logic; "
            "please report it to the maintainers."
        )


def _lattice_result_type(*args):
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
    if len(dtypes) == 1:
        out_dtype = dtypes[0]
        out_weak_type = weak_types[0]
    elif len(set(dtypes)) == 1 and not all(weak_types):
        # Trivial promotion case. This allows extended dtypes through.
        out_dtype = dtypes[0]
        out_weak_type = False
    elif all(weak_types):
        # If all inputs are weakly typed, we compute the bound of the
        # strongly-typed counterparts and apply the weak type at the end. This
        # avoids returning the incorrect result with non-canonical weak types
        # (e.g. weak int16).
        # TODO: explore removing this special case.
        result_type = _least_upper_bound(
            *{_keras_core_type(dtype, False) for dtype in dtypes}
        )
        out_dtype = dtype(result_type)
        out_weak_type = True
    else:
        result_type = _least_upper_bound(
            *{_keras_core_type(d, w) for d, w in zip(dtypes, weak_types)}
        )
        out_dtype = dtype(result_type)
        out_weak_type = any(result_type is t for t in _weak_types)
    return out_dtype, (out_dtype != "bool") and out_weak_type


@keras_export("keras.backend.result_dtype")
def result_dtype(*args, return_weak_type_flag=False):
    """Convenience function to apply Keras Core argument dtype promotion.

    This function attempts to match the result of `jax.numpy.result_dtype`.
    """
    # TODO: support weaktype
    if len(args) == 0:
        raise ValueError("at least one array or dtype is required")
    dtype, weak_type = _lattice_result_type(
        *(backend.floatx() if arg is None else arg for arg in args)
    )
    if weak_type:
        dtype = _default_types(dtype)
    else:
        dtype = canonicalize_dtype(backend.floatx(), dtype)
    return (dtype, weak_type) if return_weak_type_flag else dtype
