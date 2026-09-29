"""Utilities shared by the built-in datasets."""

import io
import pickle
import zipfile

import numpy as np

# Globals required to reconstruct the numpy object arrays stored inside the
# IMDB / Reuters `.npz` files. These datasets store ragged sequences as
# `dtype=object` arrays, which numpy can only persist as a pickle stream, so
# `np.load(allow_pickle=False)` refuses them outright ("Object arrays cannot be
# loaded when allow_pickle=False"). Rebuilding such an array uses exactly three
# globals -- `numpy.ndarray`, `numpy.dtype` and `multiarray._reconstruct` -- and
# nothing else (verified against the actual IMDB and Reuters files). Both the
# numpy < 2 (`numpy.core`) and numpy >= 2 (`numpy._core`) spellings of
# `multiarray` are listed so files written by either can be read. Anything
# outside this set (e.g. `os.system`, `builtins.eval`, `subprocess.Popen`) is
# refused, which prevents a tampered `.npz` from executing code through a pickle
# `__reduce__` gadget.
_ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)

_RECONSTRUCT_GLOBALS = frozenset(
    {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)


def _validate_shape(shape):
    """Returns `shape` as a tuple of Python ints, or refuses the stream.

    `numpy.core.multiarray._reconstruct` and `ndarray.__setstate__` both take a
    shape; validating it here keeps anything that is not a plain, non-negative
    tuple of integers out of numpy's C allocation code.
    """
    if not isinstance(shape, tuple):
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array whose shape is not a tuple "
            f"({type(shape).__name__}) while loading a Keras dataset."
        )
    try:
        shape = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as e:
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array whose shape contains a "
            "non-integer dimension while loading a Keras dataset."
        ) from e
    if any(dimension < 0 for dimension in shape):
        raise pickle.UnpicklingError(
            f"Refusing to deserialize an array with a negative shape ({shape}) "
            "while loading a Keras dataset."
        )
    return shape


def _validate_dtype(dtype):
    """Returns `dtype` as a `numpy.dtype`, or refuses the stream."""
    try:
        return np.dtype(dtype)
    except TypeError as e:
        raise pickle.UnpicklingError(
            f"Refusing to deserialize an array with an invalid dtype ({dtype!r}) "
            "while loading a Keras dataset."
        ) from e


def _nbytes(shape, dtype):
    """Number of bytes `shape` x `dtype` describes."""
    size = dtype.itemsize
    for dimension in shape:
        size *= dimension
    return size


class _RestrictedArray(np.ndarray):
    """An `ndarray` whose pickle state is validated before it is applied.

    The global allowlist above is not sufficient on its own: a stream that only
    ever asks for allowlisted globals can still build an array with one dtype
    and then hand numpy a *state* that carries another one. `_reconstruct()`
    allocates for the dtype it is given, while `array_setstate()` uses the
    dtype and shape of the state to decide how many bytes to copy, so a state
    describing more bytes than the array holds makes numpy memcpy past the end
    of the allocation.

    Every array produced by `RestrictedUnpickler` is therefore an instance of
    this class, and the state is checked against the array that was actually
    allocated before `ndarray.__setstate__` gets to it.
    """

    def __setstate__(self, state):
        # numpy's own `__reduce__` always writes the 5-tuple
        # (version, shape, dtype, is_fortran_order, raw_data).
        if not isinstance(state, tuple) or len(state) != 5:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state is not a "
                "5-element tuple while loading a Keras dataset."
            )
        version, shape, dtype, _is_fortran_order, raw_data = state
        if version != 1:
            raise pickle.UnpicklingError(
                f"Refusing to deserialize an array with unknown state version "
                f"{version} while loading a Keras dataset."
            )
        shape = _validate_shape(shape)
        dtype = _validate_dtype(dtype)

        # numpy serializes an object array as an empty `int8` placeholder plus
        # a state carrying the real dtype, shape and contents, and
        # `array_setstate()` allocates a fresh buffer for it. Any other array
        # was allocated with real data, and numpy reuses that buffer, so the
        # state must describe the very same array: a different dtype would
        # reinterpret the buffer, and more bytes than it holds would be copied
        # past its end.
        placeholder = self.size == 0 and self.nbytes == 0
        declared = _nbytes(shape, dtype)
        if not placeholder and dtype != self.dtype:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state changes its "
                f"dtype from {self.dtype!r} to {dtype!r} while loading a Keras "
                "dataset."
            )
        if not placeholder and declared > self.nbytes:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state declares "
                f"{declared} bytes over a {self.nbytes}-byte buffer; applying it "
                "would write out of bounds while loading a Keras dataset."
            )

        # `raw_data` is `None`, the list of objects of an object array, or the
        # raw bytes of a numeric one.
        if raw_data is not None and not isinstance(raw_data, (bytes, list)):
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state carries "
                f"unexpected data ({type(raw_data).__name__}) while loading a "
                "Keras dataset."
            )
        if dtype.hasobject and isinstance(raw_data, bytes):
            # An object array must carry its contents as a list of objects;
            # raw bytes would be reinterpreted as object pointers.
            raise pickle.UnpicklingError(
                "Refusing to deserialize an object array whose pickle state "
                "carries raw bytes instead of objects while loading a Keras "
                "dataset."
            )
        if isinstance(raw_data, bytes) and len(raw_data) != declared:
            raise pickle.UnpicklingError(
                "Refusing to deserialize an array whose pickle state carries an "
                f"unexpected number of bytes ({len(raw_data)} != {declared}) "
                "while loading a Keras dataset."
            )
        super().__setstate__(state)


class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that only allows numpy array reconstruction globals."""

    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing to deserialize `{module}.{name}` while loading a Keras "
                "dataset. The file may be corrupted or malicious."
            )
        obj = super().find_class(module, name)
        if (module, name) in _RECONSTRUCT_GLOBALS:
            reconstruct = obj

            def restricted_reconstruct(subtype, shape, dtype):
                return _reconstruct_array(reconstruct, subtype, shape, dtype)

            return restricted_reconstruct
        if (module, name) == ("numpy", "ndarray"):
            # `ndarray(...)` can also be reached directly through `REDUCE`, so
            # the class itself is replaced by the validating subclass.
            return _RestrictedArray
        return obj


def _reconstruct_array(reconstruct, subtype, shape, dtype):
    """`multiarray._reconstruct` returning a state-validating array."""
    if not (isinstance(subtype, type) and issubclass(subtype, np.ndarray)):
        raise pickle.UnpicklingError(
            "Refusing to deserialize an array with a non-array subtype while "
            "loading a Keras dataset."
        )
    shape = _validate_shape(shape)
    dtype = _validate_dtype(dtype)
    array = reconstruct(np.ndarray, shape, dtype)
    return array.view(_RestrictedArray)


def load_npy_member(fp):
    """Read a single `.npy` stream without allowing arbitrary unpickling.

    Numeric arrays are read with pickling disabled entirely. Object arrays
    (which genuinely require pickle) are read with `RestrictedUnpickler`, so
    only numpy array reconstruction is permitted.
    """
    try:
        # Fast path: numeric arrays load with pickle fully disabled.
        return np.lib.format.read_array(fp, allow_pickle=False)
    except ValueError:
        # Object array: rewind, skip the header, then restrict the unpickler.
        fp.seek(0)
        version = np.lib.format.read_magic(fp)
        if version[0] == 1:
            np.lib.format.read_array_header_1_0(fp)
        elif version[0] in (2, 3):
            # numpy exposes no public `read_array_header_3_0`. The 3.0 format
            # only differs from 2.0 by encoding the header string as UTF-8
            # instead of latin1; the 4-byte header-length layout we skip past
            # to reach the pickle stream is identical, so 2.0's reader handles
            # both.
            np.lib.format.read_array_header_2_0(fp)
        else:
            raise ValueError(f"Unsupported `.npy` file version: {version}.")
        array = RestrictedUnpickler(fp).load()
        if isinstance(array, np.ndarray):
            # Hand back a plain array: the validating subclass is an
            # implementation detail of the unpickler.
            array = array.view(np.ndarray)
        return array


def load_npz(path):
    """Safely load an `.npz` archive into a `dict` of arrays.

    This is a drop-in replacement for `np.load(path, allow_pickle=True)` for
    the built-in datasets that store ragged object arrays. Unlike
    `allow_pickle=True`, it only ever unpickles numpy arrays, so loading a
    maliciously crafted file cannot execute arbitrary code.

    Args:
        path: Path to the `.npz` file.

    Returns:
        A `dict` mapping each member name to its array.
    """
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as raw:
                buffer = io.BytesIO(raw.read())
            arrays[member[: -len(".npy")]] = load_npy_member(buffer)
    return arrays
