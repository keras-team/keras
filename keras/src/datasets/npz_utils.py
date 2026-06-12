"""Utilities shared by the built-in datasets."""

import io
import pickle
import zipfile

import numpy as np

# Globals that are legitimately required to reconstruct the numpy object arrays
# stored inside the IMDB / Reuters `.npz` files (ragged sequences are saved as
# `dtype=object` arrays, which numpy persists as pickle streams). Both the
# numpy < 2 (`numpy.core`) and numpy >= 2 (`numpy._core`) module spellings are
# listed so files written by either can be read. Anything outside this set
# (e.g. `os.system`, `builtins.eval`, `subprocess.Popen`) is refused, which
# prevents a tampered `.npz` from executing code through a pickle `__reduce__`
# gadget.
_ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that only allows numpy array reconstruction globals."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to deserialize `{module}.{name}` while loading a Keras "
            "dataset. The `.npz` file may be corrupted or malicious."
        )


def _load_npy_member(fp):
    """Read a single `.npy` stream without allowing arbitrary unpickling.

    Numeric arrays are read with pickling disabled entirely. Object arrays
    (which genuinely require pickle) are read with `_RestrictedUnpickler`, so
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
        return _RestrictedUnpickler(fp).load()


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
            arrays[member[: -len(".npy")]] = _load_npy_member(buffer)
    return arrays
