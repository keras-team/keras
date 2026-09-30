import os
import pickle
import zipfile

import numpy as np

from keras.src import testing
from keras.src.datasets import npz_utils

# numpy < 2 spells the module `numpy.core`, numpy >= 2 `numpy._core`; both
# spellings are allow-listed by `RestrictedUnpickler`.
_RECONSTRUCT = (getattr(np, "_core", None) or np.core).multiarray._reconstruct


def _crafted_npz(path, construct_args, state):
    """Writes a `.npz` whose `.npy` member holds a hand-built pickle stream.

    `construct_args` are the arguments handed to `multiarray._reconstruct` and
    `state` the 5-tuple numpy's `__reduce__` writes, so the tests below can
    describe an array that is deliberately inconsistent with what the
    constructor allocated.
    """

    class Crafted:
        def __reduce__(self):
            return (_RECONSTRUCT, construct_args, state())

    header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
    header += b" " * ((64 - (len(header) + 10) % 64) % 64)
    npy = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    npy += pickle.dumps(Crafted(), protocol=3)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x.npy", npy)
    return path


class LoadNpzTest(testing.TestCase):
    def test_loads_ragged_object_arrays(self):
        # IMDB/Reuters store ragged sequences as `dtype=object` arrays.
        xs = np.array([[1, 14, 22], [1, 194], [1, 14, 47, 8]], dtype=object)
        ys = np.array([1, 0, 1])
        path = os.path.join(self.get_temp_dir(), "ragged.npz")
        np.savez(path, x=xs, y=ys)

        loaded = npz_utils.load_npz(path)

        self.assertEqual(
            [list(seq) for seq in loaded["x"]],
            [[1, 14, 22], [1, 194], [1, 14, 47, 8]],
        )
        self.assertAllEqual(loaded["y"], ys)

    def test_loads_numeric_arrays(self):
        path = os.path.join(self.get_temp_dir(), "numeric.npz")
        np.savez(path, a=np.arange(10), b=np.ones((3, 4), dtype="float32"))

        loaded = npz_utils.load_npz(path)

        self.assertAllEqual(loaded["a"], np.arange(10))
        self.assertEqual(loaded["b"].shape, (3, 4))
        self.assertEqual(loaded["b"].dtype, np.float32)

    def test_rejects_pickle_gadget(self):
        # A crafted member whose unpickling would run code must be refused,
        # without executing the payload.
        marker = os.path.join(self.get_temp_dir(), "marker")

        class Exploit:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        payload = np.empty(1, dtype=object)
        payload[0] = Exploit()
        path = os.path.join(self.get_temp_dir(), "evil.npz")
        np.savez(path, x=payload)

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)
        self.assertFalse(os.path.exists(marker))

    def test_rejects_array_state_dtype_confusion(self):
        # Only allow-listed globals are used, but the state declares a larger
        # dtype than `_reconstruct` allocated for the same shape: applying it
        # used to write past the end of the buffer and kill the interpreter.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "confusion.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (
                1,
                (1,),
                np.dtype([("a", "i8"), ("b", "O")]),
                False,
                b"\x00" * 16,
            ),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_array_state_byte_overflow(self):
        # Same idea without changing the dtype: the state describes more bytes
        # than the array holds.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "overflow.npz"),
            (np.ndarray, (1,), np.dtype("i8")),
            lambda: (1, (8,), np.dtype("i8"), False, b"\x00" * 64),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_carrying_raw_bytes(self):
        # An object array must carry its contents as a list of objects; raw
        # bytes would be reread as object pointers.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "rawbytes.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (1, (1,), np.dtype("O"), False, b"\x00" * 8),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_with_fewer_objects_than_shape(self):
        # A state that carries fewer objects than `shape` has elements used to
        # make numpy index past the end of the list without any bounds check,
        # which segfaulted the interpreter.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_too_few.npz"),
            (np.ndarray, (2, 3), np.dtype("O")),
            lambda: (1, (2, 3), np.dtype("O"), False, [None] * 5),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_with_empty_object_list(self):
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_empty.npz"),
            (np.ndarray, (1,), np.dtype("O")),
            lambda: (1, (1,), np.dtype("O"), False, []),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_object_state_with_more_objects_than_shape(self):
        # A list longer than the array is silently truncated by numpy, and the
        # surplus objects are dropped without being released.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "objects_too_many.npz"),
            (np.ndarray, (2,), np.dtype("O")),
            lambda: (1, (2,), np.dtype("O"), False, [None] * 3),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_numeric_state_carrying_a_list(self):
        # Raw bytes are the only thing numpy writes for a numeric array; a list
        # used to reach numpy and raise a bare `TypeError`.
        path = _crafted_npz(
            os.path.join(self.get_temp_dir(), "numeric_list.npz"),
            (np.ndarray, (2,), np.dtype("i8")),
            lambda: (1, (2,), np.dtype("i8"), False, [1, 2]),
        )

        with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
            npz_utils.load_npz(path)

    def test_rejects_shape_with_non_integer_dimension(self):
        # `int(dimension)` truncated a float shape silently, and `inf` raised an
        # `OverflowError` that escaped as something other than an unpickling
        # error, so both are now refused before any conversion happens.
        for name, dimension in (
            ("float", 1.5),
            ("integral_float", 1.0),
            ("infinity", float("inf")),
            ("nan", float("nan")),
            ("bool", True),
            ("string", "1"),
        ):
            path = _crafted_npz(
                os.path.join(self.get_temp_dir(), f"shape_{name}.npz"),
                (np.ndarray, (1,), np.dtype("i8")),
                lambda dimension=dimension: (
                    1,
                    (dimension,),
                    np.dtype("i8"),
                    False,
                    b"\x00" * 8,
                ),
            )

            with self.assertRaisesRegex(pickle.UnpicklingError, "Refusing"):
                npz_utils.load_npz(path)

    def test_loads_empty_object_array(self):
        # A zero-element object array legitimately carries an empty list, which
        # the object-state length check has to keep accepting.
        path = os.path.join(self.get_temp_dir(), "empty_object.npz")
        np.savez(path, x=np.empty((0,), dtype=object))

        loaded = npz_utils.load_npz(path)

        self.assertEqual(loaded["x"].shape, (0,))
        self.assertEqual(loaded["x"].dtype, np.dtype(object))
