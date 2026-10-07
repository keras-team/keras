from unittest import mock

from keras.src import testing
from keras.src.datasets import fashion_mnist


class _AllFilesRequested(Exception):
    """Raised by the fake `get_file` once every request is recorded."""


class FashionMnistTest(testing.TestCase):
    def test_downloads_are_hash_verified(self):
        # `get_file` only verifies a download, and reuses an existing cache
        # entry, when it is given a `file_hash`. Without one, whatever sits at
        # the predictable cache path is read as training data.
        requested = []

        def fake_get_file(fname, origin=None, **kwargs):
            requested.append((fname, kwargs.get("file_hash")))
            if len(requested) == len(fashion_mnist.FILE_HASHES):
                raise _AllFilesRequested()
            return fname

        with mock.patch.object(fashion_mnist, "get_file", fake_get_file):
            with self.assertRaises(_AllFilesRequested):
                fashion_mnist.load_data()

        self.assertEqual(
            sorted(fname for fname, _ in requested),
            sorted(fashion_mnist.FILE_HASHES),
        )
        for fname, file_hash in requested:
            self.assertEqual(file_hash, fashion_mnist.FILE_HASHES[fname])
            self.assertRegex(file_hash, r"^[0-9a-f]{64}$")
