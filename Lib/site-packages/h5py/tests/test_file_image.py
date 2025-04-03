import numpy as np

import h5py
from h5py import h5f, h5p

from .common import ut, TestCase

class TestFileImage(TestCase):
    def test_load_from_image(self):
        from binascii import a2b_base64
        from zlib import decompress

        compressed_image = 'eJzr9HBx4+WS4mIAAQ4OBhYGAQZk8B8KKjhQ+TD5BCjNCKU7oPQKJpg4I1hOAiouCDUfXV1IkKsrSPV/NACzx4AFQnMwjIKRCDxcHQNAdASUD0ulJ5hQ1ZWkFpeAaFh69KDQXkYGNohZjDA+JCUzMkIEmKHqELQAWKkAByytOoBJViAPJM7ExATWyAE0B8RgZkyAJmlYDoEAIahukJoNU6+HMTA0UOgT6oBgP38XUI6G5UMFZrzKR8EoGAUjGMDKYVgxDSsuAHcfMK8='

        image = decompress(a2b_base64(compressed_image))

        fapl = h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_fapl_core()
        fapl.set_file_image(image)

        fid = h5f.open(self.mktemp().encode(), h5py.h5f.ACC_RDONLY, fapl=fapl)
        f = h5py.File(fid)

        self.assertTrue('test' in f)

    def test_open_from_image(self):
        from binascii import a2b_base64
        from zlib import decompress

        compressed_image = 'eJzr9HBx4+WS4mIAAQ4OBhYGAQZk8B8KKjhQ+TD5BCjNCKU7oPQKJpg4I1hOAiouCDUfXV1IkKsrSPV/NACzx4AFQnMwjIKRCDxcHQNAdASUD0ulJ5hQ1ZWkFpeAaFh69KDQXkYGNohZjDA+JCUzMkIEmKHqELQAWKkAByytOoBJViAPJM7ExATWyAE0B8RgZkyAJmlYDoEAIahukJoNU6+HMTA0UOgT6oBgP38XUI6G5UMFZrzKR8EoGAUjGMDKYVgxDSsuAHcfMK8='

        image = decompress(a2b_base64(compressed_image))

        fid = h5f.open_file_image(image)
        f = h5py.File(fid)

        self.assertTrue('test' in f)


def test_in_memory():
    arr = np.arange(10)
    # Passing one fcpl & one fapl parameter to exercise the code splitting them:
    with h5py.File.in_memory(track_order=True, rdcc_nbytes=2_000_000) as f1:
        f1['a'] = arr
        f1.flush()
        img = f1.id.get_file_image()

        # Open while f1 is still open
        with h5py.File.in_memory(img) as f2:
            np.testing.assert_array_equal(f2['a'][:], arr)

    # Reuse image now that previous files are closed
    with h5py.File.in_memory(img) as f3:
        np.testing.assert_array_equal(f3['a'][:], arr)
