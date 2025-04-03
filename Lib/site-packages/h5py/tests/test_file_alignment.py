import h5py
from .common import TestCase


def is_aligned(dataset, offset=4096):
    # Here we check if the dataset is aligned
    return dataset.id.get_offset() % offset == 0


def dataset_name(i):
    return f"data{i:03}"


class TestFileAlignment(TestCase):
    """
        Ensure that setting the file alignment has the desired effect
        in the internal structure.
    """
    def test_no_alignment_set(self):
        fname = self.mktemp()
        # 881 is a prime number, so hopefully this help randomize the alignment
        # enough
        # A nice even number might give a pathological case where
        # While we don't want the data to be aligned, it ends up aligned...
        shape = (881,)

        with h5py.File(fname, 'w') as h5file:
            # Create up to 1000 datasets
            # At least one of them should be misaligned.
            # While this isn't perfect, it seems that there
            # The case where 1000 datasets get created is one where the data
            # is aligned. Therefore, during correct operation, this test is
            # expected to finish quickly
            for i in range(1000):
                dataset = h5file.create_dataset(
                    dataset_name(i), shape, dtype='uint8')
                # Assign data so that the dataset is instantiated in
                # the file
                dataset[...] = i
                if not is_aligned(dataset):
                    # Break early asserting that the file is not aligned
                    break
            else:
                raise RuntimeError("Data was all found to be aligned to 4096")

    def test_alignment_set_above_threshold(self):
        # 2022/01/19 hmaarrfk
        # UnitTest (TestCase) doesn't play well with pytest parametrization.
        alignment_threshold = 1000
        alignment_interval = 4096

        for shape in [
            (1033,),  # A prime number above the threshold
            (1000,),  # Exactly equal to the threshold
            (1001,),  # one above the threshold
        ]:
            fname = self.mktemp()
            with h5py.File(fname, 'w',
                           alignment_threshold=alignment_threshold,
                           alignment_interval=alignment_interval) as h5file:
                # Create up to 1000 datasets
                # They are all expected to be aligned
                for i in range(1000):
                    dataset = h5file.create_dataset(
                        dataset_name(i), shape, dtype='uint8')
                    # Assign data so that the dataset is instantiated in
                    # the file
                    dataset[...] = (i % 256)  # Truncate to uint8
                    assert is_aligned(dataset, offset=alignment_interval)

    def test_alignment_set_below_threshold(self):
        # 2022/01/19 hmaarrfk
        # UnitTest (TestCase) doesn't play well with pytest parametrization.
        alignment_threshold = 1000
        alignment_interval = 1024

        for shape in [
            (881,),  # A prime number below the threshold
            (999,),  # Exactly one below the threshold
        ]:
            fname = self.mktemp()
            with h5py.File(fname, 'w',
                           alignment_threshold=alignment_threshold,
                           alignment_interval=alignment_interval) as h5file:
                # Create up to 1000 datasets
                # At least one of them should be misaligned.
                # While this isn't perfect, it seems that there
                # The case where 1000 datasets get created is one where the
                # data is aligned. Therefore, during correct operation, this
                # test is expected to finish quickly
                for i in range(1000):
                    dataset = h5file.create_dataset(
                        dataset_name(i), shape, dtype='uint8')
                    # Assign data so that the dataset is instantiated in
                    # the file
                    dataset[...] = i
                    if not is_aligned(dataset, offset=alignment_interval):
                        # Break early asserting that the file is not aligned
                        break
                else:
                    raise RuntimeError(
                        "Data was all found to be aligned to "
                        f"{alignment_interval}. This is highly unlikely.")
