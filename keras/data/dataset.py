# TODO add some default like hdf5 dataset, directory, etc
class Dataset():
    """
    Base object for every dataset
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
