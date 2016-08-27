from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import sys
from collections import defaultdict


class HDF5Matrix():
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start, end, normalizer=None):
        import h5py

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.start = start
        self.end = end
        self.data = f[dataset]
        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start+self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key + self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        if self.normalizer is not None:
            return self.normalizer(self.data[idx])
        else:
            return self.data[idx]

    @property
    def shape(self):
        return (self.end - self.start,) + self.data.shape[1:]


def save_array(array, name):
    import tables
    f = tables.open_file(name, 'w')
    atom = tables.Atom.from_dtype(array.dtype)
    ds = f.createCArray(f.root, 'data', atom, array.shape)
    ds[:] = array
    f.close()


def load_array(name):
    import tables
    f = tables.open_file(name)
    array = f.root.data
    a = np.empty(shape=array.shape, dtype=array.dtype)
    a[:] = array[:]
    f.close()
    return a


def ask_to_proceed_with_overwrite(filepath):
    get_input = input
    if sys.version_info[:2] <= (2, 7):
        get_input = raw_input
    overwrite = get_input('[WARNING] %s already exists - overwrite? '
                          '[y/n]' % (filepath))
    while overwrite not in ['y', 'n']:
        overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True
