"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict
import sys
import contextlib


import six
try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None


if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle


class HDF5Matrix(object):
    """Representation of HDF5 dataset to be used instead of a Numpy array.

    # Example

    ```python
        x_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(x_data)
    ```

    Providing `start` and `end` allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.

    # Arguments
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified
            in datapath
        start: int, start of desired slice of the specified dataset
        end: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved

    # Returns
        An array-like HDF5 dataset.
    """
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
        if h5py is None:
            raise ImportError('The use of HDF5Matrix requires '
                              'HDF5 and h5py installed.')

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.data = f[dataset]
        self.start = start
        if end is None:
            self.end = self.data.shape[0]
        else:
            self.end = end
        self.normalizer = normalizer
        if self.normalizer is not None:
            first_val = self.normalizer(self.data[0:1])
        else:
            first_val = self.data[0:1]
        self._base_shape = first_val.shape[1:]
        self._base_dtype = first_val.dtype

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if stop + self.start <= self.end:
                idx = slice(start + self.start, stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, (int, np.integer)):
            if key + self.start < self.end:
                idx = key + self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        else:
            # Assume list/iterable
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
        """Gets a numpy-style shape tuple giving the dataset dimensions.

        # Returns
            A numpy-style shape tuple.
        """
        return (self.end - self.start,) + self._base_shape

    @property
    def dtype(self):
        """Gets the datatype of the dataset.

        # Returns
            A numpy dtype string.
        """
        return self._base_dtype

    @property
    def ndim(self):
        """Gets the number of dimensions (rank) of the dataset.

        # Returns
            An integer denoting the number of dimensions (rank) of the dataset.
        """
        return self.data.ndim

    @property
    def size(self):
        """Gets the total dataset size (number of elements).

        # Returns
            An integer denoting the number of elements in the dataset.
        """
        return np.prod(self.shape)


def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.

    # Arguments
        filepath: the path to the file to be overwritten.

    # Returns
        True if we can proceed with overwrite, False otherwise.
    """
    overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                                '[y/n]' % (filepath)).strip().lower()
    while overwrite not in ('y', 'n'):
        overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                    '(cancel).').strip().lower()
    if overwrite == 'n':
        return False
    print('[TIP] Next time specify overwrite=True!')
    return True


class H5Dict(object):
    """ A dict-like wrapper around h5py groups (or dicts).

    This allows us to have a single serialization logic
    for both pickling and saving to disk.

    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization/
    deserialization.

    # Arguments
        path: Either a string (path on disk), a Path, a dict, or a HDF5 Group.
        mode: File open mode (one of `{"a", "r", "w"}`).
    """

    def __init__(self, path, mode='a'):
        if isinstance(path, h5py.Group):
            self.data = path
            self._is_file = False
        elif isinstance(path, six.string_types) or _is_path_instance(path):
            self.data = h5py.File(path, mode=mode)
            self._is_file = True
        elif isinstance(path, dict):
            self.data = path
            self._is_file = False
            if mode == 'w':
                self.data.clear()
            # Flag to check if a dict is user defined data or a sub group:
            self.data['_is_group'] = True
        else:
            raise TypeError('Required Group, str, Path or dict. '
                            'Received: {}.'.format(type(path)))
        self.read_only = mode == 'r'

    @staticmethod
    def is_supported_type(path):
        """Check if `path` is of supported type for instantiating a `H5Dict`"""
        return (
            isinstance(path, h5py.Group) or
            isinstance(path, dict) or
            isinstance(path, six.string_types) or
            _is_path_instance(path)
        )

    def __setitem__(self, attr, val):
        if self.read_only:
            raise ValueError('Cannot set item in read-only mode.')
        is_np = type(val).__module__ == np.__name__
        if isinstance(self.data, dict):
            if isinstance(attr, bytes):
                attr = attr.decode('utf-8')
            if is_np:
                self.data[attr] = pickle.dumps(val)
                # We have to remember to unpickle in __getitem__
                self.data['_{}_pickled'.format(attr)] = True
            else:
                self.data[attr] = val
            return
        if isinstance(self.data, h5py.Group) and attr in self.data:
            raise KeyError('Cannot set attribute. '
                           'Group with name "{}" exists.'.format(attr))
        if is_np:
            dataset = self.data.create_dataset(attr, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                dataset[()] = val
            else:
                dataset[:] = val
        elif isinstance(val, (list, tuple)):
            # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
            # because in that case even chunking the array would not make the saving
            # possible.
            bad_attributes = [x for x in val if len(x) > HDF5_OBJECT_HEADER_LIMIT]

            # Expecting this to never be true.
            if bad_attributes:
                raise RuntimeError('The following attributes cannot be saved to '
                                   'HDF5 file because they are larger than '
                                   '%d bytes: %s' % (HDF5_OBJECT_HEADER_LIMIT,
                                                     ', '.join(bad_attributes)))

            if (val and sys.version_info[0] == 3 and isinstance(
                    val[0], six.string_types)):
                # convert to bytes
                val = [x.encode('utf-8') for x in val]

            data_npy = np.asarray(val)

            num_chunks = 1
            chunked_data = np.array_split(data_npy, num_chunks)

            # This will never loop forever thanks to the test above.
            is_too_big = lambda x: x.nbytes > HDF5_OBJECT_HEADER_LIMIT
            while any(map(is_too_big, chunked_data)):
                num_chunks += 1
                chunked_data = np.array_split(data_npy, num_chunks)

            if num_chunks > 1:
                for chunk_id, chunk_data in enumerate(chunked_data):
                    self.data.attrs['%s%d' % (attr, chunk_id)] = chunk_data
            else:
                self.data.attrs[attr] = val
        else:
            self.data.attrs[attr] = val

    def __getitem__(self, attr):
        if isinstance(self.data, dict):
            if isinstance(attr, bytes):
                attr = attr.decode('utf-8')
            if attr in self.data:
                val = self.data[attr]
                if isinstance(val, dict) and val.get('_is_group'):
                    val = H5Dict(val)
                elif '_{}_pickled'.format(attr) in self.data:
                    val = pickle.loads(val)
                return val
            else:
                if self.read_only:
                    raise ValueError('Cannot create group in read-only mode.')
                val = {'_is_group': True}
                self.data[attr] = val
                return H5Dict(val)
        if attr in self.data.attrs:
            val = self.data.attrs[attr]
            if type(val).__module__ == np.__name__:
                if val.dtype.type == np.string_:
                    val = val.tolist()
        elif attr in self.data:
            val = self.data[attr]
            if isinstance(val, h5py.Dataset):
                val = np.asarray(val)
            else:
                val = H5Dict(val)
        else:
            # could be chunked
            chunk_attr = '%s%d' % (attr, 0)
            is_chunked = chunk_attr in self.data.attrs
            if is_chunked:
                val = []
                chunk_id = 0
                while chunk_attr in self.data.attrs:
                    chunk = self.data.attrs[chunk_attr]
                    val.extend([x.decode('utf8') for x in chunk])
                    chunk_id += 1
                    chunk_attr = '%s%d' % (attr, chunk_id)
            else:
                if self.read_only:
                    raise ValueError('Cannot create group in read-only mode.')
                val = H5Dict(self.data.create_group(attr))
        return val

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def iter(self):
        return iter(self.data)

    def __getattr__(self, attr):

        def wrapper(f):
            def h5wrapper(*args, **kwargs):
                out = f(*args, **kwargs)
                if isinstance(self.data, type(out)):
                    return H5Dict(out)
                else:
                    return out
            return h5wrapper

        return wrapper(getattr(self.data, attr))

    def close(self):
        if isinstance(self.data, h5py.Group):
            self.data.file.flush()
            if self._is_file:
                self.data.close()

    def update(self, *args):
        if isinstance(self.data, dict):
            self.data.update(*args)
        raise NotImplementedError

    def __contains__(self, key):
        if isinstance(self.data, dict):
            return key in self.data
        else:
            return (key in self.data) or (key in self.data.attrs)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


h5dict = H5Dict


def load_from_binary_h5py(load_function, stream):
    """Calls `load_function` on a `h5py.File` read from the binary `stream`.

    # Arguments
        load_function: A function that takes a `h5py.File`, reads from it, and
            returns any object.
        stream: Any file-like object implementing the method `read` that returns
            `bytes` data (e.g. `io.BytesIO`) that represents a valid h5py file image.

    # Returns
        The object returned by `load_function`.
    """
    # Implementation based on suggestion solution here:
    #   https://github.com/keras-team/keras/issues/9343#issuecomment-440903847
    binary_data = stream.read()
    file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    file_access_property_list.set_fapl_core(backing_store=False)
    file_access_property_list.set_file_image(binary_data)
    file_id_args = {'fapl': file_access_property_list,
                    'flags': h5py.h5f.ACC_RDONLY,
                    'name': b'in-memory-h5py'}  # name does not matter
    h5_file_args = {'backing_store': False,
                    'driver': 'core',
                    'mode': 'r'}
    with contextlib.closing(h5py.h5f.open(**file_id_args)) as file_id:
        with h5py.File(file_id, **h5_file_args) as h5_file:
            return load_function(h5_file)


def save_to_binary_h5py(save_function, stream):
    """Calls `save_function` on an in memory `h5py.File`.

    The file is subsequently written to the binary `stream`.

     # Arguments
        save_function: A function that takes a `h5py.File`, writes to it and
            (optionally) returns any object.
        stream: Any file-like object implementing the method `write` that accepts
            `bytes` data (e.g. `io.BytesIO`).
     """
    with h5py.File('in-memory-h5py', driver='core', backing_store=False) as h5file:
        # note that filename does not matter here.
        return_value = save_function(h5file)
        h5file.flush()
        binary_data = h5file.fid.get_file_image()
    stream.write(binary_data)

    return return_value


def _is_path_instance(path):
    # We can't use isinstance here because it would require
    # us to add pathlib2 to the Python 2 dependencies.
    class_name = type(path).__name__
    return class_name == 'PosixPath' or class_name == 'WindowsPath'
