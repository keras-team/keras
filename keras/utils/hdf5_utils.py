"""HDF5 related utilities."""
import numpy as np
import sys


try:
    import h5py
    HDF5_OBJECT_HEADER_LIMIT = 64512
except ImportError:
    h5py = None


if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle


#FLAGS
_is_boxed = '_is_dict_boxed'


class H5Dict(object):
    """ A dict-like wrapper around h5py groups (or dicts).

    This allows us to have a single serialization logic
    for both pickling and saving to disk.

    Note: This is not intended to be a generic wrapper.
    There are lot of edge cases which have been hardcoded,
    and makes sense only in the context of model serialization / 
    deserialization.
    """

    def __init__(self, path, mode='a'):
        if isinstance(path, h5py.Group):
            self.data = path
            self._is_file = False
        elif type(path) is str:
            self.data = h5py.File(path,)
            self._is_file = True
        elif type(path) is dict:
            self.data = path
            self._is_file = False
            self.data[_is_boxed] = True
        else:
            raise Exception('Required Group, str or dict. Received {}.'.format(type(path)))
        self.read_only = mode == 'r'

    def __setitem__(self, attr, val):
        if self.read_only:
            raise Exception('Can not set item in read only mode.')
        is_np = type(val).__module__ == np.__name__
        if type(self.data) is dict:
            if type(attr) is bytes:
                attr = attr.decode('utf-8')
            if is_np:
                self.data[attr] = pickle.dumps(val)
                # We have to remeber to unpickle in __getitem__
                self.data['_{}_pickled'.format(attr)] = True
            else:
                self.data[attr] = val
            return
        if attr in self:
            raise Exception('Can not set attribute. Group with name {} exists.'.format(attr))
        if is_np:
            dataset = self.data.create_dataset(attr, val.shape, dtype=val.dtype)
            if not val.shape:
                # scalar
                dataset[()] = val
            else:
                dataset[:] = val
        if isinstance(val, list):
            # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
            # because in that case even chunking the array would not make the saving
            # possible.
            bad_attributes = [x for x in val if len(x) > HDF5_OBJECT_HEADER_LIMIT]

            # Expecting this to never be true.
            if len(bad_attributes) > 0:
                raise RuntimeError('The following attributes cannot be saved to HDF5 '
                                'file because they are larger than %d bytes: %s'
                                % (HDF5_OBJECT_HEADER_LIMIT,
                                    ', '.join([x for x in bad_attributes])))

            data_npy = np.asarray(val)

            num_chunks = 1
            chunked_data = np.array_split(data_npy, num_chunks)

            # This will never loop forever thanks to the test above.
            while any(map(lambda x: x.nbytes > HDF5_OBJECT_HEADER_LIMIT, chunked_data)):
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
        if type(self.data) is dict:
            if type(attr) is bytes:
                attr = attr.decode('utf-8')
            if attr in self.data:
                val = self.data[attr]
                if type(val) is dict and val.get(_is_boxed):
                    val =  H5Dict(val)
                elif '_{}_pickled'.format(attr) in self.data:
                    val = pickle.loads(val)
                return val
            else:
                if self.read_only:
                    raise Exception('Can not create group in read only mode.')
                val = {_is_boxed : True}
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
            #could be chunked
            chunk_attr = '%s%d' % (attr, 0)
            is_chunked =  chunk_attr in self.data.attrs
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
                    raise Exception('Can not create group in read only mode.')
                val = H5Dict(self.data.create_group(attr))
        return val

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(list(self.data))

    def iter(self):
        return list(self.data)

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
        if type(self.data) is dict:
            self.data.update(*args)
        raise NotImplementedError

    def __contains__(self, key):
        if type(self.data) is dict:
            return key in self.data
        else:
            return (key in self.data) or (key in self.data.attrs)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


h5dict = H5Dict
