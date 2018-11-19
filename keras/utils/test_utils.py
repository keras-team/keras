"""Utilities related to Keras unit tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO

from mock import patch, Mock, MagicMock

import numpy as np
from numpy.testing import assert_allclose

from .generic_utils import has_arg
from ..engine import Model, Input
from .. import backend as K

try:
    from tensorflow.python.lib.io import file_io as tf_file_io
except ImportError:
    tf_file_io = None


def get_test_data(num_train=1000, num_test=500, input_shape=(10,),
                  output_shape=(2,),
                  classification=True, num_classes=2):
    """Generates test data to train a model on.

    classification=True overrides output_shape
    (i.e. output_shape is set to (1,)) and the output
    consists in integers in [0, num_classes-1].

    Otherwise: float output with shape output_shape.
    """
    samples = num_train + num_test
    if classification:
        y = np.random.randint(0, num_classes, size=(samples,))
        X = np.zeros((samples,) + input_shape, dtype=np.float32)
        for i in range(samples):
            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((samples,))
        X = np.zeros((samples,) + input_shape, dtype=np.float32)
        y = np.zeros((samples,) + output_shape, dtype=np.float32)
        for i in range(samples):
            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (X[:num_train], y[:num_train]), (X[num_train:], y[num_train:])


def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None, fixed_batch_size=False):
    """Test routine for a layer with a single input tensor
    and single output tensor.
    """
    # generate input data
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = K.floatx()
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = (10 * np.random.random(input_data_shape))
        input_data = input_data.astype(input_dtype)
    else:
        if input_shape is None:
            input_shape = input_data.shape
        if input_dtype is None:
            input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    expected_output_shape = layer.compute_output_shape(input_shape)

    # test in functional API
    if fixed_batch_size:
        x = Input(batch_shape=input_shape, dtype=input_dtype)
    else:
        x = Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    assert K.dtype(y) == expected_output_dtype

    # check with the functional API
    model = Model(x, y)

    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(expected_output_shape,
                                        actual_output_shape):
        if expected_dim is not None:
            assert expected_dim == actual_dim

    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = model.__class__.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        _output = recovered_model.predict(input_data)
        assert_allclose(_output, actual_output, rtol=1e-3)

    # test training mode (e.g. useful when the layer has a
    # different behavior at training and testing time).
    if has_arg(layer.call, 'training'):
        model.compile('rmsprop', 'mse')
        model.train_on_batch(input_data, actual_output)

    # test instantiation from layer config
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    # for further checks in the caller function
    return actual_output


class tf_file_io_proxy(object):
    """TODO"""
    _gcp_prefix = 'gs://'
    _test_bucket_env_key = 'GCP_TEST_BUCKET'

    def __init__(self, file_io_module=None, bucket_name=None):
        if bucket_name is None:
            bucket_name = os.environ.get(self._test_bucket_env_key, None)
        if bucket_name is None:
            # will mock gcp locally for tests
            if file_io_module is None:
                raise ValueError('`file_io_module` must be provided for mocking')
            self.mock_gcp = True
            self.file_io_module = file_io_module
            self.objects = {}
            self.bucket_name = 'mock-bucket'
        else:
            # will use real bucket for tests
            if bucket_name.startswith(self._gcp_prefix):
                bucket_name = bucket_name[len(self._gcp_prefix):]
            self.bucket_name = bucket_name
            if tf_file_io is None:
                raise ImportError(
                    'tensorflow must be installed to read/write to GCP')
            try:
                # check that bucket exists and is accessible
                tf_file_io.is_directory(self.bucket_path)
            except Exception:  # TODO
                raise IOError(
                    'could not access provided bucket {}'.format(self.bucket_path))
            self.mock_gcp = False
            self.file_io_module = None
            self.objects = None

        self.patched_file_io = None
        self._is_started = False

    @property
    def bucket_path(self):
        """Returns the full GCP bucket path"""
        return self._gcp_prefix + self.bucket_name

    def get_filepath(self, filename):
        """Returns filename appended to bucketpath"""
        return os.path.join(self.bucket_path, filename)

    def FileIO(self, filepath, mode):
        """Proxy for tensorflow.python.lib.io.file_io.FileIO class. Mocks the class
        if a real GCP bucket is not available for testing.
        """
        self._check_started()
        if filepath.startswith(self._gcp_prefix):
            mock_fio = MagicMock()
            mock_fio.__enter__ = Mock(return_value=mock_fio)
            if mode == 'r':
                if filepath not in self.objects:
                    raise IOError('TODO')
                self.objects[filepath].seek(0)
                mock_fio.read = self.objects[filepath].read
            elif mode == 'w':
                self.objects[filepath] = BytesIO()
                mock_fio.write = self.objects[filepath].write
            else:
                raise ValueError(
                    '{} only supports wrapping of FileIO for `mode` "r" or "w"')
            return mock_fio
        else:
            return open(filepath, mode)

    def file_exists(self, filepath):
        """Proxy for tensorflow.python.lib.io.file_io.file_exists class. Mocks the
        function if a real GCP bucket is not available for testing.
        """
        self._check_started()
        if not self.mock_gcp:
            return tf_file_io.file_exists(filepath)

        if filepath.startswith(self._gcp_prefix):
            return filepath in self.objects
        return os.path.exists(filepath)

    def assert_exists(self, filepath):
        """Convenience method to verfiy that a file exists after writing."""
        self._check_started()
        if not self.file_exists(filepath):
            raise AssertionError('{} does not exist'.format(filepath))

    def _check_started(self):
        if not self._is_started:
            raise RuntimeError('tf_file_io_proxy is not started')

    def start(self):
        """Start mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if self._is_started:
            raise RuntimeError('start called on already started tf_file_io_proxy')
        if self.mock_gcp:
            mock_module = Mock()
            mock_module.FileIO = self.FileIO
            mock_module.file_exists = self.file_exists
            patched_file_io = patch(self.file_io_module, new=mock_module)
            self.patched_file_io = patched_file_io
            self.patched_file_io.start()
        self._is_started = True

    def stop(self):
        """Stop mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if not self._is_started:
            raise RuntimeError('stop called on unstarted tf_file_io_proxy')
        if self.mock_gcp:
            self.patched_file_io.stop()
        self._is_started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
