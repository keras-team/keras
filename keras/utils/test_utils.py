"""Utilities related to Keras unit tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO

import numpy as np
from numpy.testing import assert_allclose

from .generic_utils import has_arg
from ..engine import Model, Input
from .. import backend as K

try:
    from tensorflow.python.lib.io import file_io as tf_file_io
except ImportError:
    tf_file_io = None

try:
    from unittest.mock import patch, Mock, MagicMock
except:
    from mock import patch, Mock, MagicMock


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
    """Context manager for mock patching `tensorflow.python.lib.io.file_io` in tests.

    The purpose of this class is to be able to tests model saving/loading to/from
    Google Cloud Storage, for witch the tensorflow `file_io` package is used.

    If a `bucket_name` is provided, either as an input argument or by setting the
    environment variable GCS_TEST_BUCKET, *NO mocking* will be done and files will be
    transferred to the real GCS bucket. For this to work, valid Google application
    credentials must be available, see:
        https://cloud.google.com/video-intelligence/docs/common/auth
    for further details.

    If a `bucket_name` is not provided, an identifier of the import of the file_io
    module to mock must be provided, using the `file_io_module` argument.
    NOTE that only part of the module is mocked and that the same Exceptions
    are not raised in mock implementation.

    Since the bucket name can be provided using an environment variable, it is
    recommended to use method `get_filepath(filename)` in tests to make them
    pass with and without a real GCS bucket during testing. See example below.

    # Arguments
        file_io_module: String identifier of the file_io module import to patch. E.g
            'keras.engine.saving.tf_file_io'
        bucket_name: String identifier of *a real* GCS bucket (with or without the
            'gs://' prefix). A bucket name provided with argument precedes what is
            specified using the GCS_TEST_BUCKET environment variable.

    # Example
    ```python
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))

    with tf_file_io_proxy('keras.engine.saving.tf_file_io') as file_io_proxy:
        gcs_filepath = file_io_proxy.get_filepath(filename='model.h5')
        save_model(model, gcs_filepath)
        file_io_proxy.assert_exists(gcs_filepath)
        new_model_gcs = load_model(gcs_filepath)
        file_io_proxy.delete_file(gcs_filepath)  # cleanup
    ```
    """
    _gcs_prefix = 'gs://'
    _test_bucket_env_key = 'GCS_TEST_BUCKET'

    def __init__(self, file_io_module=None, bucket_name=None):
        if bucket_name is None:
            bucket_name = os.environ.get(self._test_bucket_env_key, None)
        if bucket_name is None:
            # will mock gcs locally for tests
            if file_io_module is None:
                raise ValueError('`file_io_module` must be provided for mocking')
            self.mock_gcs = True
            self.file_io_module = file_io_module
            self.local_objects = {}
            self.bucket_name = 'mock-bucket'
        else:
            # will use real bucket for tests
            if bucket_name.startswith(self._gcs_prefix):
                bucket_name = bucket_name[len(self._gcs_prefix):]
            self.bucket_name = bucket_name
            if tf_file_io is None:
                raise ImportError(
                    'tensorflow must be installed to read/write to GCS')
            try:
                # check that bucket exists and is accessible
                tf_file_io.is_directory(self.bucket_path)
            except:
                raise IOError(
                    'could not access provided bucket {}'.format(self.bucket_path))
            self.mock_gcs = False
            self.file_io_module = None
            self.local_objects = None

        self.patched_file_io = None
        self._is_started = False

    @property
    def bucket_path(self):
        """Returns the full GCS bucket path"""
        return self._gcs_prefix + self.bucket_name

    def get_filepath(self, filename):
        """Returns filename appended to bucketpath"""
        return os.path.join(self.bucket_path, filename)

    def FileIO(self, name, mode):
        """Proxy for tensorflow.python.lib.io.file_io.FileIO class. Mocks the class
        if a real GCS bucket is not available for testing.
        """
        self._check_started()
        if not self.mock_gcs:
            return tf_file_io.FileIO(name, mode)

        filepath = name
        if filepath.startswith(self._gcs_prefix):
            mock_fio = MagicMock()
            mock_fio.__enter__ = Mock(return_value=mock_fio)
            if mode == 'rb':
                if filepath not in self.local_objects:
                    raise IOError('{} does not exist'.format(filepath))
                self.local_objects[filepath].seek(0)
                mock_fio.read = self.local_objects[filepath].read
            elif mode == 'wb':
                self.local_objects[filepath] = BytesIO()
                mock_fio.write = self.local_objects[filepath].write
            else:
                raise ValueError(
                    '{} only supports wrapping of FileIO for `mode` "rb" or "wb"')
            return mock_fio

        return open(filepath, mode)

    def file_exists(self, filename):
        """Proxy for tensorflow.python.lib.io.file_io.file_exists class. Mocks the
        function if a real GCS bucket is not available for testing.
        """
        self._check_started()
        if not self.mock_gcs:
            return tf_file_io.file_exists(filename)

        if filename.startswith(self._gcs_prefix):
            return filename in self.local_objects

        return os.path.exists(filename)

    def delete_file(self, filename):
        """Proxy for tensorflow.python.lib.io.file_io.delete_file function. Mocks
        the function if a real GCS bucket is not available for testing.
        """
        if not self.mock_gcs:
            tf_file_io.delete_file(filename)
        elif filename.startswith(self._gcs_prefix):
            self.local_objects.pop(filename)
        else:
            os.remove(filename)

    def assert_exists(self, filepath):
        """Convenience method for verifying that a file exists after writing."""
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
        if self.mock_gcs:
            mock_module = Mock()
            mock_module.FileIO = self.FileIO
            mock_module.file_exists = self.file_exists
            mock_module.delete_file = self.delete_file
            patched_file_io = patch(self.file_io_module, new=mock_module)
            self.patched_file_io = patched_file_io
            self.patched_file_io.start()
        self._is_started = True

    def stop(self):
        """Stop mocking of `self.file_io_module` if real bucket not
        available for testing"""
        if not self._is_started:
            raise RuntimeError('stop called on unstarted tf_file_io_proxy')
        if self.mock_gcs:
            self.patched_file_io.stop()
        self._is_started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
