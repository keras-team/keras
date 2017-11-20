import pytest
from keras.backend import mxnet_backend as K
from keras.backend import tensorflow_backend as KTF
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
import six
from test_util import *
from mxnet.base import MXNetError

"""Note: This is a backend test only for using MXNet backend. Many test case are copied from 
Keras backend test with some modification. The goal is quickly verifying MXNet backend works 
as expected. We will merge the MXNet specfic test cases into Keras backend tests. 
"""


def tensorflow_clean_session(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        KTF.clear_session()
        return output
    return wrapper


def parse_shape_or_val(shape_or_val):
    if isinstance(shape_or_val, np.ndarray):
        return shape_or_val.shape, shape_or_val
    else:
        return shape_or_val, np.random.random(shape_or_val) - 0.5  # [-0.5, 0.5]


@tensorflow_clean_session
def check_single_tensor_operation(function_name, x_shape_or_val, **kwargs):
    """
    :param function_name:
        The function in the backend
    :param x_shape_or_val:
        It can be shape of the numpy array or actual numpy array
    :param kwargs:
        Some arguments that the backend function need
    :return:
    """
    is_shape = kwargs.pop('is_shape', True)
    assert_value_with_ref = kwargs.pop('assert_value_with_ref', None)
    if is_shape:
        x_shape, x_val = parse_shape_or_val(x_shape_or_val)
    if is_shape:
        z_mx = K.eval(
            getattr(K, function_name)(K.variable(x_val), **kwargs))
        z_tf = KTF.eval(
            getattr(KTF, function_name)(KTF.variable(x_val), **kwargs))
    else:
        z_mx = K.eval(
            getattr(K, function_name)(x_shape_or_val, **kwargs))
        z_tf = KTF.eval(
            getattr(KTF, function_name)(x_shape_or_val, **kwargs))
    assert_allclose(z_mx, z_tf, atol=1e-05)
    if assert_value_with_ref is not None:
        assert_allclose(z_mx, assert_value_with_ref, atol=1e-05)


@tensorflow_clean_session
def check_two_tensor_operation(function_name, x_shape_or_val, y_shape_or_val, **kwargs):
    """
    :param function_name:
        The function in the backend
    :param x_shape_or_val:
        It can be shape of the numpy array or actual numpy array
    :param y_shape_or_val:
        It can be shape of the numpy array or actual numpy array
    :param kwargs:
        Some arguments that the backend function need
    :return:
    """
    is_shape = kwargs.pop('is_shape', True)
    if is_shape:
        x_shape, x_val = parse_shape_or_val(x_shape_or_val)
        y_shape, y_val = parse_shape_or_val(y_shape_or_val)
    if is_shape:
        z_mx = K.eval(
            getattr(K, function_name)(K.variable(x_val), K.variable(y_val), **kwargs))
        z_tf = KTF.eval(
            getattr(KTF, function_name)(KTF.variable(x_val), KTF.variable(y_val), **kwargs))
    else:
        z_mx = K.eval(
            getattr(K, function_name)(x_shape_or_val, y_shape_or_val, **kwargs))
        z_tf = K.eval(
            getattr(KTF, function_name)(x_shape_or_val, y_shape_or_val, **kwargs))
    # here z_mx and z_tf are the output numpy array
    assert_allclose(z_mx, z_tf, atol=1e-05)


class TestKerasMXNet(object):
    def test_symbol_addtion(self):
        symbol1 = K.placeholder(shape=(2,))
        symbol2 = K.placeholder(shape=(2,))
        symbol3 = symbol1 + symbol2
        assert symbol1 in symbol3.get_neighbor()
        assert symbol2 in symbol3.get_neighbor()
        assert symbol3 in symbol1.get_neighbor()
        assert symbol3 in symbol2.get_neighbor()

    def test_variable_addition(self):
        var1 = K.variable([1, 2, 3])
        var2 = K.variable([2, 3, 4])
        var3 = var1 + var2
        assert var1 in var3.get_neighbor()
        assert var2 in var3.get_neighbor()
        assert var3 in var1.get_neighbor()
        assert var3 in var2.get_neighbor()
        assert_array_equal(K.eval(var3), np.array([3, 5, 7]))

    def tset_single_operator_linear_operation(self):
        check_single_tensor_operation('transpose', (4, 2))
        check_single_tensor_operation('reverse', (4, 3, 2), axes=1)
        check_single_tensor_operation('reverse', (4, 3, 2), axes=(1, 2))

    # TODO: batch_dot is not correct, now the argument becomes axes instead of dim
    @pytest.mark.skip
    def test_two_operators_linear_operations(self):
        check_two_tensor_operation('dot', (4, 2), (2, 4))
        check_two_tensor_operation('dot', (4, 2), (5, 2, 3))
        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 5, 3), axes=(2, 2))
        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 3), axes=(2, 1))
        check_two_tensor_operation('batch_dot', (4, 2), (4, 2, 3), axes=(1, 1))
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20), axes=1)
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20), axes=(1, 1))

    def test_none_shape_operations(self):
        pass

    def test_repeat_elements(self):
        reps = 3
        for ndims in [1, 2, 3]:
            shape = np.arange(2, 2 + ndims)
            arr = np.arange(np.prod(shape)).reshape(shape)

            for rep_axis in range(ndims):
                np_rep = np.repeat(arr, reps, axis=rep_axis)
                check_single_tensor_operation('repeat_elements', arr,
                                              rep=reps, axis=rep_axis,
                                              assert_value_with_ref=np_rep)

    def test_tile(self):
        shape = (3, 4)
        arr = np.arange(np.prod(shape)).reshape(shape)
        check_single_tensor_operation('tile', arr, n=[2, 1])
        check_single_tensor_operation('tile', (2, 5), n=[5, 2])

    @tensorflow_clean_session
    def test_gather(self):
        shape = (10, 2, 3)
        ref = np.arange(np.prod(shape)).reshape(shape)
        inds = [1, 3, 7, 9]
        z_mx = K.eval(K.gather(K.variable(ref), K.variable(inds, dtype='int32')))
        z_tf = KTF.eval(KTF.gather(KTF.variable(ref), KTF.variable(inds, dtype='int32')))
        assert_allclose(z_mx, z_tf)

    def test_value_manipulation(self):
        pass


    def test_elementwise_operations(self):
        check_single_tensor_operation('max', (4, 2))
        check_single_tensor_operation('max', (4, 2), axis=1, keepdims=True)

        check_single_tensor_operation('min', (4, 2))
        check_single_tensor_operation('min', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('min', (4, 2, 3), axis=[1, -1])

        check_single_tensor_operation('mean', (4, 2))
        check_single_tensor_operation('mean', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), axis=-1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), axis=[1, -1])

        check_single_tensor_operation('std', (4, 2))
        check_single_tensor_operation('std', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('std', (4, 2, 3), axis=[1, -1])

        check_single_tensor_operation('prod', (4, 2))
        check_single_tensor_operation('prod', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('prod', (4, 2, 3), axis=[1, -1])

        check_single_tensor_operation('any', (4, 2))
        check_single_tensor_operation('any', (4, 2), axis=1, keepdims=True)

        check_single_tensor_operation('all', (4, 2))
        check_single_tensor_operation('all', (4, 2), axis=1, keepdims=True)

        check_single_tensor_operation('argmax', (4, 2))
        check_single_tensor_operation('argmax', (4, 2), axis=1)

        check_single_tensor_operation('argmin', (4, 2))
        check_single_tensor_operation('argmin', (4, 2), axis=1)

        check_single_tensor_operation('square', (4, 2))
        check_single_tensor_operation('abs', (4, 2))
        check_single_tensor_operation('sqrt', (4, 2))
        check_single_tensor_operation('exp', (4, 2))
        check_single_tensor_operation('log', (4, 2))
        check_single_tensor_operation('round', (4, 2))
        check_single_tensor_operation('sign', (4, 2))
        check_single_tensor_operation('pow', (4, 2), a=3)
        check_single_tensor_operation('clip', (4, 2), min_value=0.4, max_value=0.6)

        # two-tensor ops
        check_two_tensor_operation('equal', (4, 2), (4, 2))
        check_two_tensor_operation('not_equal', (4, 2), (4, 2))
        check_two_tensor_operation('greater', (4, 2), (4, 2))
        check_two_tensor_operation('greater_equal', (4, 2), (4, 2))
        check_two_tensor_operation('less', (4, 2), (4, 2))
        check_two_tensor_operation('less_equal', (4, 2), (4, 2))
        check_two_tensor_operation('maximum', (4, 2), (4, 2))
        check_two_tensor_operation('minimum', (4, 2), (4, 2))

    def test_gradient(self):
        pass

    def test_stop_gradient(self):
        pass

    def test_function(self):
        pass


if __name__ == '__main__':
    pytest.main([__file__])
