import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sparse
import warnings

from keras import backend as K
from keras.utils.conv_utils import convert_kernel
from keras.backend import numpy_backend as KNP


try:
    from keras.backend import cntk_backend as KC
except ImportError:
    KC = None
    warnings.warn('Could not import the CNTK backend')

try:
    from keras.backend import tensorflow_backend as KTF
except ImportError:
    KTF = None
    warnings.warn('Could not import the TensorFlow backend.')

try:
    from keras.backend import theano_backend as KTH
except ImportError:
    KTH = None
    warnings.warn('Could not import the Theano backend')

if K.backend() == 'theano':
    WITH_NP = [KTH, KNP]
elif K.backend() == 'cntk':
    WITH_NP = [KC, KNP]
else:
    WITH_NP = [KTF, KNP]

if K.backend() == 'cntk':
    supports_sparse = False
elif K.backend() == 'theano' and not KTH.th_sparse_module:
    supports_sparse = False
elif K.backend() == 'tensorflow':
    # Must wait for tf.keras to support sparse ops.
    supports_sparse = False
else:
    supports_sparse = True


def check_dtype(var, dtype):
    if K.backend() == 'tensorflow':
        assert dtype in str(var.dtype.name)
    else:
        assert dtype in str(var.dtype)


def cntk_func_tensors(function_name, shapes_or_vals, **kwargs):
    placeholders = []
    variables = []
    for shape_or_val in shapes_or_vals:
        if isinstance(shape_or_val, tuple):
            shape = shape_or_val
            placeholders.append(KC.placeholder(shape))
        else:
            value = shape_or_val
            variables.append(KC.variable(value))

    output_cntk = getattr(KC, function_name)(*(placeholders + variables), **kwargs)
    cntk_func = KC.function(placeholders, [output_cntk])
    return output_cntk, cntk_func


def parse_shape_or_val(shape_or_val):
    if isinstance(shape_or_val, np.ndarray):
        return shape_or_val.shape, shape_or_val
    else:
        return shape_or_val, np.random.random(shape_or_val).astype(np.float32) - 0.5


def assert_list_pairwise(z_list,
                         shape=True,
                         allclose=True,
                         itself=False,
                         atol=1e-05):
    for (z1, z2) in zip(z_list[1:], z_list[:-1]):
        if shape:
            assert z1.shape == z2.shape
        if allclose:
            assert_allclose(z1, z2, atol=atol)
        if itself:
            assert z1 == z2


def assert_list_keras_shape(t_list, z_list):
    for t, z in zip(t_list, z_list):
        if hasattr(t, '_keras_shape') and len(t._keras_shape) > 1:
            for i, s in enumerate(t._keras_shape):
                if s:
                    assert t._keras_shape[i] == z.shape[i]


def check_single_tensor_operation(function_name,
                                  x_shape_or_val,
                                  backend_list,
                                  **kwargs):
    shape_or_val = kwargs.pop('shape_or_val', True)
    assert_value_equality = kwargs.pop('assert_value_equality', True)
    cntk_dynamicity = kwargs.pop('cntk_dynamicity', False)

    if shape_or_val:
        x_shape, x_val = parse_shape_or_val(x_shape_or_val)

    t_list = []
    z_list = []
    for k in backend_list:
        if shape_or_val:
            if (k == KC) & (cntk_dynamicity):
                t, f = cntk_func_tensors(function_name, [x_shape], **kwargs)
                z = f([x_val])[0]
            else:
                t = getattr(k, function_name)(k.variable(x_val), **kwargs)
                z = k.eval(t)
        else:
            t = getattr(k, function_name)(x_shape_or_val, **kwargs)
            z = k.eval(t)
        t_list += [t]
        z_list += [z]

    assert_list_pairwise(z_list, allclose=assert_value_equality)
    assert_list_keras_shape(t_list, z_list)


def check_two_tensor_operation(function_name,
                               x_shape_or_val,
                               y_shape_or_val,
                               backend_list,
                               **kwargs):
    concat_args = kwargs.pop('concat_args', False)
    cntk_dynamicity = kwargs.pop('cntk_dynamicity', False)
    cntk_two_dynamicity = kwargs.pop('cntk_two_dynamicity', False)

    x_shape, x_val = parse_shape_or_val(x_shape_or_val)
    y_shape, y_val = parse_shape_or_val(y_shape_or_val)

    t_list = []
    z_list = []
    for k in backend_list:
        if (k == KC) & (cntk_dynamicity):
            t, f = cntk_func_tensors(function_name, [x_shape, y_val], **kwargs)
            z = f([x_val])[0]
        elif (k == KC) & (cntk_two_dynamicity):
            t, f = cntk_func_tensors(function_name, [x_shape, y_shape], **kwargs)
            z = f([x_val, y_val])[0]
        elif (k == KTH) & (function_name[:4] == 'conv'):
            t = getattr(k, function_name)(
                k.variable(x_val), k.variable(convert_kernel(y_val)), **kwargs)
            z = k.eval(t)
        elif concat_args:
            t = getattr(k, function_name)(
                [k.variable(x_val), k.variable(y_val)], **kwargs)
            z = k.eval(t)
        else:
            t = getattr(k, function_name)(
                k.variable(x_val), k.variable(y_val), **kwargs)
            z = k.eval(t)
        t_list += [t]
        z_list += [z]

    assert_list_pairwise(z_list)
    assert_list_keras_shape(t_list, z_list)


def check_composed_tensor_operations(first_function_name,
                                     first_function_args,
                                     second_function_name,
                                     second_function_args,
                                     input_shape,
                                     backend_list):
    val = np.random.random(input_shape) - 0.5

    z_list = []
    for k in backend_list:
        x = k.variable(val)
        y = getattr(k, first_function_name)(x, **first_function_args)
        z = k.eval(getattr(k, second_function_name)(y, **second_function_args))
        z_list += [z]

    assert_list_pairwise(z_list)


def check_rnn_operation(step_function_k,
                        step_function_np,
                        inputs_np,
                        initial_states_np,
                        mask_np=None,
                        constants_np=None,
                        **kwargs):
    inputs_k = K.variable(inputs_np)
    initial_states_k = [K.variable(s) for s in initial_states_np]
    if mask_np is not None:
        mask_k = K.variable(mask_np)
    else:
        mask_k = None
    if constants_np is not None:
        constants_k = [K.variable(c) for c in constants_np]
    else:
        constants_k = None

    last_output_np, output_np, last_states_np = KNP.rnn(
        step_function_np,
        inputs_np,
        initial_states_np,
        mask=mask_np,
        constants=constants_np,
        **kwargs)
    # note that numpy reference implementation is independent of `unroll` argument
    if 'unroll' in kwargs:
        unroll_options = [kwargs.pop('unroll')]
    else:
        unroll_options = [True, False]
    for unroll in unroll_options:
        last_output_k, output_k, last_states_k = K.rnn(
            step_function_k,
            inputs_k,
            initial_states_k,
            mask=mask_k,
            constants=constants_k,
            unroll=unroll,
            input_length=inputs_np.shape[1] if unroll else None,
            **kwargs)

        last_states_k = [K.eval(s) for s in last_states_k]
        last_output_k = K.eval(last_output_k)
        output_k = K.eval(output_k)

        assert_allclose(last_output_k, last_output_np, atol=1e-05)
        assert_allclose(output_k, output_np, atol=1e-05)
        assert len(last_states_k) == len(last_states_np)
        for s_k, s_np in zip(last_states_k, last_states_np):
            assert_allclose(s_k, s_np, atol=1e-05)


class TestBackend(object):

    def test_is_keras_tensor(self):
        np_var = np.array([1, 2])
        with pytest.raises(ValueError):
            K.is_keras_tensor(np_var)

        keras_var = K.variable(np_var)
        assert K.is_keras_tensor(keras_var) is False
        keras_placeholder = K.placeholder(shape=(2, 4, 5))
        assert K.is_keras_tensor(keras_placeholder) is False

    def test_set_learning_phase(self):
        # not supported learning_phase
        with pytest.raises(ValueError):
            K.set_learning_phase(2)

    def test_creation_operations(self):
        check_single_tensor_operation('eye', 3, WITH_NP, shape_or_val=False)
        check_single_tensor_operation('eye', (3, 2), WITH_NP, shape_or_val=False)
        check_single_tensor_operation('eye', (3, 4), WITH_NP, shape_or_val=False)

        check_single_tensor_operation('ones', (3, 5, 10, 8),
                                      WITH_NP, shape_or_val=False)
        check_single_tensor_operation('zeros', (3, 5, 10, 8),
                                      WITH_NP, shape_or_val=False)

        check_single_tensor_operation('ones_like', (3, 5, 10, 8), WITH_NP)
        check_single_tensor_operation('zeros_like', (3, 5, 10, 8), WITH_NP)

    def test_linear_operations(self):
        check_two_tensor_operation('dot', (4, 2), (2, 4), WITH_NP)
        check_two_tensor_operation('dot', (4, 2), (5, 2, 3), WITH_NP)

        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 5, 3),
                                   WITH_NP, cntk_two_dynamicity=True, axes=(2, 2))
        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 3),
                                   WITH_NP, cntk_two_dynamicity=True, axes=(2, 1))
        check_two_tensor_operation('batch_dot', (4, 2), (4, 2, 3),
                                   WITH_NP, cntk_two_dynamicity=True, axes=(1, 1))
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20),
                                   WITH_NP, cntk_two_dynamicity=True, axes=1)
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20),
                                   WITH_NP, cntk_two_dynamicity=True, axes=(1, 1))
        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 5, 3),
                                   WITH_NP, axes=(2, 2))
        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 3),
                                   WITH_NP, axes=(2, 1))
        check_two_tensor_operation('batch_dot', (4, 2), (4, 2, 3),
                                   WITH_NP, axes=(1, 1))
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20),
                                   WITH_NP, axes=1)
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20),
                                   WITH_NP, axes=(1, 1))

        check_single_tensor_operation('transpose', (4, 2), WITH_NP)
        check_single_tensor_operation('reverse', (4, 3, 2), WITH_NP, axes=1)
        check_single_tensor_operation('reverse', (4, 3, 2), WITH_NP, axes=(1, 2))
        check_single_tensor_operation('reverse', (4, 3, 2), WITH_NP, axes=(0, -1))

    def test_random_variables(self):
        check_single_tensor_operation('random_uniform_variable', (2, 3), WITH_NP,
                                      low=0., high=1.,
                                      shape_or_val=False,
                                      assert_value_equality=False)
        check_single_tensor_operation('random_normal_variable', (2, 3), WITH_NP,
                                      mean=0., scale=1.,
                                      shape_or_val=False,
                                      assert_value_equality=False)

    def test_batch_dot_shape(self):
        # Note : batch_dot implementation is different for
        # placeholders and variables in CNTK backend

        test_cases = []
        test_cases.append([(None, 3, 4, 5), (None, 2, 3, 4), (2, 3)])
        test_cases.append([(None, 3, 4, 5), (None, 2, 4), 2])
        test_cases.append([(None, 3, 4), (None, 2, 3, 4), (2, 3)])
        test_cases.append([(None, 4, 3), (None, 3, 5), (2, 1)])
        test_cases.append([(None, 4), (None, 3, 4), (1, 2)])
        test_cases.append([(None, 4), (None, 4), None])

        batch_size = 7

        def batch_shape(shape):
            return (batch_size, ) + shape[1:]

        def random(shape):
            return np.random.random(batch_shape(shape))

        for x_shape, y_shape, axes in test_cases:
            x_np = random(x_shape)
            y_np = random(y_shape)
            z_np = KNP.batch_dot(x_np, y_np, axes)

            # test with placeholders
            x = K.placeholder(shape=x_shape)
            y = K.placeholder(shape=y_shape)
            z = K.batch_dot(x, y, axes)

            z_shape = K.int_shape(z)
            if z_shape is not None:
                assert z_shape[1:] == z_np.shape[1:]

            f = K.function([x, y], [z])

            assert_allclose(f([x_np, y_np])[0], z_np, atol=1e-05)

            # test with placeholders (no shape info)
            if K.backend() != 'cntk':
                x = K.placeholder(ndim=len(x_shape))
                y = K.placeholder(ndim=len(y_shape))
                z = K.batch_dot(x, y, axes)

                z_shape = K.int_shape(z)
                if z_shape is not None:
                    assert len(z_shape) == z_np.ndim
                    assert set(z_shape) <= set((None, 1))

                f = K.function([x, y], [z])

                assert_allclose(f([x_np, y_np])[0], z_np, atol=1e-05)

            # test with variables
            x = K.variable(x_np)
            y = K.variable(y_np)
            z = K.batch_dot(x, y, axes)

            z_shape = K.int_shape(z)
            if z_shape is not None:
                assert z_shape[1:] == z_np.shape[1:]

            z = K.eval(z)
            assert_allclose(z, z_np, atol=1e-05)

    def test_shape_operations(self):
        check_single_tensor_operation('reshape', (4, 2), WITH_NP, shape=(8, 1))
        check_single_tensor_operation('permute_dimensions', (4, 2, 3), WITH_NP,
                                      pattern=(2, 0, 1))
        check_single_tensor_operation('repeat', (4, 1), WITH_NP, n=3)
        check_single_tensor_operation('flatten', (4, 1), WITH_NP)
        check_single_tensor_operation('batch_flatten', (20, 2, 5), WITH_NP,
                                      cntk_dynamicity=True)
        check_single_tensor_operation('expand_dims', (4, 3), WITH_NP, axis=-1)
        check_single_tensor_operation('expand_dims', (4, 3, 2), WITH_NP, axis=1)
        check_single_tensor_operation('squeeze', (4, 3, 1), WITH_NP, axis=2)
        check_single_tensor_operation('squeeze', (4, 1, 1), WITH_NP, axis=1)
        check_composed_tensor_operations('reshape', {'shape': (4, 3, 1, 1)},
                                         'squeeze', {'axis': 2},
                                         (4, 3, 1, 1), WITH_NP)

    @pytest.mark.skipif(K.backend() != 'theano',
                        reason='We only test the shape inference of the '
                               'theano backend.')
    def test_none_shape_operations(self):
        # Test shape inference when input
        # shape has `None` entries
        x = K.placeholder((3, None, 4))

        y = K.batch_flatten(x)
        if hasattr(y, '_keras_shape'):
            assert y._keras_shape == (3, None)

        y = K.flatten(x)
        if hasattr(y, '_keras_shape'):
            assert y._keras_shape == (None,)

    def test_repeat_elements(self):
        reps = 3
        for ndims in [1, 2, 3]:
            shape = np.arange(2, 2 + ndims)
            arr = np.arange(np.prod(shape)).reshape(shape)

            for rep_axis in range(ndims):
                check_single_tensor_operation('repeat_elements', arr, WITH_NP,
                                              rep=reps, axis=rep_axis)

                if K.backend() != 'cntk':
                    shape = list(shape)
                    shape[rep_axis] = None
                    x = K.placeholder(shape=shape)
                    y = K.repeat_elements(x, reps, axis=rep_axis)
                    assert y._keras_shape == tuple(shape)
                    assert y._keras_shape == K.int_shape(y)

    def test_tile(self):
        check_single_tensor_operation('tile', (3, 4), WITH_NP, n=2)
        check_single_tensor_operation('tile', (3, 4), WITH_NP, n=(2, 1))
        check_single_tensor_operation('tile', (3, 4, 5), WITH_NP, n=2)
        check_single_tensor_operation('tile', (3, 4, 5), WITH_NP, n=(1, 2))
        check_single_tensor_operation('tile', (3, 4, 5), WITH_NP, n=(3, 1, 2))

        # test theano shape inference when
        # input shape has None entries
        if K.backend() == 'theano':
            x = K.placeholder(shape=(None, 4))
            n = 2
            y = K.tile(x, n)
            assert y._keras_shape == (None, 8)
            n = (4, 3)
            y = K.tile(x, n)
            assert y._keras_shape == (None, 12)

    def test_gather(self):
        shape = (10, 2, 3)
        ref = np.arange(np.prod(shape)).reshape(shape)
        inds = [1, 3, 7, 9]
        t_list = [k.gather(k.variable(ref), k.variable(inds, dtype='int32'))
                  for k in WITH_NP]
        z_list = [k.eval(k.gather(k.variable(ref), k.variable(inds, dtype='int32')))
                  for k in WITH_NP]

        assert_list_pairwise(z_list)
        assert_list_keras_shape(t_list, z_list)

        # test theano shape inference when
        # input shape has None entries
        if K.backend() == 'theano':
            x = K.placeholder(shape=(None, 3, 4))
            indices = K.placeholder(shape=(5, 6), dtype='int32')
            y = K.gather(x, indices)
            assert y._keras_shape == (5, 6, 3, 4)

    @pytest.mark.parametrize('function_name',
                             ['get_value', 'count_params',
                              'int_shape', 'get_variable_shape'])
    def test_value_manipulation(self, function_name):
        val = np.random.random((4, 2))
        v_list = [getattr(k, function_name)(k.variable(val))
                  for k in WITH_NP]

        if function_name == 'get_value':
            assert_list_pairwise(v_list)
        else:
            assert_list_pairwise(v_list,
                                 shape=False,
                                 allclose=False,
                                 itself=True)

    def test_print_tensor(self, capsys):
        # TODO: somehow this capture mechanism doesn't work for TF
        # even though the TF op does print to stdout.
        for k in [KTH]:
            x = k.placeholder((1, 1))
            y = k.print_tensor(x, 'msg')
            fn = k.function([x], [y])
            _ = fn([np.ones((1, 1))])
            out, err = capsys.readouterr()
            # Theano inserts "__str__ = " for no good reason
            assert out.replace('__str__ = ', '') == 'msg [[1.]]\n'

    def test_elementwise_operations(self):
        check_single_tensor_operation('max', (4, 2), WITH_NP)
        check_single_tensor_operation('max', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('max', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('min', (4, 2), WITH_NP)
        check_single_tensor_operation('min', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('min', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('mean', (4, 2), WITH_NP)
        check_single_tensor_operation('mean', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3),
                                      WITH_NP, axis=-1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('var', (4, 2), WITH_NP)
        check_single_tensor_operation('var', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('var', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('std', (4, 2), WITH_NP)
        check_single_tensor_operation('std', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('std', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('prod', (4, 2), WITH_NP)
        check_single_tensor_operation('prod', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('prod', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('any', (4, 2), WITH_NP)
        check_single_tensor_operation('any', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('any', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('all', (4, 2), WITH_NP)
        check_single_tensor_operation('all', (4, 2), WITH_NP, axis=1, keepdims=True)
        check_single_tensor_operation('all', (4, 2, 3), WITH_NP, axis=[1, -1])

        check_single_tensor_operation('argmax', (4, 2), WITH_NP)
        check_single_tensor_operation('argmax', (4, 2), WITH_NP, axis=1)

        check_single_tensor_operation('argmin', (4, 2), WITH_NP)
        check_single_tensor_operation('argmin', (4, 2), WITH_NP, axis=1)

        check_single_tensor_operation('square', (4, 2), WITH_NP)
        check_single_tensor_operation('abs', (4, 2), WITH_NP)
        check_single_tensor_operation('sqrt', (4, 2), WITH_NP)
        check_single_tensor_operation('exp', (4, 2), WITH_NP)

        check_single_tensor_operation('round', (4, 2), WITH_NP)
        check_single_tensor_operation('sign', (4, 2), WITH_NP)
        check_single_tensor_operation('pow', (4, 2), WITH_NP, a=3)
        check_single_tensor_operation('clip', (4, 2), WITH_NP, min_value=0.4,
                                      max_value=0.6)

        check_single_tensor_operation('cos', (4, 2), WITH_NP)
        check_single_tensor_operation('sin', (4, 2), WITH_NP)

        # two-tensor ops
        check_two_tensor_operation('equal', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('not_equal', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('greater', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('greater_equal', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('less', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('less_equal', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('maximum', (4, 2), (4, 2), WITH_NP)
        check_two_tensor_operation('minimum', (4, 2), (4, 2), WITH_NP)

    # assumes first uid will always be the same
    def test_reset_uids(self):
        first = K.get_uid()
        K.get_uid()
        K.reset_uids()
        assert K.get_uid() == first

    def test_cumsum(self):
        check_single_tensor_operation('cumsum', (4, 2), WITH_NP)
        check_single_tensor_operation('cumsum', (4, 2), WITH_NP, axis=1)

    def test_cumprod(self):
        check_single_tensor_operation('cumprod', (4, 2), WITH_NP)
        check_single_tensor_operation('cumprod', (4, 2), WITH_NP, axis=1)

    @pytest.mark.skipif(K.backend() == 'cntk',
                        reason='cntk return -85.1 for zero or '
                               'negative number, not nan, so can\'t '
                               'compare with other backend.')
    def test_log(self):
        check_single_tensor_operation('log', (4, 2), WITH_NP)

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='theano returns tuples for updates; cntk buggy')
    def test_update(self):
        x = np.ones((3, 4))
        x_var = K.variable(x)
        new_x = np.random.random((3, 4))

        op = K.update(x_var, new_x)
        K.eval(op)

        assert_allclose(new_x, K.eval(x_var), atol=1e-05)

    @pytest.mark.skipif(K.backend() == 'theano',
                        reason='theano returns tuples for update ops')
    def test_update_add(self):
        x = np.ones((3, 4))
        x_var = K.variable(x)
        increment = np.random.random((3, 4))

        op = K.update_add(x_var, increment)
        K.eval(op)

        assert_allclose(x + increment, K.eval(x_var), atol=1e-05)

    @pytest.mark.skipif(K.backend() == 'theano',
                        reason='theano returns tuples for update ops')
    def test_update_sub(self):
        x = np.ones((3, 4))
        x_var = K.variable(x)
        decrement = np.random.random((3, 4))

        op = K.update_sub(x_var, decrement)
        K.eval(op)

        assert_allclose(x - decrement, K.eval(x_var), atol=1e-05)

    @pytest.mark.skipif(K.backend() == 'cntk',
                        reason='cntk doesn\'t support gradient in this way.')
    def test_gradient(self):
        val = np.random.random((4, 2))
        x_list = [k.placeholder(shape=(4, 2)) for k in [KTH, KTF]]
        z_list = []
        zero_list = []
        for x, k in zip(x_list, [KTH, KTF]):
            exp = x * k.exp(x)
            loss = k.sum(exp)
            zero_loss = k.stop_gradient(loss)
            grad = k.gradients(loss, [exp])

            zero_grad = k.gradients(loss + zero_loss, [exp])
            grad_eval_fn = k.function([x], [grad[0]])
            zero_grad_eval_fn = k.function([x], [zero_grad[0]])
            z_list.append(grad_eval_fn([val])[0])
            zero_list.append(zero_grad_eval_fn([val])[0])

        assert_list_pairwise(z_list)
        assert_list_pairwise(zero_list)
        for i in range(len(z_list)):
            assert_allclose(zero_list[i], z_list[i], atol=1e-05)

    def test_stop_gradient(self):
        # This test checks the consistency of the stop_gradient backend API.
        # It doesn't check the functionality (which is checked at the
        # test_gradient test).
        val = np.random.random((4, 2))
        a = K.variable(val)
        b = K.square(a)
        c, d = K.stop_gradient([a, b])
        e = K.stop_gradient(b)

    @pytest.mark.skipif(K.backend() == 'cntk',
                        reason='cntk currently not support function in this '
                               'way, so can\'t test as this.')
    def test_function(self):
        test_backend = [KTH, KTF]
        val = np.random.random((4, 2))
        input_val = np.random.random((4, 2))

        f_list = []
        x_list = []
        for k in test_backend:
            x = k.variable(val)
            x_list.append(x)
            y = k.placeholder(ndim=2)
            exp = k.square(x) + y
            # Need to use `identity` to make this symbolic
            # (TODO: fix in tf.keras)
            update = k.identity(x) * 2
            f = k.function([y], [exp], updates=[(x, update)])
            f_list.append(f)

        function_outputs_list = [f([input_val])[0] for f in f_list]
        assert_list_pairwise(function_outputs_list)

        new_val_list = [k.get_value(x) for x, k in zip(x_list, test_backend)]
        assert_list_pairwise(new_val_list)

    @pytest.mark.skipif(K.backend() != 'tensorflow' or not KTF._is_tf_1(),
                        reason='Uses the `fetches` argument.')
    def test_function_tf_fetches(self):
        # Additional operations can be passed to tf.Session().run() via its
        # `fetches` arguments. In contrast to `updates` argument of
        # KTF.function() these do not have control dependency on `outputs`, so
        # they can run in parallel. Also they should not contribute to output of
        # KTF.function().

        x = K.variable(0.)
        y = K.variable(0.)
        x_placeholder = K.placeholder(shape=())
        y_placeholder = K.placeholder(shape=())

        f = K.function(inputs=[x_placeholder, y_placeholder],
                       outputs=[x_placeholder + y_placeholder],
                       updates=[(x, x_placeholder + 1.)],
                       fetches=[K.update(y, 5.)])
        output = f([10., 20.])
        assert output == [30.]
        assert K.get_session().run(fetches=[x, y]) == [11., 5.]

    @pytest.mark.skipif(K.backend() != 'tensorflow' or not KTF._is_tf_1(),
                        reason='Uses the `feed_dict` argument.')
    def test_function_tf_feed_dict(self):
        # Additional substitutions can be passed to `tf.Session().run()` via its
        # `feed_dict` arguments. Note that the feed_dict is passed once in the
        # constructor but we can modify the values in the dictionary. Through
        # this feed_dict we can provide additional substitutions besides Keras
        # inputs.

        x = K.variable(0.)
        y = K.variable(0.)
        x_placeholder = K.placeholder(shape=())
        y_placeholder = K.placeholder(shape=())

        feed_dict = {y_placeholder: 3.}

        f = K.function(inputs=[x_placeholder],
                       outputs=[x_placeholder + 1.],
                       updates=[(x, x_placeholder + 10.)],
                       feed_dict=feed_dict,
                       fetches=[K.update(y, y_placeholder * 10.)])
        output = f([10.])
        assert output == [11.]
        assert K.get_session().run(fetches=[x, y]) == [20., 30.]

        # updated value in feed_dict will be modified within the K.function()
        feed_dict[y_placeholder] = 4.
        output = f([20.])
        assert output == [21.]
        assert K.get_session().run(fetches=[x, y]) == [30., 40.]

    @pytest.mark.skipif(K.backend() != 'tensorflow' or not KTF._is_tf_1(),
                        reason='Uses the `options` and `run_metadata` arguments.')
    def test_function_tf_run_options_with_run_metadata(self):
        from tensorflow.core.protobuf import config_pb2
        x_placeholder = K.placeholder(shape=())
        y_placeholder = K.placeholder(shape=())

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        # enable run_options.
        f = K.function(inputs=[x_placeholder, y_placeholder],
                       outputs=[x_placeholder + y_placeholder],
                       options=run_options,
                       run_metadata=run_metadata)
        output = f([10., 20.])
        assert output == [30.]
        assert len(run_metadata.partition_graphs) > 0
        # disable run_options.
        f = K.function(inputs=[x_placeholder, y_placeholder],
                       outputs=[x_placeholder + y_placeholder],
                       run_metadata=run_metadata)
        output = f([10., 20.])
        assert output == [30.]
        assert len(run_metadata.partition_graphs) == 0

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='Uses the `string` type for a tensor.')
    def test_function_tf_string_input(self):
        # Test functions with string inputs.

        x_placeholder = K.placeholder(shape=(), dtype="string")
        x_identity = K.identity(x_placeholder)

        f = K.function(inputs=[x_placeholder], outputs=[x_identity])
        output = f([b'test'])
        assert output == [b'test']

    def test_rnn(self):
        # implement a simple RNN
        num_samples = 4
        input_dim = 5
        output_dim = 3
        timesteps = 6

        _, x = parse_shape_or_val((num_samples, timesteps, input_dim))
        _, h0 = parse_shape_or_val((num_samples, output_dim))
        _, wi = parse_shape_or_val((input_dim, output_dim))
        _, wh = parse_shape_or_val((output_dim, output_dim))
        mask = np.random.randint(2, size=(num_samples, timesteps))

        wi_k = K.variable(wi)
        wh_k = K.variable(wh)

        def get_step_function(backend, w_i, w_h):

            def simple_rnn(inputs, states):
                assert len(states) == 1
                h = states[0]
                y = backend.dot(inputs, w_i) + backend.dot(h, w_h)
                return y, [y]

            return simple_rnn

        kwargs_list = [
            {'go_backwards': False, 'mask': None},
            {'go_backwards': True, 'mask': None},
            {'go_backwards': False, 'mask': mask},
            {'go_backwards': True, 'mask': mask},
        ]
        for kwargs in kwargs_list:
            check_rnn_operation(step_function_k=get_step_function(K, wi_k, wh_k),
                                step_function_np=get_step_function(KNP, wi, wh),
                                inputs_np=x,
                                initial_states_np=[h0],
                                mask_np=kwargs.pop('mask', None),
                                **kwargs)

    @pytest.mark.skipif(K.backend() == 'theano', reason='Not supported')
    def test_rnn_unroll_with_len_1(self):
        num_samples = 4
        input_dim = 5
        output_dim = 3

        _, x = parse_shape_or_val((num_samples, 1, input_dim))
        _, h0 = parse_shape_or_val((num_samples, output_dim))
        _, wi = parse_shape_or_val((input_dim, output_dim))
        _, wh = parse_shape_or_val((output_dim, output_dim))

        wi_k = K.variable(wi)
        wh_k = K.variable(wh)

        def get_step_function(backend, w_i, w_h):

            def simple_rnn(inputs, states):
                assert len(states) == 1
                h = states[0]
                y = backend.dot(inputs, w_i) + backend.dot(h, w_h)
                return y, [y]

            return simple_rnn

        kwargs_list = [
            {'go_backwards': False},
            {'go_backwards': True},
        ]
        for kwargs in kwargs_list:
            check_rnn_operation(step_function_k=get_step_function(K, wi_k, wh_k),
                                step_function_np=get_step_function(KNP, wi, wh),
                                inputs_np=x,
                                initial_states_np=[h0],
                                unroll=True,
                                **kwargs)

    def test_rnn_additional_states(self):
        # implement a simple RNN with an additional state
        # whose shape is different from that of the output
        num_samples = 4
        input_dim = 5
        output_dim = 3
        timesteps = 6

        _, x = parse_shape_or_val((num_samples, timesteps, input_dim))
        _, h0 = parse_shape_or_val((num_samples, output_dim))
        h1 = np.concatenate([h0, h0], axis=-1)
        _, wi = parse_shape_or_val((input_dim, output_dim))
        _, wh = parse_shape_or_val((output_dim, output_dim))
        mask = np.random.randint(2, size=(num_samples, timesteps))

        wi_k = K.variable(wi)
        wh_k = K.variable(wh)

        def get_step_function(backend, w_i, w_h):

            def simple_rnn_with_extra_mock_state(inputs, states):
                assert len(states) == 2
                h = states[0]
                y = backend.dot(inputs, w_i) + backend.dot(h, w_h)
                return y, [y, backend.concatenate([y, y], axis=-1)]

            return simple_rnn_with_extra_mock_state

        kwargs_list = [
            {'go_backwards': False, 'mask': None},
            {'go_backwards': True, 'mask': None},
            {'go_backwards': False, 'mask': mask},
            {'go_backwards': True, 'mask': mask},
        ]
        for kwargs in kwargs_list:
            check_rnn_operation(step_function_k=get_step_function(K, wi_k, wh_k),
                                step_function_np=get_step_function(KNP, wi, wh),
                                inputs_np=x,
                                initial_states_np=[h0, h1],
                                mask_np=kwargs.pop('mask', None),
                                **kwargs)

    def test_rnn_no_states(self):
        # implement a simple RNN without states
        num_samples = 3
        input_dim = 8
        output_dim = 4
        timesteps = 5

        _, x = parse_shape_or_val((num_samples, timesteps, input_dim))
        _, wi = parse_shape_or_val((input_dim, output_dim))
        mask = np.random.randint(2, size=(num_samples, timesteps))

        wi_k = K.variable(wi)

        def get_step_function(backend, w_i):

            def simple_no_states(inputs, states):
                assert len(states) == 0
                y = backend.dot(inputs, w_i)
                return y, []

            return simple_no_states

        kwargs_list = [
            {'go_backwards': False},
            {'go_backwards': True},
        ]
        for kwargs in kwargs_list:
            check_rnn_operation(step_function_k=get_step_function(K, wi_k),
                                step_function_np=get_step_function(KNP, wi),
                                inputs_np=x,
                                initial_states_np=[],
                                mask_np=None,
                                **kwargs)

    def test_rnn_constants(self):
        # implement a simple RNN
        num_samples = 4
        input_dim = 5
        output_dim = 3
        timesteps = 6

        _, x = parse_shape_or_val((num_samples, timesteps, input_dim))
        _, h0 = parse_shape_or_val((num_samples, output_dim))
        _, c = parse_shape_or_val((num_samples, output_dim))
        _, wi = parse_shape_or_val((input_dim, output_dim))
        _, wh = parse_shape_or_val((output_dim, output_dim))
        mask = np.random.randint(2, size=(num_samples, timesteps))

        wi_k = K.variable(wi)
        wh_k = K.variable(wh)

        def get_step_function(backend, w_i, w_h):

            def simple_rnn_add_constant(inputs, states_and_constants):
                # constants are appended to states in K.rnn
                [h, c] = states_and_constants
                y = backend.dot(inputs, w_i) + backend.dot(h, w_h) + c
                return y, [y]

            return simple_rnn_add_constant

        kwargs_list = [
            {'go_backwards': False, 'mask': None},
            {'go_backwards': True, 'mask': None},
            {'go_backwards': False, 'mask': mask},
            {'go_backwards': True, 'mask': mask},
        ]
        for kwargs in kwargs_list:
            check_rnn_operation(step_function_k=get_step_function(K, wi_k, wh_k),
                                step_function_np=get_step_function(KNP, wi, wh),
                                inputs_np=x,
                                initial_states_np=[h0],
                                mask_np=kwargs.pop('mask', None),
                                constants_np=[c],
                                **kwargs)

    def test_rnn_output_and_state_masking_independent(self):
        num_samples = 2
        num_timesteps = 4
        state_and_io_size = 5
        mask_last_num_timesteps = 2  # for second sample only

        # a step function that just outputs inputs,
        # but increments states +1 per timestep
        def step_function(inputs, states):
            return inputs, [s + 1 for s in states]

        inputs_vals = np.random.random(
            (num_samples, num_timesteps, state_and_io_size))
        initial_state_vals = np.random.random((num_samples, state_and_io_size))
        # masking of two last timesteps for second sample only
        mask_vals = np.ones((num_samples, num_timesteps))
        mask_vals[1, -mask_last_num_timesteps:] = 0

        # outputs expected to be same as inputs for the first sample
        expected_outputs = inputs_vals.copy()
        # but for the second sample all outputs in masked region should be the same
        # as last output before masked region
        expected_outputs[1, -mask_last_num_timesteps:] = expected_outputs[
            1, -(mask_last_num_timesteps + 1)]

        expected_state = initial_state_vals.copy()
        # first state should be incremented for every timestep (no masking)
        expected_state[0] += num_timesteps
        # second state should not be incremented for last two timesteps
        expected_state[1] += (num_timesteps - mask_last_num_timesteps)

        # verify same expected output for `unroll=true/false`
        inputs = K.constant(inputs_vals)
        initial_states = [K.constant(initial_state_vals)]
        mask = K.constant(mask_vals)
        for unroll in [True, False]:
            last_output, outputs, last_states = K.rnn(
                step_function,
                inputs,
                initial_states,
                mask=mask,
                unroll=unroll,
                input_length=num_timesteps if unroll else None)

            assert_allclose(K.eval(outputs), expected_outputs)
            assert_allclose(K.eval(last_states[0]), expected_state)

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported')
    def test_rnn_output_num_dim_larger_than_2_masking(self):
        num_samples = 3
        num_timesteps = 4
        num_features = 5

        def step_function(inputs, states):
            outputs = K.tile(K.expand_dims(inputs), [1, 1, 2])
            return outputs, states

        inputs_vals = np.random.random((num_samples, num_timesteps, num_features))
        initial_state_vals = np.random.random((num_samples, 6))
        mask_vals = np.ones((num_samples, num_timesteps))
        mask_vals[-1, -1] = 0  # final timestep masked for last sample

        expected_outputs = np.repeat(inputs_vals[..., None], repeats=2, axis=-1)
        # for the last sample, the final timestep (in masked region) should be the
        # same as the second to final output (before masked region)
        expected_outputs[-1, -1] = expected_outputs[-1, -2]

        inputs = K.constant(inputs_vals)
        initial_states = [K.constant(initial_state_vals)]
        mask = K.constant(mask_vals)
        for unroll in [True, False]:
            last_output, outputs, last_states = K.rnn(
                step_function,
                inputs,
                initial_states,
                mask=mask,
                unroll=unroll,
                input_length=num_timesteps if unroll else None)

            assert_allclose(K.eval(outputs), expected_outputs)

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported')
    def test_rnn_state_num_dim_larger_than_2_masking(self):
        num_samples = 3
        num_timesteps = 4

        def step_function(inputs, states):
            return inputs, [s + 1 for s in states]

        inputs_vals = np.random.random((num_samples, num_timesteps, 5))
        initial_state_vals = np.random.random((num_samples, 6, 7))
        mask_vals = np.ones((num_samples, num_timesteps))
        mask_vals[0, -2:] = 0  # final two timesteps masked for first sample

        expected_last_state = initial_state_vals.copy()
        expected_last_state[0] += (num_timesteps - 2)
        expected_last_state[1:] += num_timesteps

        inputs = K.variable(inputs_vals)
        initial_states = [K.variable(initial_state_vals)]
        mask = K.variable(mask_vals)
        for unroll in [True, False]:
            last_output, outputs, last_states = K.rnn(
                step_function,
                inputs,
                initial_states,
                mask=mask,
                unroll=unroll,
                input_length=num_timesteps if unroll else None)

            # not updated last timestep:
            assert_allclose(K.eval(last_states[0]), expected_last_state)

    @pytest.mark.parametrize('shape', [(3, ), (1, 3), (2, 1), (4, 2), (4, 2, 3)])
    def test_logsumexp(self, shape):
        check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=None)
        check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=0)
        check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=-1)
        check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=-1,
                                      keepdims=True)
        if len(shape) > 1:
            check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=1)
            check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=1,
                                          keepdims=True)
        if len(shape) > 2:
            check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=[1, -1])
            check_single_tensor_operation('logsumexp', shape, WITH_NP, axis=[1, -1],
                                          keepdims=True)

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='The optimization is applied only with TensorFlow.')
    def test_logsumexp_optim(self):
        '''
        Check if optimization works.
        '''
        x_np = np.array([1e+4, 1e-4])
        result = K.eval(K.logsumexp(K.variable(x_np), axis=0))
        assert_allclose(result, 1e4, rtol=1e-5)

    def test_switch(self):
        # scalar
        val = np.random.random()
        z_list = []
        for k in WITH_NP:
            x = k.variable(val)
            x = k.switch(k.greater_equal(x, 0.5), x * 0.1, x * 0.2)
            z_list.append(k.eval(x))
        assert_list_pairwise(z_list)
        # non scalar
        shapes = []
        shapes.append([(4, 3, 2), (4, 3, 2), (4, 3, 2)])
        for s in shapes:
            z_list = []
            arrays = list(map(np.random.random, s))
            for k in WITH_NP:
                x, then_expr, else_expr = map(k.variable, arrays)
                cond = k.greater_equal(x, 0.5)
                z_list.append(k.eval(k.switch(cond, then_expr, else_expr)))
            assert_list_pairwise(z_list)

    def test_dropout(self):
        val = np.random.random((100, 100))
        z_list = [k.eval(k.dropout(k.variable(val), level=0.2))
                  for k in WITH_NP]
        assert_list_pairwise(z_list, allclose=False)
        # dropout patterns are different, only check mean
        for i in range(len(z_list) - 1):
            assert np.abs(z_list[i].mean() - z_list[i + 1].mean()) < 0.05

        z_list = [k.eval(k.dropout(k.variable(val), level=0.2,
                                   noise_shape=list(val.shape)))
                  for k in WITH_NP]
        assert_list_pairwise(z_list, allclose=False)
        # dropout patterns are different, only check mean
        for i in range(len(z_list) - 1):
            assert np.abs(z_list[i].mean() - z_list[i + 1].mean()) < 0.05

        # Test invalid use cases
        with pytest.raises(ValueError):
            z = K.dropout(K.variable(val), level=-0.5)

    @pytest.mark.parametrize('alpha,max_value,threshold', [
        (0.0, None, 0.0),  # standard relu
        (0.1, None, 0.0),  # set alpha only
        (0.0, 5.0, 0.0),   # set max_value only
        (0.0, None, 0.8),  # set threshold only
        (0.1, 5.0, 0.0),   # set alpha and max_value
        (0.1, None, 0.8),  # set alpha and threshold
        (0.0, 5.0, 0.8),   # set max_value and threshold
        (0.1, 5.0, 0.8),   # set all
        (0.1, 0.0, 0.8),   # max_value is zero
        (0.1, 5.0, -2.8),  # threshold is negative
        (0.1, 9.0, 0.8),   # max_value > 6
    ])
    def test_relu(self, alpha, max_value, threshold):
        check_single_tensor_operation('relu', (4, 2), WITH_NP, alpha=alpha,
                                      max_value=max_value, threshold=threshold)

    def test_nn_operations(self):
        check_single_tensor_operation('softsign', (4, 10), WITH_NP)
        check_single_tensor_operation('softplus', (4, 10), WITH_NP)
        check_single_tensor_operation('elu', (4, 10), WITH_NP, alpha=0.5)

        check_single_tensor_operation('sigmoid', (4, 2), WITH_NP)
        check_single_tensor_operation('hard_sigmoid', (4, 2), WITH_NP)
        check_single_tensor_operation('tanh', (4, 2), WITH_NP)

        check_single_tensor_operation('softmax', (4, 10), WITH_NP)
        check_single_tensor_operation('softmax', (4, 5, 3), WITH_NP, axis=1)
        check_single_tensor_operation('softmax', (4, 5, 3, 10), WITH_NP, axis=2)

        check_single_tensor_operation('l2_normalize', (4, 3), WITH_NP, axis=-1)
        check_single_tensor_operation('l2_normalize', (4, 3), WITH_NP, axis=1)

    def test_crossentropy(self):
        # toy label matrix (4 samples, 2 classes)
        label = np.array([[.4, .6], [.3, .7], [.1, .9], [.2, .8]], dtype=np.float32)
        binary_targets = np.array([[.3, .7], [.2, .8], [.4, .6], [.1, .9]],
                                  dtype=np.float32)
        categorical_targets = np.array([[1, 0], [1, 0], [0, 1], [0, 1]],
                                       dtype=np.float32)
        check_two_tensor_operation(
            'binary_crossentropy', label, binary_targets, WITH_NP)
        check_two_tensor_operation('binary_crossentropy', label, (4, 2),
                                   WITH_NP, from_logits=True)
        check_two_tensor_operation(
            'categorical_crossentropy', label, categorical_targets,
            WITH_NP, cntk_two_dynamicity=True)
        check_two_tensor_operation('categorical_crossentropy', label, (4, 2),
                                   WITH_NP, cntk_two_dynamicity=True,
                                   from_logits=True)

        # toy label matrix (2 samples, 3 classes)
        label = np.array([[.4, .1, .5], [.2, .6, .2]], dtype=np.float32)
        categorical_targets = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
        check_two_tensor_operation(
            'categorical_crossentropy', label, categorical_targets,
            WITH_NP, cntk_two_dynamicity=True)
        check_two_tensor_operation('categorical_crossentropy', label, (2, 3),
                                   WITH_NP, cntk_two_dynamicity=True,
                                   from_logits=True)

    def test_in_top_k(self):
        batch_size = 20
        num_classes = 10

        # Random prediction test case
        predictions = np.random.random((batch_size, num_classes)).astype('float32')
        targets = np.random.randint(num_classes, size=batch_size, dtype='int32')

        # (k == 0 or k > num_classes) does not raise an error
        # but just return an unmeaningful tensor.
        for k in range(1, 2 if K.backend() == 'cntk' else (num_classes + 1)):
            z_list = [b.eval(b.in_top_k(b.variable(predictions, dtype='float32'),
                                        b.variable(targets, dtype='int32'), k))
                      for b in WITH_NP]
            assert_list_pairwise(z_list)

        # Identical prediction test case:
        # randomly set half of the predictions to an identical value
        num_identical = num_classes // 2
        for i in range(batch_size):
            idx_identical = np.random.choice(num_classes,
                                             size=num_identical, replace=False)
            predictions[i, idx_identical] = predictions[i, 0]
        targets = np.zeros(batch_size, dtype='int32')

        for k in range(1, 2 if K.backend() == 'cntk' else (num_classes + 1)):
            z_list = [b.eval(b.in_top_k(b.variable(predictions, dtype='float32'),
                                        b.variable(targets, dtype='int32'), k))
                      for b in WITH_NP]
            assert_list_pairwise(z_list)

    @pytest.mark.parametrize('op,input_shape,kernel_shape,padding,data_format', [
        ('conv1d', (2, 8, 2), (3, 2, 3), 'same', 'channels_last'),
        ('conv1d', (1, 8, 2), (3, 2, 3), 'valid', 'channels_last'),
        ('conv1d', (1, 2, 8), (3, 2, 3), 'valid', 'channels_first'),
        ('conv2d', (2, 3, 4, 5), (3, 3, 3, 2), 'same', 'channels_first'),
        ('conv2d', (2, 3, 5, 6), (4, 3, 3, 4), 'valid', 'channels_first'),
        ('conv2d', (1, 6, 5, 3), (3, 4, 3, 2), 'valid', 'channels_last'),
        ('conv2d', (1, 7, 6, 3), (3, 3, 3, 4), 'same', 'channels_last'),
        ('conv3d', (2, 3, 4, 5, 4), (3, 3, 3, 3, 4), 'same', 'channels_first'),
        ('conv3d', (2, 3, 5, 4, 6), (3, 2, 4, 3, 4), 'valid', 'channels_first'),
        ('conv3d', (1, 2, 2, 2, 1), (2, 2, 2, 1, 1), 'valid', 'channels_last'),
        ('conv3d', (1, 3, 5, 4, 2), (3, 3, 3, 2, 3), 'same', 'channels_last'),
    ])
    def test_conv(self, op, input_shape, kernel_shape, padding, data_format):
        check_two_tensor_operation(
            op, input_shape, kernel_shape, WITH_NP,
            padding=padding, data_format=data_format,
            cntk_dynamicity=True)

    @pytest.mark.parametrize(
        'op,input_shape,kernel_shape,output_shape,padding,data_format', [
            ('conv2d_transpose', (2, 5, 6, 3), (3, 3, 2, 3), (2, 5, 6, 2),
             'same', 'channels_last'),
            ('conv2d_transpose', (2, 3, 8, 9), (3, 3, 2, 3), (2, 2, 8, 9),
             'same', 'channels_first'),
        ])
    def test_conv_transpose(self,
                            op,
                            input_shape,
                            kernel_shape,
                            output_shape,
                            padding,
                            data_format):
        check_two_tensor_operation(
            op, input_shape, kernel_shape, WITH_NP,
            output_shape=output_shape, padding=padding, data_format=data_format,
            cntk_dynamicity=True)

    @pytest.mark.skipif((K.backend() == 'cntk' and KC.dev.type() == 0),
                        reason='cntk only supports dilated conv on GPU')
    @pytest.mark.parametrize(
        'op,input_shape,kernel_shape,padding,data_format,dilation_rate', [
            ('conv1d', (2, 8, 3), (4, 3, 2), 'valid', 'channels_last', 2),
            ('conv1d', (2, 3, 8), (4, 3, 2), 'valid', 'channels_first', 2),
            ('conv2d', (2, 8, 9, 3), (3, 3, 3, 2),
             'same', 'channels_last', (2, 2)),
            ('conv2d', (2, 3, 9, 8), (4, 3, 3, 4),
             'valid', 'channels_first', (2, 2)),
            ('conv3d', (2, 5, 4, 6, 3), (2, 2, 3, 3, 4),
             'valid', 'channels_last', (2, 2, 2)),
            ('conv3d', (2, 3, 5, 4, 6), (2, 2, 3, 3, 4),
             'same', 'channels_first', (2, 2, 2)),
        ])
    def test_dilated_conv(self,
                          op,
                          input_shape,
                          kernel_shape,
                          padding,
                          data_format,
                          dilation_rate):
        check_two_tensor_operation(
            op, input_shape, kernel_shape, WITH_NP,
            padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, cntk_dynamicity=True)

    @pytest.mark.skipif((K.backend() == 'cntk' and KC.dev.type() == 0),
                        reason='cntk only supports dilated conv transpose on GPU')
    @pytest.mark.parametrize(
        'op,input_shape,kernel_shape,output_shape,padding,data_format,dilation_rate',
        [
            ('conv2d_transpose', (2, 5, 6, 3), (3, 3, 2, 3), (2, 5, 6, 2),
             'same', 'channels_last', (2, 2)),
            ('conv2d_transpose', (2, 3, 8, 9), (3, 3, 2, 3), (2, 2, 8, 9),
             'same', 'channels_first', (2, 2)),
        ])
    def test_dilated_conv_transpose(self,
                                    op,
                                    input_shape,
                                    kernel_shape,
                                    output_shape,
                                    padding,
                                    data_format,
                                    dilation_rate):
        check_two_tensor_operation(
            op, input_shape, kernel_shape, WITH_NP, output_shape=output_shape,
            padding=padding, data_format=data_format, dilation_rate=dilation_rate,
            cntk_dynamicity=True)

    @pytest.mark.parametrize('op,input_shape,kernel_shape,padding,data_format', [
        ('depthwise_conv2d', (2, 3, 4, 5), (3, 3, 3, 2), 'same', 'channels_first'),
        ('depthwise_conv2d', (2, 3, 5, 6), (4, 3, 3, 4), 'valid', 'channels_first'),
        ('depthwise_conv2d', (1, 6, 5, 3), (3, 4, 3, 2), 'valid', 'channels_last'),
        ('depthwise_conv2d', (1, 7, 6, 3), (3, 3, 3, 4), 'same', 'channels_last'),
    ])
    def test_depthwise_conv(self,
                            op,
                            input_shape,
                            kernel_shape,
                            padding,
                            data_format):
        check_two_tensor_operation(
            op, input_shape, kernel_shape, WITH_NP,
            padding=padding, data_format=data_format,
            cntk_dynamicity=True)

    @pytest.mark.parametrize(
        'op,input_shape,pool_size,strides,padding,data_format,pool_mode', [
            ('pool2d', (2, 3, 7, 7), (3, 3), (1, 1),
             'same', 'channels_first', 'avg'),
            ('pool2d', (3, 3, 8, 5), (2, 3), (1, 1),
             'valid', 'channels_first', 'max'),
            ('pool2d', (2, 9, 5, 3), (3, 2), (1, 1),
             'valid', 'channels_last', 'avg'),
            ('pool2d', (3, 6, 7, 3), (3, 3), (1, 1),
             'same', 'channels_last', 'max'),
            ('pool3d', (2, 3, 7, 7, 7), (3, 3, 3), (1, 1, 1),
             'same', 'channels_first', 'avg'),
            ('pool3d', (3, 3, 8, 5, 9), (2, 3, 2), (1, 1, 1),
             'valid', 'channels_first', 'max'),
            ('pool3d', (2, 8, 9, 5, 3), (3, 2, 3), (1, 1, 1),
             'valid', 'channels_last', 'avg'),
            ('pool3d', (3, 5, 6, 7, 3), (3, 3, 3), (1, 1, 1),
             'same', 'channels_last', 'max'),
        ])
    def test_pool(self,
                  op,
                  input_shape,
                  pool_size,
                  strides,
                  padding,
                  data_format,
                  pool_mode):
        check_single_tensor_operation(
            op, input_shape, WITH_NP,
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, pool_mode=pool_mode,
            cntk_dynamicity=True)

    @pytest.mark.parametrize(
        'op,input_shape,kernel_shape,depth_multiplier,padding,data_format', [
            ('separable_conv1d', (2, 8, 2), (3,), 1, 'same', 'channels_last'),
            ('separable_conv1d', (1, 8, 2), (3,), 2, 'valid', 'channels_last'),
            ('separable_conv2d', (2, 3, 4, 5), (3, 3), 1, 'same', 'channels_first'),
            ('separable_conv2d', (2, 3, 5, 6), (4, 3), 2, 'valid', 'channels_first'),
            ('separable_conv2d', (1, 6, 5, 3), (3, 4), 1, 'valid', 'channels_last'),
            ('separable_conv2d', (1, 7, 6, 3), (3, 3), 2, 'same', 'channels_last'),
        ])
    def test_separable_conv(self,
                            op,
                            input_shape,
                            kernel_shape,
                            depth_multiplier,
                            padding,
                            data_format):
        if data_format == 'channels_first':
            input_depth = input_shape[1]
        else:
            input_depth = input_shape[-1]
        _, x = parse_shape_or_val(input_shape)
        _, depthwise = parse_shape_or_val(kernel_shape +
                                          (input_depth, depth_multiplier))
        _, pointwise = parse_shape_or_val((1,) * len(kernel_shape) +
                                          (input_depth * depth_multiplier, 7))
        y1 = KNP.separable_conv(x, depthwise, pointwise,
                                padding=padding, data_format=data_format)
        if K.backend() == 'cntk':
            _, cntk_func = cntk_func_tensors(
                op, [input_shape, depthwise, pointwise],
                padding=padding, data_format=data_format)
            y2 = cntk_func([x])[0]
        else:
            y2 = K.eval(getattr(K, op)(
                K.variable(x),
                K.variable(depthwise), K.variable(pointwise),
                padding=padding, data_format=data_format))
        assert_allclose(y1, y2, atol=1e-05)

    def test_random_normal(self):
        # TODO: make this a parameterized test
        for mean, std in [(0., 1.), (-10., 5.)]:
            rand = K.eval(K.random_normal((200, 200),
                                          mean=mean,
                                          stddev=std))
            assert rand.shape == (200, 200)
            assert np.abs(np.mean(rand) - mean) < std * 0.015
            assert np.abs(np.std(rand) - std) < std * 0.015

    def test_random_uniform(self):
        min_val = -1.
        max_val = 1.
        rand = K.eval(K.random_uniform((200, 200), min_val, max_val))
        assert rand.shape == (200, 200)
        assert np.abs(np.mean(rand)) < 0.015
        assert max_val - 0.015 < np.max(rand) <= max_val
        assert min_val + 0.015 > np.min(rand) >= min_val

    def test_random_binomial(self):
        p = 0.5
        rand = K.eval(K.random_binomial((200, 200), p))
        assert rand.shape == (200, 200)
        assert np.abs(np.mean(rand) - p) < 0.015
        assert np.max(rand) == 1
        assert np.min(rand) == 0

    def test_truncated_normal(self):
        mean = 0.
        std = 1.
        min_val = -2.
        max_val = 2.
        rand = K.eval(K.truncated_normal((200, 200),
                                         mean=mean,
                                         stddev=std))
        assert rand.shape == (200, 200)
        assert np.abs(np.mean(rand) - mean) < 0.016
        assert np.max(rand) <= max_val
        assert np.min(rand) >= min_val

        # assumption in initializers.VarianceScaling
        assert np.abs(np.std(rand) - std * 0.87962) < 0.015

    def test_conv_invalid_use(self):
        dummy_x_1d = K.variable(np.ones((4, 8, 2)))
        dummy_w_1d = K.variable(np.ones((3, 2, 3)))
        dummy_x_2d = K.variable(np.ones((2, 3, 4, 5)))
        dummy_w_2d = K.variable(np.ones((2, 2, 3, 4)))
        dummy_x_3d = K.variable(np.ones((2, 3, 4, 5, 4)))
        dummy_w_3d = K.variable(np.ones((2, 2, 2, 3, 4)))
        dummy_w1x1_2d = K.variable(np.ones((1, 1, 12, 7)))

        with pytest.raises(ValueError):
            K.conv1d(dummy_x_1d, dummy_w_1d, data_format='channels_middle')

        with pytest.raises(ValueError):
            K.conv2d(dummy_x_2d, dummy_w_2d, data_format='channels_middle')

        with pytest.raises(ValueError):
            K.conv3d(dummy_x_3d, dummy_w_3d, data_format='channels_middle')

        with pytest.raises(ValueError):
            K.separable_conv2d(dummy_x_2d, dummy_w_2d, dummy_w1x1_2d,
                               data_format='channels_middle')

        with pytest.raises(ValueError):
            K.depthwise_conv2d(dummy_x_2d, dummy_w_2d,
                               data_format='channels_middle')

        if K.backend() == 'cntk':
            with pytest.raises(ValueError):
                K.separable_conv2d(dummy_x_2d, dummy_w_2d, dummy_w1x1_2d,
                                   dilation_rate=(1, 2))
            with pytest.raises(ValueError):
                K.separable_conv2d(dummy_x_2d, dummy_w_2d, dummy_w1x1_2d,
                                   strides=(2, 2), dilation_rate=(1, 2))
            with pytest.raises(ValueError):
                K.depthwise_conv2d(dummy_x_2d, dummy_w_2d,
                                   dilation_rate=(1, 2))
            with pytest.raises(ValueError):
                K.depthwise_conv2d(dummy_x_2d, dummy_w_2d,
                                   strides=(2, 2), dilation_rate=(1, 2))

    def test_pooling_invalid_use(self):
        for (input_shape, pool_size) in zip([(5, 10, 12, 3), (5, 10, 12, 6, 3)],
                                            [(2, 2), (2, 2, 2)]):
            x = K.variable(np.random.random(input_shape))
            if len(pool_size) == 2:
                with pytest.raises(ValueError):
                    K.pool2d(x, pool_size=pool_size, data_format='channels_middle')
                with pytest.raises(ValueError):
                    K.pool2d(x, pool_size=pool_size, padding='twice')
                with pytest.raises(ValueError):
                    K.pool2d(x, pool_size=pool_size, pool_mode='median')
            else:
                with pytest.raises(ValueError):
                    K.pool3d(x, pool_size=pool_size, data_format='channels_middle')
                with pytest.raises(ValueError):
                    K.pool3d(x, pool_size=pool_size, padding='twice')
                with pytest.raises(ValueError):
                    K.pool3d(x, pool_size=pool_size, pool_mode='median')

    def test_resize_images(self):
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5)
            if data_format == 'channels_first':
                x_shape = (2, 3) + shape
            elif data_format == 'channels_last':
                x_shape = (2,) + shape + (3,)
            check_single_tensor_operation('resize_images', x_shape,
                                          WITH_NP, cntk_dynamicity=True,
                                          height_factor=2,
                                          width_factor=2,
                                          data_format=data_format)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        with pytest.raises(ValueError):
            K.resize_images(K.variable(xval), 2, 2,
                            data_format='channels_middle')

    @staticmethod
    def _helper_bilinear(data_format, height_factor, width_factor):
        x_shape = (2, 3, 4, 5)
        check_single_tensor_operation('resize_images', x_shape,
                                      [KTF, KTH],
                                      height_factor=height_factor,
                                      width_factor=width_factor,
                                      data_format=data_format,
                                      interpolation='bilinear')

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported.')
    @pytest.mark.parametrize('data_format', ['channels_first', 'channels_last'])
    def test_resize_images_bilinear(self, data_format):
        self._helper_bilinear(data_format, 2, 2)
        with pytest.raises(NotImplementedError):
            self._helper_bilinear(data_format, 4, 4)

    def test_resize_volumes(self):
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5, 5)
            if data_format == 'channels_first':
                x_shape = (2, 3) + shape
            elif data_format == 'channels_last':
                x_shape = (2,) + shape + (3,)
            check_single_tensor_operation('resize_volumes', x_shape,
                                          WITH_NP, cntk_dynamicity=True,
                                          depth_factor=2,
                                          height_factor=2,
                                          width_factor=2,
                                          data_format=data_format)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        with pytest.raises(ValueError):
            K.resize_volumes(K.variable(xval), 2, 2, 2,
                             data_format='channels_middle')

    def test_temporal_padding(self):
        check_single_tensor_operation('temporal_padding', (4, 3, 3),
                                      WITH_NP)
        check_single_tensor_operation('temporal_padding', (2, 3, 4),
                                      WITH_NP, padding=(1, 2))

    def test_spatial_2d_padding(self):
        padding = ((1, 2), (2, 1))
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5)
            if data_format == 'channels_first':
                x_shape = (1, 3) + shape
            else:
                x_shape = (1,) + shape + (3,)
            check_single_tensor_operation('spatial_2d_padding', x_shape, WITH_NP,
                                          padding=padding, data_format=data_format)
        # Check handling of dynamic shapes.
        if K in [KTF, KTH]:
            x = K.placeholder(shape=(1, None, None, 1))
            y = K.spatial_2d_padding(x, padding=padding, data_format='channels_last')
            assert K.int_shape(y) == (1, None, None, 1)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        with pytest.raises(ValueError):
            K.spatial_2d_padding(K.variable(xval), padding=padding,
                                 data_format='channels_middle')

    def test_spatial_3d_padding(self):
        padding = ((1, 2), (2, 1), (1, 2))
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5, 5)
            if data_format == 'channels_first':
                x_shape = (1, 3) + shape
            else:
                x_shape = (1,) + shape + (3,)
            check_single_tensor_operation('spatial_3d_padding', x_shape, WITH_NP,
                                          padding=padding, data_format=data_format)
        # Check handling of dynamic shapes.
        if K in [KTF, KTH]:
            x = K.placeholder(shape=(1, None, None, None, 1))
            y = K.spatial_3d_padding(x, padding=padding, data_format='channels_last')
            assert K.int_shape(y) == (1, None, None, None, 1)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        with pytest.raises(ValueError):
            K.spatial_3d_padding(K.variable(xval), padding=padding,
                                 data_format='channels_middle')

    def test_bias_add(self):
        for data_format in ['channels_first', 'channels_last']:
            for shape in [(), (3,), (2, 3), (5, 3, 2)]:
                if data_format == 'channels_first':
                    x_shape = (1, 4) + shape
                else:
                    x_shape = (1,) + shape + (4,)
                bias_shape = (4,)
                check_two_tensor_operation('bias_add', x_shape, bias_shape,
                                           WITH_NP, cntk_dynamicity=True,
                                           data_format=data_format)

            if data_format == 'channels_first':
                x_shape = (20, 6, 10)
            else:
                x_shape = (20, 10, 6)
            check_two_tensor_operation('bias_add', x_shape, (10, 6),
                                       WITH_NP, cntk_dynamicity=True,
                                       data_format=data_format)

        # Test invalid use cases
        x = K.variable(np.random.random(x_shape))
        b = K.variable(np.random.random(bias_shape))
        with pytest.raises(ValueError):
            K.bias_add(x, b, data_format='channels_middle')

    @pytest.mark.skipif(K.backend() == 'theano',
                        reason='Theano behaves differently '
                               'because of the broadcast.')
    @pytest.mark.parametrize('axis', [1, -1])
    @pytest.mark.parametrize('x_shape', [(3, 2, 4, 5), (3, 2, 4)])
    def test_batch_normalization(self, axis, x_shape):
        other_shape = [1] * len(x_shape)
        other_shape[axis] = x_shape[axis]
        other_shape = tuple(other_shape)
        x_np = np.random.random(x_shape)
        mean_np = np.random.random(other_shape)
        var_np = np.random.random(other_shape)
        beta_np = np.random.random(other_shape)
        gamma_np = np.random.random(other_shape)
        output_tensors = []
        output_arrays = []
        for k in WITH_NP:
            x = k.variable(x_np)
            mean = k.variable(mean_np)
            var = k.variable(var_np)
            beta = k.variable(beta_np)
            gamma = k.variable(gamma_np)
            output = k.batch_normalization(x, mean, var, beta, gamma, axis=axis)
            output_tensors.append(output)
            output_arrays.append(k.eval(output))
        assert_list_pairwise(output_arrays)
        assert_list_keras_shape(output_tensors, output_arrays)

    @pytest.mark.skipif(K.backend() != 'theano',
                        reason='Specific to Theano.')
    @pytest.mark.parametrize('x_shape', [(1, 4, 2, 3), (1, 2, 3, 4)])
    def test_batchnorm_th(self, x_shape):
        x_val = np.random.random(x_shape).astype(np.float32)
        x = K.variable(x_val)
        z, _, _ = K.normalize_batch_in_training(
            x, None, None, reduction_axes='per-activation')
        z = K.eval(z)
        assert z.shape == x_shape

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='Specific to Tensorflow.')
    @pytest.mark.parametrize('x_shape', [(1, 4, 2, 3), (1, 2, 3, 4)])
    def test_batchnorm_tf(self, x_shape):
        x_val = np.random.random(x_shape).astype(np.float32)
        x = K.variable(x_val)
        z, _, _ = K.normalize_batch_in_training(
            x, None, None, reduction_axes=[0, 1, 2, 3])
        z = K.eval(z)
        assert z.shape == x_shape

    @pytest.mark.skipif(K.backend() != 'cntk', reason='Specific to CNTK.')
    @pytest.mark.parametrize('x_shape', [(1, 4, 2, 3), (1, 2, 3, 4)])
    def test_batchnorm_cntk(self, x_shape):
        x_val = np.random.random(x_shape).astype(np.float32)
        x = K.placeholder(x_shape)
        z, _, _ = K.normalize_batch_in_training(
            x, None, None, reduction_axes=[0, 1, 2, 3])
        z = K.function([x], [z])([x_val])[0]
        assert z.shape == x_shape

    # the Theano and TensorFlow CTC code use different methods to ensure
    # numerical stability.  The Theano code subtracts out the max
    # before the final log, so the results are different but scale
    # identically and still train properly
    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported.')
    def test_ctc(self):
        if K.backend() == 'theano':
            ref = [1.73308, 3.81351]
        else:
            ref = [3.34211, 5.42262]
        # simplified version of TensorFlow's test

        label_lens = np.expand_dims(np.asarray([5, 4]), 1)
        input_lens = np.expand_dims(np.asarray([5, 5]), 1)  # number of timesteps

        # dimensions are batch x time x categories
        labels = np.asarray([[0, 1, 2, 1, 0], [0, 1, 1, 0, -1]])
        inputs = np.asarray(
            [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
              [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
              [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
              [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
              [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
             [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
              [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
              [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
              [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
              [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]],
            dtype=np.float32)

        k_labels = K.variable(labels, dtype="int32")
        k_inputs = K.variable(inputs, dtype="float32")
        k_input_lens = K.variable(input_lens, dtype="int32")
        k_label_lens = K.variable(label_lens, dtype="int32")
        res = K.eval(K.ctc_batch_cost(k_labels, k_inputs, k_input_lens,
                                      k_label_lens))
        if K.backend() == 'theano':
            assert_allclose(res[0, :], ref, atol=1e-05)
        else:
            assert_allclose(res[:, 0], ref, atol=1e-05)

        # test when batch_size = 1, that is, one sample only
        # get only first sample from above test case
        if K.backend() == 'theano':
            ref = [1.73308]
        else:
            ref = [3.34211]

        input_lens = np.expand_dims(np.asarray([5]), 1)
        label_lens = np.expand_dims(np.asarray([5]), 1)

        labels = np.asarray([[0, 1, 2, 1, 0]])
        inputs = np.asarray(
            [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
              [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
              [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
              [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
              [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]]],
            dtype=np.float32)

        k_labels = K.variable(labels, dtype="int32")
        k_inputs = K.variable(inputs, dtype="float32")
        k_input_lens = K.variable(input_lens, dtype="int32")
        k_label_lens = K.variable(label_lens, dtype="int32")
        res = K.eval(K.ctc_batch_cost(k_labels, k_inputs, k_input_lens,
                                      k_label_lens))
        if K.backend() == 'theano':
            assert_allclose(res[0, :], ref, atol=1e-05)
        else:
            assert_allclose(res[:, 0], ref, atol=1e-05)

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='Test adapted from tensorflow.')
    def test_ctc_decode_greedy(self):
        """Test two batch entries - best path decoder."""
        max_time_steps = 6

        seq_len_0 = 4
        input_prob_matrix_0 = np.asarray(
            [[1.0, 0.0, 0.0, 0.0],  # t=0
             [0.0, 0.0, 0.4, 0.6],  # t=1
             [0.0, 0.0, 0.4, 0.6],  # t=2
             [0.0, 0.9, 0.1, 0.0],  # t=3
             [0.0, 0.0, 0.0, 0.0],  # t=4 (ignored)
             [0.0, 0.0, 0.0, 0.0]],  # t=5 (ignored)
            dtype=np.float32)

        seq_len_1 = 5
        # dimensions are time x depth

        input_prob_matrix_1 = np.asarray(
            [[0.1, 0.9, 0.0, 0.0],  # t=0
             [0.0, 0.9, 0.1, 0.0],  # t=1
             [0.0, 0.0, 0.1, 0.9],  # t=2
             [0.0, 0.9, 0.1, 0.1],  # t=3
             [0.9, 0.1, 0.0, 0.0],  # t=4
             [0.0, 0.0, 0.0, 0.0]],  # t=5 (ignored)
            dtype=np.float32)

        # len max_time_steps array of batch_size x depth matrices
        inputs = [np.vstack([input_prob_matrix_0[t, :],
                             input_prob_matrix_1[t, :]])
                  for t in range(max_time_steps)]

        # change tensorflow order to keras backend order
        inputs = np.asarray(inputs).transpose((1, 0, 2))

        # batch_size length vector of sequence_lengths
        input_length = np.array([seq_len_0, seq_len_1], dtype=np.int32)

        decode_pred_np, log_prob_pred_np = KNP.ctc_decode(inputs,
                                                          input_length, greedy=True)
        inputs = K.variable(inputs)
        input_length = K.variable(input_length)
        decode_pred_tf, log_prob_pred_tf = K.ctc_decode(inputs,
                                                        input_length, greedy=True)

        assert len(decode_pred_tf) == 1

        decode_pred = K.eval(decode_pred_tf[0])
        log_prob_pred = K.eval(log_prob_pred_tf)

        assert np.alltrue(decode_pred_np == decode_pred)
        assert np.allclose(log_prob_pred_np, log_prob_pred)

    @pytest.mark.parametrize('shape,start,size', [
        ((2, 5), (0, 1), (2, 3)),
        ((2, 5), (1, 0), (1, 4)),
        ((3, 2, 3), (1, 1, 0), (1, 1, 3)),
        ((3, 2, 3), (1, 0, 0), (1, 2, 3)),
        ((3, 2, 3), (1, 0, 0), (2, 1, 3)),
    ])
    def test_slice(self, shape, start, size):
        check_single_tensor_operation('slice', shape, WITH_NP,
                                      start=start, size=size)
        with pytest.raises(ValueError):
            K.slice(K.variable(np.random.random(shape)),
                    start=[1, 0, 0, 0], size=size)

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='Beam search is only implemented with '
                               'the TensorFlow backend.')
    def test_ctc_decode_beam_search(self):
        """Test one batch, two beams - hibernating beam search."""

        depth = 6

        seq_len_0 = 5
        input_prob_matrix_0 = np.asarray(
            [[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
             [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
             [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
             [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
             [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
             # Random entry added in at time=5
             [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]],
            dtype=np.float32)

        # Add arbitrary offset - this is fine
        input_prob_matrix_0 = input_prob_matrix_0 + 2.0

        # len max_time_steps array of batch_size x depth matrices
        inputs = ([input_prob_matrix_0[t, :][np.newaxis, :]
                   for t in range(seq_len_0)] +  # Pad to max_time_steps = 8
                  2 * [np.zeros((1, depth), dtype=np.float32)])

        # Take exponential as we directly apply ctc_decode_beam_search
        inputs = np.exp(inputs)

        # change tensorflow order to keras backend order
        inputs = K.variable(inputs.transpose((1, 0, 2)))

        # batch_size length vector of sequence_lengths
        input_length = K.variable(np.array([seq_len_0], dtype=np.int32))
        # batch_size length vector of log probabilities
        log_prob_truth = np.array(
            [
                -5.811451,  # output beam 0
                -6.63339  # output beam 1
            ],
            np.float32)[np.newaxis, :]

        decode_truth = [np.array([1, 0]), np.array([[1]])]

        beam_width = 2
        top_paths = 2

        decode_pred_tf, log_prob_pred_tf = K.ctc_decode(inputs,
                                                        input_length,
                                                        greedy=False,
                                                        beam_width=beam_width,
                                                        top_paths=top_paths)

        assert len(decode_pred_tf) == top_paths

        log_prob_pred = K.eval(log_prob_pred_tf)

        for i in range(top_paths):
            assert np.alltrue(decode_truth[i] == K.eval(decode_pred_tf[i]))

        assert np.allclose(log_prob_truth, log_prob_pred)

    @pytest.mark.skipif(K.backend() != 'tensorflow',
                        reason='Beam search is only implemented with '
                               'the TensorFlow backend.')
    def test_ctc_decode_beam_search_no_merge(self):
        # A simple CTC probability map with some repeating characters,
        # shape(batch, input_width, char_count)
        # Without merging should be decoded as: "AABB", with merging as: "AB".
        input_prob = np.array([
            [  # blank, A ,B
                [0, 0, 1],  # blank
                [1, 0, 0],  # A
                [0, 0, 1],  # blank
                [1, 0, 0],  # A
                [0, 1, 0],  # B
                [0, 0, 1],  # blank
                [0, 1, 0]  # B
            ]
        ])
        input_len = np.array(input_prob.shape[0] * [input_prob.shape[1]])

        def decode(merge_repeated):
            input_prob_tensor = K.placeholder(shape=(None, None, None),
                                              dtype='float32')
            input_len_tensor = K.placeholder(shape=(None), dtype='int64')
            paths_tensors, _ = K.ctc_decode(input_prob_tensor, input_len_tensor,
                                            greedy=False, beam_width=1, top_paths=1,
                                            merge_repeated=merge_repeated)
            decode_func = K.function([input_prob_tensor, input_len_tensor],
                                     paths_tensors)
            paths = decode_func([input_prob, input_len])
            return paths

        # merged: A B
        assert np.allclose(decode(merge_repeated=True), [np.array([[0, 1]])])
        # not merged: A A B B
        assert np.allclose(decode(merge_repeated=False), [np.array([[0, 0, 1, 1]])])

    def test_one_hot(self):
        input_length = 10
        num_classes = 20
        batch_size = 30
        indices = np.random.randint(0, num_classes, size=(batch_size, input_length))
        oh = KNP.one_hot(np.int32(indices), num_classes)
        koh = K.eval(K.one_hot(K.variable(indices, dtype='int32'), num_classes))
        assert np.all(koh == oh)

    @pytest.mark.skipif(not supports_sparse,
                        reason='Sparse tensors are not supported in cntk '
                               'and Theano has some dependency issues for sparse.')
    def test_sparse_dot(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()

        W = np.random.random((5, 4))
        t_W = K.variable(W)
        k_s = K.eval(K.dot(K.variable(x_sparse), t_W))
        k_d = K.eval(K.dot(K.variable(x_dense), t_W))

        assert k_s.shape == k_d.shape
        assert_allclose(k_s, k_d, atol=1e-05)

    @pytest.mark.skipif(not supports_sparse,
                        reason='Sparse tensors are not supported in cntk '
                               'and Theano has some dependency issues for sparse.')
    def test_sparse_concat(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_1 = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))

        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse_2 = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))

        x_dense_1 = x_sparse_1.toarray()
        x_dense_2 = x_sparse_2.toarray()

        k_s = K.concatenate([K.variable(x_sparse_1), K.variable(x_sparse_2)])
        assert K.is_sparse(k_s)

        k_s_d = K.eval(k_s)

        k_d = K.eval(K.concatenate([K.variable(x_dense_1), K.variable(x_dense_2)]))

        assert k_s_d.shape == k_d.shape
        assert_allclose(k_s_d, k_d, atol=1e-05)

    @pytest.mark.parametrize('shape,shape2,axis', [
        ((5, 2), (7, 2), 0),
        ((5, 4, 6), (5, 3, 6), 1),
        ((5, 4, 6, 10), (5, 4, 6, 2), 3),
        ((5, 4, 6, 3), (5, 4, 6, 2), -1),
    ])
    def test_concat_operations(self, shape, shape2, axis):
        # In stack, each array must have the same shape.
        check_two_tensor_operation('stack', shape, shape, WITH_NP,
                                   axis=axis, concat_args=True)
        check_two_tensor_operation('concatenate', shape, shape2, WITH_NP,
                                   axis=axis, concat_args=True)
        check_two_tensor_operation('concatenate', shape, shape2, WITH_NP,
                                   axis=axis, concat_args=True)

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported.')
    def test_map(self):
        x = np.random.rand(10, 3).astype(np.float32)
        vx = K.variable(x)
        kx = K.eval(K.map_fn(K.sum, vx))
        # make sure we can also walk the indexes in tensorflow which we
        # can't without specifying dtype
        kx2 = K.eval(K.map_fn(
            lambda i: K.sum(vx[i]),
            K.arange(10),
            dtype=K.floatx()
        ))

        assert (10,) == kx.shape
        assert (10,) == kx2.shape
        assert_allclose(x.sum(axis=1), kx, atol=1e-05)
        assert_allclose(kx, kx2, atol=1e-05)

    def test_foldl(self):
        x = np.random.rand(10, 3).astype(np.float32)
        kx = K.eval(K.foldl(lambda a, b: a + b, K.variable(x)))

        assert (3,) == kx.shape
        assert_allclose(x.sum(axis=0), kx, atol=1e-05)

    def test_foldr(self):
        # This test aims to make sure that we walk the array from right to left
        # and checks it in the following way: multiplying left to right 1e-40
        # cannot be held into a float32 so it causes an underflow while from
        # right to left we have no such problem and the result is larger
        x = np.array([1e-20, 1e-20, 10, 10, 10], dtype=np.float32)
        vx = K.variable(x)
        p1 = K.eval(K.foldl(lambda a, b: a * b, vx))
        p2 = K.eval(K.foldr(lambda a, b: a * b, vx))

        assert p1 < p2
        assert 9e-38 < p2 <= 1e-37

    @pytest.mark.skipif(K.backend() == 'cntk',
                        reason='cntk has issues with negative number.')
    def test_arange(self):
        for test_value in (-20, 0, 1, 10):
            a_list = []
            dtype_list = []
            for k in WITH_NP:
                t = k.arange(test_value)
                a = k.eval(t)
                assert np.array_equal(a, np.arange(test_value))
                dtype_list.append(k.dtype(t))
                a_list.append(a)

            for i in range(len(a_list) - 1):
                assert np.array_equal(a_list[i], a_list[i + 1])

        for start, stop, step in ((0, 5, 1), (-5, 5, 2), (0, 1, 2)):
            a_list = []
            for k in WITH_NP:
                a = k.eval(k.arange(start, stop, step))
                assert np.array_equal(a, np.arange(start, stop, step))
                a_list.append(a)
            for i in range(len(a_list) - 1):
                assert np.array_equal(a_list[i], a_list[i + 1])

        for dtype in ('int32', 'int64', 'float32', 'float64'):
            for k in WITH_NP:
                t = k.arange(10, dtype=dtype)
                assert k.dtype(t) == dtype

        start = K.constant(1, dtype='int32')
        t = K.arange(start)
        assert len(K.eval(t)) == 1

        start = K.constant(-1, dtype='int32')
        t = K.arange(start)
        assert len(K.eval(t)) == 0

    @pytest.mark.parametrize('training', [True, False])
    def test_in_train_phase(self, training):
        check_two_tensor_operation('in_train_phase', (3, 3), (2, 2), WITH_NP,
                                   training=training)
        check_two_tensor_operation('in_train_phase', (2, 3), (2, 3), WITH_NP,
                                   training=training)

    @pytest.mark.parametrize('training', [True, False])
    def test_in_test_phase(self, training):
        check_two_tensor_operation('in_test_phase', (3, 3), (2, 2), WITH_NP,
                                   training=training)
        check_two_tensor_operation('in_test_phase', (2, 3), (2, 3), WITH_NP,
                                   training=training)

    @pytest.mark.parametrize('dtype', ['', 'beerfloat', 123])
    def test_setfloatx_incorrect_values(self, dtype):
        # Keep track of the old value
        old_floatx = K.floatx()
        with pytest.raises(ValueError):
            K.set_floatx(dtype)
        assert K.floatx() == old_floatx

    @pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
    def test_setfloatx_correct_values(self, dtype):
        # Keep track of the old value
        old_floatx = K.floatx()
        # Check correct values
        K.set_floatx(dtype)
        assert K.floatx() == dtype
        # Make sure that changes to the global floatx are effectively
        # taken into account by the backend.
        check_dtype(K.variable([10]), dtype)
        # Restore old value
        K.set_floatx(old_floatx)

    @pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
    def test_dtype(self, dtype):
        assert K.dtype(K.variable(1, dtype=dtype)) == dtype

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported')
    def test_variable_support_bool_dtype(self):
        assert K.dtype(K.variable(1, dtype='int16')) == 'int16'
        assert K.dtype(K.variable(False, dtype='bool')) == 'bool'
        with pytest.raises(TypeError):
            K.variable('', dtype='unsupported')

    @pytest.mark.parametrize('shape', [(4, 2), (2, 3)])
    def test_clip_supports_tensor_arguments(self, shape):
        # GitHub issue: 11435
        _, x = parse_shape_or_val(shape)
        _, min_val = parse_shape_or_val(shape)
        max_val = min_val + 1.
        x_k = K.variable(x)
        min_val_k = K.variable(min_val)
        max_val_k = K.variable(max_val)
        assert np.allclose(K.eval(K.clip(x_k, min_val_k, max_val_k)),
                           KNP.eval(KNP.clip(x, min_val, max_val)))


if __name__ == '__main__':
    pytest.main([__file__])
