import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sparse
import warnings
from keras.utils.test_utils import keras_test

from keras import backend as K
from keras.backend import floatx, set_floatx, variable
from keras.utils.conv_utils import convert_kernel
import reference_operations


BACKENDS = []  # Holds a list of all available back-ends

try:
    from keras.backend import cntk_backend as KC
    BACKENDS.append(KC)
except ImportError:
    KC = None
    warnings.warn('Could not import the CNTK backend')

try:
    from keras.backend import tensorflow_backend as KTF
    BACKENDS.append(KTF)
except ImportError:
    KTF = None
    warnings.warn('Could not import the TensorFlow backend.')

try:
    from keras.backend import theano_backend as KTH
    BACKENDS.append(KTH)
except ImportError:
    KTH = None
    warnings.warn('Could not import the Theano backend')


def check_dtype(var, dtype):
    if K._BACKEND == 'theano':
        assert var.dtype == dtype
    else:
        assert var.dtype.name == '%s_ref' % dtype


def cntk_func_single_tensor(function_name, x_shape, **kwargs):
    xc = KC.placeholder(x_shape)
    output_cntk = getattr(KC, function_name)(xc, **kwargs)
    return KC.function([xc], [output_cntk])


def cntk_func_two_tensor(function_name, x_shape, y, **kwargs):
    if isinstance(y, (np.generic, np.ndarray)):
        xc = KC.placeholder(x_shape)
        output_cntk = getattr(KC, function_name)(xc, KC.variable(y), **kwargs)
        return KC.function([xc], [output_cntk])
    else:
        xc = KC.placeholder(ndim=len(x_shape))
        yc = KC.placeholder(y)
        output_cntk = getattr(KC, function_name)(xc, yc, **kwargs)
        return KC.function([xc, yc], [output_cntk])


def cntk_func_three_tensor(function_name, x_shape, y, z, **kwargs):
    xc = KC.placeholder(x_shape)
    output_cntk = getattr(KC, function_name)(xc, KC.variable(y), KC.variable(z), **kwargs)
    return KC.function([xc], [output_cntk])


def parse_shape_or_val(shape_or_val):
    if isinstance(shape_or_val, np.ndarray):
        return shape_or_val.shape, shape_or_val
    else:
        return shape_or_val, np.random.random(shape_or_val).astype(np.float32) - 0.5


def assert_list_pairwise(z_list, shape=True, allclose=True, itself=False, atol=1e-05):
    for (z1, z2) in zip(z_list[1:], z_list[:-1]):
        if shape:
            assert z1.shape == z2.shape
        if allclose:
            assert_allclose(z1, z2, atol=atol)
        if itself:
            assert z1 == z2


def assert_list_with_ref(z_list, ref):
    for z in z_list:
        assert z.shape == ref.shape
        assert_allclose(z, ref, atol=1e-05)


def assert_list_keras_shape(z_list):
    for z in z_list:
        if hasattr(z, '_keras_shape'):
            assert z._keras_shape == z.shape


@keras_test
def check_single_tensor_operation(function_name, x_shape_or_val, backend_list, **kwargs):
    shape_or_val = kwargs.pop('shape_or_val', True)
    assert_value_equality = kwargs.pop('assert_value_equality', True)
    assert_value_with_ref = kwargs.pop('assert_value_with_ref', None)
    cntk_dynamicity = kwargs.pop('cntk_dynamicity', False)
    return_results = kwargs.pop('return_results', False)

    if shape_or_val:
        x_shape, x_val = parse_shape_or_val(x_shape_or_val)

    z_list = []
    for k in backend_list:
        if shape_or_val:
            if (k == KC) & (cntk_dynamicity):
                z = cntk_func_single_tensor(function_name, x_shape,
                                            **kwargs)([x_val])[0]
            else:
                z = k.eval(getattr(k, function_name)(k.variable(x_val), **kwargs))
        else:
            z = k.eval(getattr(k, function_name)(x_shape_or_val, **kwargs))
        z_list += [z]

    if return_results:
        if len(z_list) > 1:
            return z_list
        else:
            return z_list[0]

    if assert_value_with_ref is not None:
        assert_list_with_ref(z_list, assert_value_with_ref)
    else:
        assert_list_pairwise(z_list, allclose=assert_value_equality)
    assert_list_keras_shape(z_list)


@keras_test
def check_two_tensor_operation(function_name, x_shape_or_val,
                               y_shape_or_val, backend_list, **kwargs):
    shape_or_val = kwargs.pop('shape_or_val', True)
    concat_args = kwargs.pop('concat_args', False)
    cntk_dynamicity = kwargs.pop('cntk_dynamicity', False)
    cntk_two_dynamicity = kwargs.pop('cntk_two_dynamicity', False)
    return_results = kwargs.pop('return_results', False)

    if shape_or_val:
        x_shape, x_val = parse_shape_or_val(x_shape_or_val)
        y_shape, y_val = parse_shape_or_val(y_shape_or_val)

    z_list = []
    for k in backend_list:
        if shape_or_val:
            if (k == KC) & (cntk_dynamicity):
                z = cntk_func_two_tensor(function_name, x_shape,
                                         y=y_val, **kwargs)([x_val])[0]
            elif (k == KC) & (cntk_two_dynamicity):
                z = cntk_func_two_tensor(function_name, x_shape,
                                         y=y_shape, **kwargs)([x_val, y_val])[0]
            elif (k == KTH) & (function_name[:4] == 'conv'):
                z = k.eval(getattr(k, function_name)(
                    k.variable(x_val), k.variable(convert_kernel(y_val)), **kwargs))
            elif concat_args:
                z = k.eval(getattr(k, function_name)(
                    [k.variable(x_val), k.variable(y_val)], **kwargs))
            else:
                z = k.eval(getattr(k, function_name)(
                    k.variable(x_val), k.variable(y_val), **kwargs))
        else:
            z = k.eval(getattr(k, function_name)(
                x_shape_or_val, y_shape_or_val, **kwargs))
        z_list += [z]

    if return_results:
        if len(z_list) > 1:
            return z_list
        else:
            return z_list[0]

    assert_list_pairwise(z_list)
    assert_list_keras_shape(z_list)


@keras_test
def check_composed_tensor_operations(first_function_name, first_function_args,
                                     second_function_name, second_function_args,
                                     input_shape, backend_list):
    val = np.random.random(input_shape) - 0.5

    z_list = []
    for k in backend_list:
        x = k.variable(val)
        y = getattr(k, first_function_name)(x, **first_function_args)
        z = k.eval(getattr(k, second_function_name)(y, **second_function_args))
        z_list += [z]

    assert_list_pairwise(z_list)


class TestBackend(object):

    def test_is_keras_tensor(self):
        for k in BACKENDS:
            np_var = np.array([1, 2])
            with pytest.raises(ValueError):
                k.is_keras_tensor(np_var)

            keras_var = k.variable(np_var)
            assert k.is_keras_tensor(keras_var) is False
            keras_placeholder = k.placeholder(shape=(2, 4, 5))
            assert k.is_keras_tensor(keras_placeholder) is False

    def test_set_learning_phase(self):
        # not supported learning_phase
        for k in BACKENDS:
            with pytest.raises(ValueError):
                k.set_learning_phase(2)

    def test_eye(self):
        z_list = [k.eval(k.eye(3)) for k in BACKENDS]
        assert_list_pairwise(z_list)

    def test_linear_operations(self):
        check_two_tensor_operation('dot', (4, 2), (2, 4), BACKENDS)
        check_two_tensor_operation('dot', (4, 2), (5, 2, 3), BACKENDS)

        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 5, 3),
                                   BACKENDS, cntk_two_dynamicity=True, axes=(2, 2))
        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 3),
                                   BACKENDS, cntk_two_dynamicity=True, axes=(2, 1))
        check_two_tensor_operation('batch_dot', (4, 2), (4, 2, 3),
                                   BACKENDS, cntk_two_dynamicity=True, axes=(1, 1))
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20),
                                   BACKENDS, cntk_two_dynamicity=True, axes=1)
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20),
                                   BACKENDS, cntk_two_dynamicity=True, axes=(1, 1))

        check_single_tensor_operation('transpose', (4, 2), BACKENDS)
        check_single_tensor_operation('reverse', (4, 3, 2), BACKENDS, axes=1)
        check_single_tensor_operation('reverse', (4, 3, 2), [KTH, KTF], axes=(1, 2))

    def test_random_variables(self):
        check_single_tensor_operation('random_uniform_variable', (2, 3), BACKENDS,
                                      low=0., high=1.,
                                      shape_or_val=False, assert_value_equality=False)
        check_single_tensor_operation('random_normal_variable', (2, 3), BACKENDS,
                                      mean=0., scale=1.,
                                      shape_or_val=False, assert_value_equality=False)

    @pytest.mark.skipif(K.backend() != 'tensorflow', reason='Not supported.')
    def test_batch_dot_shape(self):
        x_batch = K.ones(shape=(32, 20))
        y_batch = K.ones(shape=(32, 20))
        xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=1)
        assert_allclose(K.eval(xy_batch_dot), np.ones((32, 1)) * 20, atol=1e-05)
        xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=0)
        assert_allclose(K.eval(xy_batch_dot), np.ones((20, 1)) * 32, atol=1e-05)
        # making sure swapping axes when ndim == 2 works
        x_batch = K.ones(shape=(32, 20))
        y_batch = K.ones(shape=(20, 32))
        xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=(0, 1))
        assert_allclose(K.eval(xy_batch_dot), np.ones((20, 1)) * 32, atol=1e-05)
        xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=(1, 0))
        assert_allclose(K.eval(xy_batch_dot), np.ones((32, 1)) * 20, atol=1e-05)

    def test_shape_operations(self):
        check_two_tensor_operation('concatenate', (4, 3), (4, 2), BACKENDS,
                                   axis=-1, concat_args=True)

        check_single_tensor_operation('reshape', (4, 2), BACKENDS, shape=(8, 1))
        check_single_tensor_operation('permute_dimensions', (4, 2, 3), BACKENDS,
                                      pattern=(2, 0, 1))
        check_single_tensor_operation('repeat', (4, 1), BACKENDS, n=3)
        check_single_tensor_operation('flatten', (4, 1), BACKENDS)
        check_single_tensor_operation('batch_flatten', (20, 2, 5), BACKENDS,
                                      cntk_dynamicity=True)
        check_single_tensor_operation('expand_dims', (4, 3), BACKENDS, axis=-1)
        check_single_tensor_operation('expand_dims', (4, 3, 2), BACKENDS, axis=1)
        check_single_tensor_operation('squeeze', (4, 3, 1), BACKENDS, axis=2)
        check_single_tensor_operation('squeeze', (4, 1, 1), BACKENDS, axis=1)
        check_composed_tensor_operations('reshape', {'shape': (4, 3, 1, 1)},
                                         'squeeze', {'axis': 2},
                                         (4, 3, 1, 1), BACKENDS)

    def test_none_shape_operations(self):
        # Test shape inference when input
        # shape has `None` entries
        if K.backend() == 'theano':
            x = KTH.placeholder((3, None, 4))

            y = KTH.batch_flatten(x)
            if hasattr(y, '_keras_shape'):
                assert y._keras_shape == (3, None)

            y = KTH.flatten(x)
            if hasattr(y, '_keras_shape'):
                assert y._keras_shape == (None, )

    def test_repeat_elements(self):
        reps = 3
        for ndims in [1, 2, 3]:
            shape = np.arange(2, 2 + ndims)
            arr = np.arange(np.prod(shape)).reshape(shape)

            for rep_axis in range(ndims):
                np_rep = np.repeat(arr, reps, axis=rep_axis)
                check_single_tensor_operation('repeat_elements', arr, BACKENDS,
                                              rep=reps, axis=rep_axis,
                                              assert_value_with_ref=np_rep)

                if K.backend() != 'cntk':
                    shape = list(shape)
                    shape[rep_axis] = None
                    x = K.placeholder(shape=shape)
                    y = K.repeat_elements(x, reps, axis=rep_axis)
                    assert y._keras_shape == tuple(shape)
                    assert y._keras_shape == K.int_shape(y)

    def test_tile(self):
        shape = (3, 4)
        arr = np.arange(np.prod(shape)).reshape(shape)
        check_single_tensor_operation('tile', arr, BACKENDS, n=[2, 1])
        check_single_tensor_operation('tile', (2, 5), BACKENDS, n=[5, 2])

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
        z_list = [k.eval(k.gather(k.variable(ref), k.variable(inds, dtype='int32')))
                  for k in BACKENDS]

        assert_list_pairwise(z_list)
        assert_list_keras_shape(z_list)

        # test theano shape inference when
        # input shape has None entries
        if K.backend() == 'theano':
            x = K.placeholder(shape=(None, 3, 4))
            indices = K.placeholder(shape=(5, 6), dtype='int32')
            y = K.gather(x, indices)
            assert y._keras_shape == (5, 6, 3, 4)

    def test_value_manipulation(self):
        val = np.random.random((4, 2))
        for function_name in ['get_value', 'count_params',
                              'int_shape', 'get_variable_shape']:
            v_list = [getattr(k, function_name)(k.variable(val))
                      for k in BACKENDS]

            if function_name == 'get_value':
                assert_list_pairwise(v_list)
            else:
                assert_list_pairwise(v_list, shape=False, allclose=False, itself=True)

        # print_tensor
        check_single_tensor_operation('print_tensor', (), BACKENDS)
        check_single_tensor_operation('print_tensor', (2,), BACKENDS)
        check_single_tensor_operation('print_tensor', (4, 3), BACKENDS)
        check_single_tensor_operation('print_tensor', (1, 2, 3), BACKENDS)

    def test_elementwise_operations(self):
        check_single_tensor_operation('max', (4, 2), BACKENDS)
        check_single_tensor_operation('max', (4, 2), BACKENDS, axis=1, keepdims=True)

        check_single_tensor_operation('min', (4, 2), BACKENDS)
        check_single_tensor_operation('min', (4, 2), BACKENDS, axis=1, keepdims=True)
        check_single_tensor_operation('min', (4, 2, 3), BACKENDS, axis=[1, -1])

        check_single_tensor_operation('mean', (4, 2), BACKENDS)
        check_single_tensor_operation('mean', (4, 2), BACKENDS, axis=1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), BACKENDS, axis=-1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), BACKENDS, axis=[1, -1])

        check_single_tensor_operation('std', (4, 2), BACKENDS)
        check_single_tensor_operation('std', (4, 2), BACKENDS, axis=1, keepdims=True)
        check_single_tensor_operation('std', (4, 2, 3), BACKENDS, axis=[1, -1])

        check_single_tensor_operation('prod', (4, 2), BACKENDS)
        check_single_tensor_operation('prod', (4, 2), BACKENDS, axis=1, keepdims=True)
        check_single_tensor_operation('prod', (4, 2, 3), BACKENDS, axis=[1, -1])

        # cntk does not support cumsum and cumprod yet
        check_single_tensor_operation('cumsum', (4, 2), [KTF, KTH])
        check_single_tensor_operation('cumsum', (4, 2), [KTF, KTH], axis=1)

        check_single_tensor_operation('cumprod', (4, 2), [KTF, KTH])
        check_single_tensor_operation('cumprod', (4, 2), [KTF, KTH], axis=1)

        check_single_tensor_operation('any', (4, 2), BACKENDS)
        check_single_tensor_operation('any', (4, 2), BACKENDS, axis=1, keepdims=True)

        check_single_tensor_operation('all', (4, 2), BACKENDS)
        check_single_tensor_operation('all', (4, 2), BACKENDS, axis=1, keepdims=True)

        check_single_tensor_operation('argmax', (4, 2), BACKENDS)
        check_single_tensor_operation('argmax', (4, 2), BACKENDS, axis=1)

        check_single_tensor_operation('argmin', (4, 2), BACKENDS)
        check_single_tensor_operation('argmin', (4, 2), BACKENDS, axis=1)

        check_single_tensor_operation('square', (4, 2), BACKENDS)
        check_single_tensor_operation('abs', (4, 2), BACKENDS)
        check_single_tensor_operation('sqrt', (4, 2), BACKENDS)
        check_single_tensor_operation('exp', (4, 2), BACKENDS)
        # cntk return -85.1 for zero or negative number, not nan, so can't compare with other backend.
        check_single_tensor_operation('log', (4, 2), [KTH, KTF])
        check_single_tensor_operation('round', (4, 2), BACKENDS)
        check_single_tensor_operation('sign', (4, 2), BACKENDS)
        check_single_tensor_operation('pow', (4, 2), BACKENDS, a=3)
        check_single_tensor_operation('clip', (4, 2), BACKENDS, min_value=0.4,
                                      max_value=0.6)

        # two-tensor ops
        check_two_tensor_operation('equal', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('not_equal', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('greater', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('greater_equal', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('less', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('less_equal', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('maximum', (4, 2), (4, 2), BACKENDS)
        check_two_tensor_operation('minimum', (4, 2), (4, 2), BACKENDS)

    # cntk doesn't support gradient in this way
    def test_gradient(self):
        val = np.random.random((4, 2))
        x_list = [k.variable(val) for k in [KTH, KTF]]
        z_list = []
        zero_list = []
        for x, k in zip(x_list, [KTH, KTF]):
            exp = x * k.exp(x)
            loss = k.sum(exp)
            zero_loss = k.stop_gradient(loss)
            grad = k.gradients(loss, [exp])
            zero_grad = k.gradients(loss + zero_loss, [exp])
            z_list.append(k.eval(grad[0]))
            zero_list.append(k.eval(zero_grad[0]))

        assert_list_pairwise(z_list)
        assert_list_pairwise(zero_list)
        for i in range(len(z_list)):
            assert_allclose(zero_list[i], z_list[i], atol=1e-05)

    def test_stop_gradient(self):
        # This test checks the consistency of the stop_gradient backend API.
        # It doesn't check the functionality (which is checked at the
        # test_gradient test).
        val = np.random.random((4, 2))
        for k in BACKENDS:
            a = k.variable(val)
            b = k.square(a)
            c, d = k.stop_gradient([a, b])
            e = k.stop_gradient(b)

    # cntk currently not support function in this way, so can't test as this
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
            update = x * 2
            f = k.function([y], [exp], updates=[(x, update)])
            f_list.append(f)

        function_outputs_list = [f([input_val])[0] for f in f_list]
        assert_list_pairwise(function_outputs_list)

        new_val_list = [k.get_value(x) for x, k in zip(x_list, test_backend)]
        assert_list_pairwise(new_val_list)

    def test_function_tf_fetches(self):
        # Additional operations can be passed to tf.Session().run() via its
        # `fetches` arguments. In contrast to `updates` argument of
        # KTF.function() these do not have control dependency on `outputs`, so
        # they can run in parallel. Also they should not contribute to output of
        # KTF.function().

        x = KTF.variable(0.)
        y = KTF.variable(0.)
        x_placeholder = KTF.placeholder(shape=())
        y_placeholder = KTF.placeholder(shape=())

        f = KTF.function(inputs=[x_placeholder, y_placeholder],
                         outputs=[x_placeholder + y_placeholder],
                         updates=[(x, x_placeholder + 1.)],
                         fetches=[KTF.update(y, 5.)])
        output = f([10., 20.])
        assert output == [30.]
        assert KTF.get_session().run(fetches=[x, y]) == [11., 5.]

    def test_function_tf_feed_dict(self):
        # Additional substitutions can be passed to `tf.Session().run()` via its
        # `feed_dict` arguments. Note that the feed_dict is passed once in the
        # constructor but we can modify the values in the dictionary. Through
        # this feed_dict we can provide additional substitutions besides Keras
        # inputs.

        x = KTF.variable(0.)
        y = KTF.variable(0.)
        x_placeholder = KTF.placeholder(shape=())
        y_placeholder = KTF.placeholder(shape=())

        feed_dict = {y_placeholder: 3.}

        f = KTF.function(inputs=[x_placeholder],
                         outputs=[x_placeholder + 1.],
                         updates=[(x, x_placeholder + 10.)],
                         feed_dict=feed_dict,
                         fetches=[KTF.update(y, y_placeholder * 10.)])
        output = f([10.])
        assert output == [11.]
        assert KTF.get_session().run(fetches=[x, y]) == [20., 30.]

        # updated value in feed_dict will be modified within the K.function()
        feed_dict[y_placeholder] = 4.
        output = f([20.])
        assert output == [21.]
        assert KTF.get_session().run(fetches=[x, y]) == [30., 40.]

    def test_function_tf_string_input(self):
        # Test functions with string inputs.

        x_placeholder = KTF.placeholder(shape=(), dtype="string")
        x_identity = KTF.identity(x_placeholder)

        f = KTF.function(inputs=[x_placeholder], outputs=[x_identity])
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

        x_k = K.variable(x)
        h0_k = [K.variable(h0)]
        wi_k = K.variable(wi)
        wh_k = K.variable(wh)
        mask_k = K.variable(mask)

        def rnn_fn(x_k, h_k):
            assert len(h_k) == 1
            y_k = K.dot(x_k, wi_k) + K.dot(h_k[0], wh_k)
            return y_k, [y_k]

        # test default setup
        last_output_list = []
        outputs_list = []
        state_list = []

        kwargs_list = [
            {'go_backwards': False, 'mask': None},
            {'go_backwards': False, 'mask': None, 'unroll': True, 'input_length': timesteps},
            {'go_backwards': True, 'mask': None},
            {'go_backwards': True, 'mask': None, 'unroll': True, 'input_length': timesteps},
            {'go_backwards': False, 'mask': mask_k},
            {'go_backwards': False, 'mask': mask_k, 'unroll': True, 'input_length': timesteps},
        ]

        for (i, kwargs) in enumerate(kwargs_list):
            last_y1, y1, h1 = reference_operations.rnn(x, [wi, wh, None], h0, **kwargs)
            last_y2, y2, h2 = K.rnn(rnn_fn, x_k, h0_k, **kwargs)

            assert len(h2) == 1
            last_y2 = K.eval(last_y2)
            y2 = K.eval(y2)
            h1 = h1[:, -1]
            h2 = K.eval(h2[0])

            if kwargs['mask'] is not None:
                last_y1 = last_y1 * np.expand_dims(mask[:, -1], -1)
                last_y2 = last_y2 * np.expand_dims(mask[:, -1], -1)
                y1 = y1 * np.expand_dims(mask, -1)
                y2 = y2 * np.expand_dims(mask, -1)
                h1 = h1 * np.expand_dims(mask[:, -1], -1)
                h2 = h2 * np.expand_dims(mask[:, -1], -1)

            last_output_list.append(last_y2)
            outputs_list.append(y2)
            state_list.append(h2)

            if i % 2 == 0:
                assert_allclose(last_y1, last_y2, atol=1e-05)
                assert_allclose(y1, y2, atol=1e-05)
                assert_allclose(h1, h2, atol=1e-05)
            else:
                assert_allclose(last_output_list[i - 1], last_output_list[i], atol=1e-05)
                assert_allclose(outputs_list[i - 1], outputs_list[i], atol=1e-05)
                assert_allclose(state_list[i - 1], state_list[i], atol=1e-05)

    def test_rnn_additional_states(self):
        # implement a simple RNN with an additional state
        # whose shape is different from that of the output
        num_samples = 4
        input_dim = 5
        output_dim = 3
        timesteps = 6

        _, x = parse_shape_or_val((num_samples, timesteps, input_dim))
        _, h0 = parse_shape_or_val((num_samples, output_dim))
        _, wi = parse_shape_or_val((input_dim, output_dim))
        _, wh = parse_shape_or_val((output_dim, output_dim))
        mask = np.random.randint(2, size=(num_samples, timesteps))

        x_k = K.variable(x)
        h0_k = [K.variable(h0), K.variable(np.concatenate([h0, h0], axis=-1))]
        wi_k = K.variable(wi)
        wh_k = K.variable(wh)
        mask_k = K.variable(mask)

        def rnn_fn(x_k, h_k):
            assert len(h_k) == 2
            y_k = K.dot(x_k, wi_k) + K.dot(h_k[0], wh_k)
            return y_k, [y_k, K.concatenate([y_k, y_k], axis=-1)]

        # test default setup
        last_output_list = []
        outputs_list = []
        state_list = []

        kwargs_list = [
            {'go_backwards': False, 'mask': None},
            {'go_backwards': False, 'mask': None, 'unroll': True, 'input_length': timesteps},
            {'go_backwards': True, 'mask': None},
            {'go_backwards': True, 'mask': None, 'unroll': True, 'input_length': timesteps},
            {'go_backwards': False, 'mask': mask_k},
            {'go_backwards': False, 'mask': mask_k, 'unroll': True, 'input_length': timesteps},
        ]

        for (i, kwargs) in enumerate(kwargs_list):
            last_y1, y1, h1 = reference_operations.rnn(x, [wi, wh, None], h0, **kwargs)
            last_y2, y2, h2 = K.rnn(rnn_fn, x_k, h0_k, **kwargs)

            assert len(h2) == 2
            last_y2 = K.eval(last_y2)
            y2 = K.eval(y2)
            h11 = h1[:, -1]
            h12 = np.concatenate([h1[:, -1], h1[:, -1]], axis=-1)
            h21 = K.eval(h2[0])
            h22 = K.eval(h2[1])

            if kwargs['mask'] is not None:
                last_y1 = last_y1 * np.expand_dims(mask[:, -1], -1)
                last_y2 = last_y2 * np.expand_dims(mask[:, -1], -1)
                y1 = y1 * np.expand_dims(mask, -1)
                y2 = y2 * np.expand_dims(mask, -1)
                h11 = h11 * np.expand_dims(mask[:, -1], -1)
                h21 = h21 * np.expand_dims(mask[:, -1], -1)
                h12 = h12 * np.expand_dims(mask[:, -1], -1)
                h22 = h22 * np.expand_dims(mask[:, -1], -1)

            last_output_list.append(last_y2)
            outputs_list.append(y2)
            state_list.append((h21, h22))

            if i % 2 == 0:
                assert_allclose(last_y1, last_y2, atol=1e-05)
                assert_allclose(y1, y2, atol=1e-05)
                assert_allclose(h11, h21, atol=1e-05)
                assert_allclose(h12, h22, atol=1e-05)
            else:
                assert_allclose(last_output_list[i - 1], last_output_list[i], atol=1e-05)
                assert_allclose(outputs_list[i - 1], outputs_list[i], atol=1e-05)
                assert_allclose(state_list[i - 1][0], state_list[i][0], atol=1e-05)
                assert_allclose(state_list[i - 1][1], state_list[i][1], atol=1e-05)

    def test_rnn_no_states(self):
        # implement a simple RNN without states
        input_dim = 8
        output_dim = 4
        timesteps = 5

        _, x = parse_shape_or_val((32, timesteps, input_dim))
        _, wi = parse_shape_or_val((input_dim, output_dim))

        x_k = K.variable(x)
        wi_k = K.variable(wi)

        def rnn_fn(x_k, h_k):
            assert len(h_k) == 0
            y_k = K.dot(x_k, wi_k)
            return y_k, []

        last_y1, y1, h1 = reference_operations.rnn(x, [wi, None, None], None,
                                                   go_backwards=False, mask=None)
        last_y2, y2, h2 = K.rnn(rnn_fn, x_k, [],
                                go_backwards=False, mask=None)

        assert len(h2) == 0
        last_y2 = K.eval(last_y2)
        y2 = K.eval(y2)

        assert_allclose(last_y1, last_y2, atol=1e-05)
        assert_allclose(y1, y2, atol=1e-05)

    def legacy_test_rnn(self):
        # implement a simple RNN
        num_samples = 4
        input_dim = 5
        output_dim = 3
        timesteps = 6

        input_val = np.random.random((num_samples, timesteps, input_dim)).astype(np.float32)
        init_state_val = np.random.random((num_samples, output_dim)).astype(np.float32)
        W_i_val = np.random.random((input_dim, output_dim)).astype(np.float32)
        W_o_val = np.random.random((output_dim, output_dim)).astype(np.float32)
        np_mask = np.random.randint(2, size=(num_samples, timesteps))

        def rnn_step_fn(k):
            W_i = k.variable(W_i_val)
            W_o = k.variable(W_o_val)

            def step_function(x, states):
                assert len(states) == 1
                prev_output = states[0]
                output = k.dot(x, W_i) + k.dot(prev_output, W_o)
                return output, [output]

            return step_function

        # test default setup
        last_output_list = [[], [], [], [], [], []]
        outputs_list = [[], [], [], [], [], []]
        state_list = [[], [], [], [], [], []]

        for k in BACKENDS:
            rnn_fn = rnn_step_fn(k)
            inputs = k.variable(input_val)
            initial_states = [k.variable(init_state_val)]
            mask = k.variable(np_mask)

            kwargs_list = [
                {'go_backwards': False, 'mask': None},
                {'go_backwards': False, 'mask': None, 'unroll': True, 'input_length': timesteps},
                {'go_backwards': True, 'mask': None},
                {'go_backwards': True, 'mask': None, 'unroll': True, 'input_length': timesteps},
                {'go_backwards': False, 'mask': mask},
                {'go_backwards': False, 'mask': mask, 'unroll': True, 'input_length': timesteps},
            ]

            for (i, kwargs) in enumerate(kwargs_list):
                last_output, outputs, new_states = k.rnn(rnn_fn, inputs,
                                                         initial_states,
                                                         **kwargs)

                last_output_list[i].append(k.eval(last_output))
                outputs_list[i].append(k.eval(outputs))
                assert len(new_states) == 1
                state_list[i].append(k.eval(new_states[0]))

        assert_list_pairwise(last_output_list[0], shape=False, atol=1e-04)
        assert_list_pairwise(outputs_list[0], shape=False, atol=1e-04)
        assert_list_pairwise(state_list[0], shape=False, atol=1e-04)
        assert_list_pairwise(last_output_list[2], shape=False, atol=1e-04)
        assert_list_pairwise(outputs_list[2], shape=False, atol=1e-04)
        assert_list_pairwise(state_list[2], shape=False, atol=1e-04)

        for l, u_l in zip(last_output_list[0], last_output_list[1]):
            assert_allclose(l, u_l, atol=1e-04)

        for o, u_o in zip(outputs_list[0], outputs_list[1]):
            assert_allclose(o, u_o, atol=1e-04)

        for s, u_s in zip(state_list[0], state_list[1]):
            assert_allclose(s, u_s, atol=1e-04)

        for b_l, b_u_l in zip(last_output_list[2], last_output_list[3]):
            assert_allclose(b_l, b_u_l, atol=1e-04)

        for b_o, b_u_o in zip(outputs_list[2], outputs_list[3]):
            assert_allclose(b_o, b_u_o, atol=1e-04)

        for b_s, b_u_s in zip(state_list[2], state_list[3]):
            assert_allclose(b_s, b_u_s, atol=1e-04)

        for m_l, u_m_l, k in zip(last_output_list[4], last_output_list[5], BACKENDS):
            if k == KTF:
                m_l = m_l * np.expand_dims(np_mask[:, -1], -1)
                u_m_l = u_m_l * np.expand_dims(np_mask[:, -1], -1)
            assert_allclose(m_l, u_m_l, atol=1e-04)

        for m_o, u_m_o, k in zip(outputs_list[4], outputs_list[5], BACKENDS):
            if k == KTF:
                m_o = m_o * np.expand_dims(np_mask, -1)
                u_m_o = u_m_o * np.expand_dims(np_mask, -1)
            assert_allclose(m_o, u_m_o, atol=1e-04)

        for m_s, u_m_s, k in zip(state_list[4], state_list[5], BACKENDS):
            assert_allclose(m_s, u_m_s, atol=1e-04)

    def legacy_test_rnn_no_states(self):
        # implement a simple RNN without states
        input_dim = 8
        output_dim = 4
        timesteps = 5

        input_val = np.random.random((32, timesteps, input_dim))
        W_i_val = np.random.random((input_dim, output_dim))

        def rnn_step_fn(k):
            W_i = k.variable(W_i_val)

            def step_function(x, states):
                assert len(states) == 0
                output = k.dot(x, W_i)
                return output, []

            return step_function

        # test default setup
        last_output_list = []
        outputs_list = []

        for k in BACKENDS:
            rnn_fn = rnn_step_fn(k)
            inputs = k.variable(input_val)
            initial_states = []
            last_output, outputs, new_states = k.rnn(rnn_fn, inputs,
                                                     initial_states,
                                                     go_backwards=False,
                                                     mask=None)
            last_output_list.append(k.eval(last_output))
            outputs_list.append(k.eval(outputs))
            assert len(new_states) == 0

        assert_list_pairwise(last_output_list, shape=False)
        assert_list_pairwise(outputs_list, shape=False)

    @pytest.mark.parametrize('x_np,axis,keepdims', [
        (np.array([1.1, 0.8, 0.9]), 0, False),
        (np.array([[1.1, 0.8, 0.9]]), 0, False),
        (np.array([[1.1, 0.8, 0.9]]), 1, False),
        (np.array([[1.1, 0.8, 0.9]]), -1, False),
        (np.array([[1.1, 0.8, 0.9]]), 1, True),
        (np.array([[1.1], [1.2]]), 0, False),
        (np.array([[1.1], [1.2]]), 1, False),
        (np.array([[1.1], [1.2]]), -1, False),
        (np.array([[1.1], [1.2]]), -1, True),
        (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), None, False),
        (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), 0, False),
        (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), 1, False),
        (np.array([[1.1, 1.2, 1.3], [0.9, 0.7, 1.4]]), -1, False),
    ])
    def test_logsumexp(self, x_np, axis, keepdims):
        '''
        Check if K.logsumexp works properly for values close to one.
        '''
        for k in BACKENDS:
            x = k.variable(x_np)
            assert_allclose(k.eval(k.logsumexp(x, axis=axis, keepdims=keepdims)),
                            np.log(np.sum(np.exp(x_np), axis=axis, keepdims=keepdims)),
                            rtol=1e-5)

    def test_logsumexp_optim(self):
        '''
        Check if optimization works.
        '''
        for k in [KTF]:
            x_np = np.array([1e+4, 1e-4])
            assert_allclose(k.eval(k.logsumexp(k.variable(x_np), axis=0)),
                            1e4,
                            rtol=1e-5)

    def test_switch(self):
        # scalar
        val = np.random.random()
        z_list = []
        for k in BACKENDS:
            x = k.variable(val)
            x = k.switch(k.greater_equal(x, 0.5), x * 0.1, x * 0.2)
            z_list.append(k.eval(x))
        assert_list_pairwise(z_list)
        # non scalar
        shapes = []
        shapes.append([(4, 3, 2), (4, 3, 2), (4, 3, 2)])
        shapes.append([(4, 3,), (4, 3, 2), (4, 3, 2)])
        shapes.append([(4,), (4, 3, 2), (4, 3, 2)])
        for s in shapes:
            z_list = []
            arrays = list(map(np.random.random, s))
            for k in BACKENDS:
                x, then_expr, else_expr = map(k.variable, arrays)
                cond = k.greater_equal(x, 0.5)
                z_list.append(k.eval(k.switch(cond, then_expr, else_expr)))
            assert_list_pairwise(z_list)

    def test_dropout(self):
        val = np.random.random((100, 100))
        z_list = [k.eval(k.dropout(k.variable(val), level=0.2))
                  for k in BACKENDS]
        assert_list_pairwise(z_list, allclose=False)
        # dropout patterns are different, only check mean
        for i in range(len(z_list) - 1):
            assert np.abs(z_list[i].mean() - z_list[i + 1].mean()) < 0.05

        z_list = [k.eval(k.dropout(k.variable(val), level=0.2,
                                   noise_shape=list(val.shape)))
                  for k in BACKENDS]
        assert_list_pairwise(z_list, allclose=False)
        # dropout patterns are different, only check mean
        for i in range(len(z_list) - 1):
            assert np.abs(z_list[i].mean() - z_list[i + 1].mean()) < 0.05

        # Test invalid use cases
        for k in BACKENDS:
            with pytest.raises(ValueError):
                z = k.dropout(k.variable(val), level=-0.5)

    def test_nn_operations(self):
        check_single_tensor_operation('relu', (4, 2), BACKENDS, alpha=0.1, max_value=0.5)
        check_single_tensor_operation('softplus', (4, 10), BACKENDS)
        check_single_tensor_operation('elu', (4, 10), BACKENDS, alpha=0.5)

        check_single_tensor_operation('sigmoid', (4, 2), BACKENDS)
        check_single_tensor_operation('hard_sigmoid', (4, 2), BACKENDS)
        check_single_tensor_operation('tanh', (4, 2), BACKENDS)

        check_single_tensor_operation('softmax', (4, 10), BACKENDS)
        check_single_tensor_operation('softmax', (4, 5, 3, 10), BACKENDS, axis=2)

        check_two_tensor_operation('binary_crossentropy', (4, 2), (4, 2), BACKENDS, from_logits=True)
        # cross_entropy call require the label is a valid probability distribution,
        # otherwise it is garbage in garbage out...
        # due to the algo difference, we can't guarantee CNTK has the same result on the garbage input.
        # so create a separate test case for valid label input
        check_two_tensor_operation('categorical_crossentropy', (4, 2), (4, 2), [KTH, KTF], from_logits=True)
        xval = np.asarray([[0.26157712, 0.0432167], [-0.43380741, 0.30559841],
                           [0.20225059, -0.38956559], [-0.13805378, 0.08506755]], dtype=np.float32)
        yval = np.asarray([[0.46221867, 0.53778133], [0.51228984, 0.48771016],
                           [0.64916514, 0.35083486], [0.47028078, 0.52971922]], dtype=np.float32)
        check_two_tensor_operation('categorical_crossentropy', yval, xval,
                                   BACKENDS, cntk_two_dynamicity=True, from_logits=True)
        check_two_tensor_operation('binary_crossentropy', (4, 2), (4, 2), BACKENDS, from_logits=False)
        check_two_tensor_operation('categorical_crossentropy', (4, 2), (4, 2), BACKENDS, from_logits=False)

        check_single_tensor_operation('l2_normalize', (4, 3), BACKENDS, axis=-1)
        check_single_tensor_operation('l2_normalize', (4, 3), BACKENDS, axis=1)

    def test_in_top_k(self):
        batch_size = 20
        num_classes = 10

        # Random prediction test case
        predictions = np.random.random((batch_size, num_classes)).astype('float32')
        targets = np.random.randint(num_classes, size=batch_size, dtype='int32')

        # (k == 0 or k > num_classes) does not raise an error but just return an unmeaningful tensor.
        for k in range(num_classes + 1):
            z_list = [b.eval(b.in_top_k(b.variable(predictions, dtype='float32'),
                                        b.variable(targets, dtype='int32'), k))
                      for b in [KTH, KTF]]
            assert_list_pairwise(z_list)

        # Identical prediction test case:
        # randomly set half of the predictions to an identical value
        num_identical = num_classes // 2
        for i in range(batch_size):
            idx_identical = np.random.choice(num_classes, size=num_identical, replace=False)
            predictions[i, idx_identical] = predictions[i, 0]
        targets = np.zeros(batch_size, dtype='int32')

        for k in range(1, num_classes + 1):
            z_list = [b.eval(b.in_top_k(b.variable(predictions, dtype='float32'),
                                        b.variable(targets, dtype='int32'), k))
                      for b in [KTH, KTF]]
            assert_list_pairwise(z_list)

    @pytest.mark.parametrize('op,input_shape,kernel_shape,padding,data_format', [
        ('conv1d', (2, 8, 2), (3, 2, 3), 'same', 'channels_last'),
        ('conv1d', (1, 8, 2), (3, 2, 3), 'valid', 'channels_last'),
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
        k = K.backend()
        _, x = parse_shape_or_val(input_shape)
        _, w = parse_shape_or_val(kernel_shape)
        y1 = reference_operations.conv(x, w, padding, data_format)
        y2 = check_two_tensor_operation(
            op, x, w, [KTH if k == 'theano' else KC if k == 'cntk' else KTF],
            padding=padding, data_format=data_format,
            cntk_dynamicity=True, return_results=True)
        assert_allclose(y1, y2, atol=1e-05)

    @pytest.mark.parametrize('op,input_shape,kernel_shape,padding,data_format', [
        ('depthwise_conv2d', (2, 3, 4, 5), (3, 3, 3, 2), 'same', 'channels_first'),
        ('depthwise_conv2d', (2, 3, 5, 6), (4, 3, 3, 4), 'valid', 'channels_first'),
        ('depthwise_conv2d', (1, 6, 5, 3), (3, 4, 3, 2), 'valid', 'channels_last'),
        ('depthwise_conv2d', (1, 7, 6, 3), (3, 3, 3, 4), 'same', 'channels_last'),
    ])
    def test_depthwise_conv(self, op, input_shape, kernel_shape, padding, data_format):
        k = K.backend()
        _, x = parse_shape_or_val(input_shape)
        _, w = parse_shape_or_val(kernel_shape)
        y1 = reference_operations.depthwise_conv(x, w, padding, data_format)
        y2 = check_two_tensor_operation(
            op, x, w, [KTH if k == 'theano' else KC if k == 'cntk' else KTF],
            padding=padding, data_format=data_format,
            cntk_dynamicity=True, return_results=True)
        assert_allclose(y1, y2, atol=1e-05)

    @pytest.mark.parametrize('op,input_shape,pool_size,strides,padding,data_format,pool_mode', [
        ('pool2d', (2, 3, 7, 7), (3, 3), (1, 1), 'same', 'channels_first', 'avg'),
        ('pool2d', (3, 3, 8, 5), (2, 3), (1, 1), 'valid', 'channels_first', 'max'),
        ('pool2d', (2, 9, 5, 3), (3, 2), (1, 1), 'valid', 'channels_last', 'avg'),
        ('pool2d', (3, 6, 7, 3), (3, 3), (1, 1), 'same', 'channels_last', 'max'),
        ('pool3d', (2, 3, 7, 7, 7), (3, 3, 3), (1, 1, 1), 'same', 'channels_first', 'avg'),
        ('pool3d', (3, 3, 8, 5, 9), (2, 3, 2), (1, 1, 1), 'valid', 'channels_first', 'max'),
        ('pool3d', (2, 8, 9, 5, 3), (3, 2, 3), (1, 1, 1), 'valid', 'channels_last', 'avg'),
        ('pool3d', (3, 5, 6, 7, 3), (3, 3, 3), (1, 1, 1), 'same', 'channels_last', 'max'),
    ])
    def test_pool(self, op, input_shape, pool_size, strides, padding, data_format, pool_mode):
        k = K.backend()
        _, x = parse_shape_or_val(input_shape)
        y1 = reference_operations.pool(x, pool_size, strides, padding, data_format, pool_mode)
        y2 = check_single_tensor_operation(
            op, x, [KTH if k == 'theano' else KC if k == 'cntk' else KTF],
            pool_size=pool_size, strides=strides,
            padding=padding, data_format=data_format, pool_mode=pool_mode,
            cntk_dynamicity=True, return_results=True)
        assert_allclose(y1, y2, atol=1e-05)

    def legacy_test_conv1d(self):
        # channels_last input shape: (n, length, input_depth)
        input_shape = (4, 8, 2)
        kernel_shape = (3, 2, 3)
        for strides in [1, 2]:
            check_two_tensor_operation('conv1d', input_shape, kernel_shape,
                                       BACKENDS, cntk_dynamicity=True,
                                       strides=strides,
                                       data_format='channels_last')

    def legacy_test_conv2d(self):
        # TF kernel shape: (rows, cols, input_depth, depth)
        # channels_first input shape: (n, input_depth, rows, cols)
        for (input_shape, kernel_shape, data_format) in [
                ((2, 3, 4, 5), (2, 2, 3, 4), 'channels_first'),
                ((2, 3, 5, 6), (4, 3, 3, 4), 'channels_first'),
                ((1, 6, 5, 3), (3, 3, 3, 2), 'channels_last')]:
            check_two_tensor_operation('conv2d', input_shape, kernel_shape,
                                       BACKENDS, cntk_dynamicity=True,
                                       data_format=data_format)

    def legacy_test_depthwise_conv_2d(self):
        # TF kernel shape: (rows, cols, input_depth, depth_multiplier)
        # channels_first input shape: (n, input_depth, rows, cols)
        for (input_shape, kernel_shape, data_format) in [
                ((2, 3, 4, 5), (2, 2, 3, 4), 'channels_first'),
                ((2, 3, 5, 6), (4, 3, 3, 4), 'channels_first'),
                ((1, 6, 5, 3), (3, 3, 3, 2), 'channels_last')]:
            check_two_tensor_operation('depthwise_conv2d',
                                       input_shape, kernel_shape,
                                       BACKENDS, cntk_dynamicity=True,
                                       data_format=data_format)

    def legacy_test_conv3d(self):
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        # TH kernel shape: (depth, input_depth, x, y, z)
        # TF kernel shape: (x, y, z, input_depth, depth)
        for (input_shape, kernel_shape, data_format) in [
                ((2, 3, 4, 5, 4), (2, 2, 2, 3, 4), 'channels_first'),
                ((2, 3, 5, 4, 6), (3, 2, 4, 3, 4), 'channels_first'),
                ((1, 2, 2, 2, 1), (2, 2, 2, 1, 1), 'channels_last')]:
            check_two_tensor_operation('conv3d', input_shape, kernel_shape,
                                       BACKENDS, cntk_dynamicity=True,
                                       data_format=data_format)

    @pytest.mark.parametrize('op,input_shape,kernel_shape,depth_multiplier,padding,data_format', [
        ('separable_conv1d', (2, 8, 2), (3,), 1, 'same', 'channels_last'),
        ('separable_conv1d', (1, 8, 2), (3,), 2, 'valid', 'channels_last'),
        ('separable_conv2d', (2, 3, 4, 5), (3, 3), 1, 'same', 'channels_first'),
        ('separable_conv2d', (2, 3, 5, 6), (4, 3), 2, 'valid', 'channels_first'),
        ('separable_conv2d', (1, 6, 5, 3), (3, 4), 1, 'valid', 'channels_last'),
        ('separable_conv2d', (1, 7, 6, 3), (3, 3), 2, 'same', 'channels_last'),
    ])
    def test_separable_conv(self, op, input_shape, kernel_shape, depth_multiplier, padding, data_format):
        input_depth = input_shape[1] if data_format == 'channels_first' else input_shape[-1]
        _, x = parse_shape_or_val(input_shape)
        _, depthwise = parse_shape_or_val(kernel_shape + (input_depth, depth_multiplier))
        _, pointwise = parse_shape_or_val((1,) * len(kernel_shape) + (input_depth * depth_multiplier, 7))
        y1 = reference_operations.separable_conv(x, depthwise, pointwise, padding, data_format)
        if K.backend() == 'cntk':
            y2 = cntk_func_three_tensor(
                op, input_shape,
                depthwise, pointwise,
                padding=padding, data_format=data_format)([x])[0]
        else:
            y2 = K.eval(getattr(K, op)(
                K.variable(x),
                K.variable(depthwise), K.variable(pointwise),
                padding=padding, data_format=data_format))
        assert_allclose(y1, y2, atol=1e-05)

    def legacy_test_pool2d(self):
        check_single_tensor_operation('pool2d', (5, 10, 12, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 2), strides=(1, 1), padding='valid')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 2), strides=(1, 1), padding='valid')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 2), strides=(1, 1), pool_mode='avg')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 3), strides=(1, 1), padding='valid')

        check_single_tensor_operation('pool2d', (2, 7, 7, 5),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(3, 3), strides=(1, 1),
                                      padding='same', pool_mode='avg')

    def legacy_test_pool3d(self):
        check_single_tensor_operation('pool3d', (5, 10, 12, 5, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 2, 2), strides=(1, 1, 1), padding='valid')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 2, 2), strides=(1, 1, 1), pool_mode='avg')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3),
                                      BACKENDS, cntk_dynamicity=True,
                                      pool_size=(2, 3, 2), strides=(1, 1, 1), padding='valid')

        check_single_tensor_operation('pool3d', (2, 6, 6, 6, 3), [KTH, KTF], pool_size=(3, 3, 3),
                                      strides=(1, 1, 1), padding='same', pool_mode='avg')

    def test_random_normal(self):
        # test standard normal as well as a normal with a different set of parameters
        for k in BACKENDS:
            for mean, std in [(0., 1.), (-10., 5.)]:
                rand = k.eval(k.random_normal((300, 200), mean=mean, stddev=std, seed=1337))
                assert rand.shape == (300, 200)
                assert np.abs(np.mean(rand) - mean) < std * 0.015
                assert np.abs(np.std(rand) - std) < std * 0.015

                # test that random_normal also generates different values when used within a function
                r = k.random_normal((1,), mean=mean, stddev=std, seed=1337)
                samples = [k.eval(r) for _ in range(60000)]
                assert np.abs(np.mean(samples) - mean) < std * 0.015
                assert np.abs(np.std(samples) - std) < std * 0.015

    def test_random_uniform(self):
        min_val = -1.
        max_val = 1.
        for k in BACKENDS:
            rand = k.eval(k.random_uniform((200, 100), min_val, max_val))
            assert rand.shape == (200, 100)
            assert np.abs(np.mean(rand)) < 0.015
            assert max_val - 0.015 < np.max(rand) <= max_val
            assert min_val + 0.015 > np.min(rand) >= min_val

            r = k.random_uniform((1,), minval=min_val, maxval=max_val)
            samples = [k.eval(r) for _ in range(20000)]
            assert np.abs(np.mean(samples)) < 0.015
            assert max_val - 0.015 < np.max(samples) <= max_val
            assert min_val + 0.015 > np.min(samples) >= min_val

    def test_random_binomial(self):
        p = 0.5
        for k in BACKENDS:
            rand = k.eval(k.random_binomial((200, 100), p))
            assert rand.shape == (200, 100)
            assert np.abs(np.mean(rand) - p) < 0.015
            assert np.max(rand) == 1
            assert np.min(rand) == 0

            r = k.random_binomial((1,), p)
            samples = [k.eval(r) for _ in range(20000)]
            assert np.abs(np.mean(samples) - p) < 0.015
            assert np.max(samples) == 1
            assert np.min(samples) == 0

    def test_truncated_normal(self):
        mean = 0.
        std = 1.
        min_val = -2.
        max_val = 2.
        for k in BACKENDS:
            rand = k.eval(k.truncated_normal((300, 200), mean=mean, stddev=std, seed=1337))
            assert rand.shape == (300, 200)
            assert np.abs(np.mean(rand) - mean) < 0.015
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

        if K.backend() != 'theano':
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
        for (input_shape, pool_size) in zip([(5, 10, 12, 3), (5, 10, 12, 6, 3)], [(2, 2), (2, 2, 2)]):
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
                                          BACKENDS, cntk_dynamicity=True,
                                          height_factor=2,
                                          width_factor=2,
                                          data_format=data_format)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        for k in BACKENDS:
            with pytest.raises(ValueError):
                k.resize_images(k.variable(xval), 2, 2,
                                data_format='channels_middle')

    def test_resize_volumes(self):
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5, 5)
            if data_format == 'channels_first':
                x_shape = (2, 3) + shape
            elif data_format == 'channels_last':
                x_shape = (2,) + shape + (3,)
            check_single_tensor_operation('resize_volumes', x_shape,
                                          BACKENDS, cntk_dynamicity=True,
                                          depth_factor=2,
                                          height_factor=2,
                                          width_factor=2,
                                          data_format=data_format)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        for k in BACKENDS:
            with pytest.raises(ValueError):
                k.resize_volumes(k.variable(xval), 2, 2, 2,
                                 data_format='channels_middle')

    def test_temporal_padding(self):
        check_single_tensor_operation('temporal_padding', (4, 3, 3),
                                      BACKENDS)
        check_single_tensor_operation('temporal_padding', (2, 3, 4),
                                      BACKENDS, padding=(1, 2))

    def test_spatial_2d_padding(self):
        padding = ((1, 2), (2, 1))
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5)
            if data_format == 'channels_first':
                x_shape = (1, 3) + shape
            else:
                x_shape = (1,) + shape + (3,)
            check_single_tensor_operation('spatial_2d_padding', x_shape, BACKENDS,
                                          padding=padding, data_format=data_format)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        for k in BACKENDS:
            with pytest.raises(ValueError):
                k.spatial_2d_padding(k.variable(xval), padding=padding,
                                     data_format='channels_middle')

    def test_spatial_3d_padding(self):
        padding = ((1, 2), (2, 1), (1, 2))
        for data_format in ['channels_first', 'channels_last']:
            shape = (5, 5, 5)
            if data_format == 'channels_first':
                x_shape = (1, 3) + shape
            else:
                x_shape = (1,) + shape + (3,)
            check_single_tensor_operation('spatial_3d_padding', x_shape, BACKENDS,
                                          padding=padding, data_format=data_format)

        # Test invalid use cases
        xval = np.random.random(x_shape)
        for k in BACKENDS:
            with pytest.raises(ValueError):
                k.spatial_3d_padding(k.variable(xval), padding=padding,
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
                                           BACKENDS, cntk_dynamicity=True,
                                           data_format=data_format)

            if data_format == 'channels_first':
                x_shape = (20, 6, 10)
            else:
                x_shape = (20, 10, 6)
            check_two_tensor_operation('bias_add', x_shape, (10, 6),
                                       BACKENDS, cntk_dynamicity=True,
                                       data_format=data_format)

        # Test invalid use cases
        for k in BACKENDS:
            x = k.variable(np.random.random(x_shape))
            b = k.variable(np.random.random(bias_shape))
            with pytest.raises(ValueError):
                k.bias_add(x, b, data_format='channels_middle')

    def test_batchnorm(self):
        shape = (2, 3)
        for data_format in ['channels_first', 'channels_last']:
            if data_format == 'channels_first':
                x_shape = (1, 4) + shape
            else:
                x_shape = (1,) + shape + (4,)
            x_val = np.random.random(x_shape).astype(np.float32)
            xth = KTH.variable(x_val)
            xtf = KTF.variable(x_val)
            xc = KC.placeholder(x_shape)
            zth, _, _ = KTH.normalize_batch_in_training(xth, None, None,
                                                        reduction_axes='per-activation')
            ztf, _, _ = KTF.normalize_batch_in_training(xtf, None, None,
                                                        reduction_axes=[0, 1, 2, 3])
            zc, _, _ = KC.normalize_batch_in_training(xc, None, None,
                                                      reduction_axes=[0, 1, 2, 3])
            zth = KTH.eval(zth)
            ztf = KTF.eval(ztf)
            zc = KC.function([xc], [zc])([x_val])[0]
            assert zth.shape == ztf.shape
            assert zth.shape == zc.shape

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
        res = K.eval(K.ctc_batch_cost(k_labels, k_inputs, k_input_lens, k_label_lens))
        assert_allclose(res[0, :] if K.backend() == 'theano' else res[:, 0], ref, atol=1e-05)

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
        res = K.eval(K.ctc_batch_cost(k_labels, k_inputs, k_input_lens, k_label_lens))
        assert_allclose(res[0, :] if K.backend() == 'theano' else res[:, 0], ref, atol=1e-05)

    '''only tensorflow tested, need special handle'''

    def test_ctc_decode_greedy(self):
        # Test adapted from tensorflow
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
        input_log_prob_matrix_0 = np.log(input_prob_matrix_0)

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
        inputs = KTF.variable(np.asarray(inputs).transpose((1, 0, 2)))
        # batch_size length vector of sequence_lengths
        input_length = KTF.variable(np.array([seq_len_0, seq_len_1], dtype=np.int32))

        # batch_size length vector of negative log probabilities
        log_prob_truth = np.array([
            np.sum(-np.log([1.0, 0.6, 0.6, 0.9])),
            np.sum(-np.log([0.9, 0.9, 0.9, 0.9, 0.9]))
        ], np.float32)[:, np.newaxis]

        # keras output, unlike tensorflow, is a dense (not sparse) tensor
        decode_truth = np.array([[0, 1, -1], [1, 1, 0]])

        decode_pred_tf, log_prob_pred_tf = KTF.ctc_decode(inputs,
                                                          input_length,
                                                          greedy=True)

        assert len(decode_pred_tf) == 1

        decode_pred = KTF.eval(decode_pred_tf[0])
        log_prob_pred = KTF.eval(log_prob_pred_tf)

        assert np.alltrue(decode_truth == decode_pred)
        assert np.allclose(log_prob_truth, log_prob_pred)

    '''tensorflow only, need special handle'''

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

        # len max_time_steps array of batch_size x depth matrices
        inputs = ([input_prob_matrix_0[t, :][np.newaxis, :]
                   for t in range(seq_len_0)] +  # Pad to max_time_steps = 8
                  2 * [np.zeros((1, depth), dtype=np.float32)])

        inputs = KTF.variable(np.asarray(inputs).transpose((1, 0, 2)))

        # batch_size length vector of sequence_lengths
        input_length = KTF.variable(np.array([seq_len_0], dtype=np.int32))
        # batch_size length vector of negative log probabilities
        log_prob_truth = np.array([
            0.584855,  # output beam 0
            0.389139  # output beam 1
        ], np.float32)[np.newaxis, :]

        decode_truth = [np.array([1, 0]), np.array([0, 1, 0])]

        beam_width = 2
        top_paths = 2

        decode_pred_tf, log_prob_pred_tf = KTF.ctc_decode(inputs,
                                                          input_length,
                                                          greedy=False,
                                                          beam_width=beam_width,
                                                          top_paths=top_paths)

        assert len(decode_pred_tf) == top_paths

        log_prob_pred = KTF.eval(log_prob_pred_tf)

        for i in range(top_paths):
            assert np.alltrue(decode_truth[i] == KTF.eval(decode_pred_tf[i]))

        assert np.allclose(log_prob_truth, log_prob_pred)

    def test_one_hot(self):
        input_length = 10
        num_classes = 20
        batch_size = 30
        indices = np.random.randint(0, num_classes, size=(batch_size, input_length))
        oh = np.eye(num_classes)[indices]
        for k in BACKENDS:
            koh = k.eval(k.one_hot(k.variable(indices, dtype='int32'), num_classes))
            assert np.all(koh == oh)

    def test_sparse_dot(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()

        W = np.random.random((5, 4))
        # cntk not support it yet
        backends = [KTF]
        if KTH.th_sparse_module:
            # Theano has some dependency issues for sparse
            backends.append(KTH)

        for k in backends:
            t_W = k.variable(W)
            k_s = k.eval(k.dot(k.variable(x_sparse), t_W))
            k_d = k.eval(k.dot(k.variable(x_dense), t_W))

            assert k_s.shape == k_d.shape
            assert_allclose(k_s, k_d, atol=1e-05)

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

        # cntk not support it yet
        backends = [KTF]
        if KTH.th_sparse_module:
            # Theano has some dependency issues for sparse
            backends.append(KTH)

        for k in backends:
            k_s = k.concatenate([k.variable(x_sparse_1), k.variable(x_sparse_2)])
            assert k.is_sparse(k_s)

            k_s_d = k.eval(k_s)

            k_d = k.eval(k.concatenate([k.variable(x_dense_1), k.variable(x_dense_2)]))

            assert k_s_d.shape == k_d.shape
            assert_allclose(k_s_d, k_d, atol=1e-05)

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

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported.')
    def test_foldl(self):
        x = np.random.rand(10, 3).astype(np.float32)
        kx = K.eval(K.foldl(lambda a, b: a + b, K.variable(x)))

        assert (3,) == kx.shape
        assert_allclose(x.sum(axis=0), kx, atol=1e-05)

    @pytest.mark.skipif(K.backend() == 'cntk', reason='Not supported.')
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

    def test_arange(self):
        for test_value in (-20, 0, 1, 10):
            a_list = []
            dtype_list = []
            # cntk has issue with negative number
            for k in [KTH, KTF]:
                t = k.arange(test_value)
                a = k.eval(t)
                assert np.array_equal(a, np.arange(test_value))
                dtype_list.append(k.dtype(t))
                a_list.append(a)

            for i in range(len(a_list) - 1):
                assert np.array_equal(a_list[i], a_list[i + 1])

        for start, stop, step in ((0, 5, 1), (-5, 5, 2), (0, 1, 2)):
            a_list = []
            for k in [KTH, KTF]:
                a = k.eval(k.arange(start, stop, step))
                assert np.array_equal(a, np.arange(start, stop, step))
                a_list.append(a)
            for i in range(len(a_list) - 1):
                assert np.array_equal(a_list[i], a_list[i + 1])

        for dtype in ('int32', 'int64', 'float32', 'float64'):
            for k in [KTH, KTF]:
                t = k.arange(10, dtype=dtype)
                assert k.dtype(t) == dtype

        for k in [KTH, KTF]:
            start = k.constant(1, dtype='int32')
            t = k.arange(start)
            assert len(k.eval(t)) == 1

            start = k.constant(-1, dtype='int32')
            t = k.arange(start)
            assert len(k.eval(t)) == 0

    def test_in_train_phase(self):
        for training in [True, False]:
            check_two_tensor_operation('in_train_phase', (3, 3), (2, 2), [KTH, KTF],
                                       training=training)
            check_two_tensor_operation('in_train_phase', (2, 3), (2, 3), BACKENDS,
                                       training=training)

    def test_in_test_phase(self):
        for training in [True, False]:
            check_two_tensor_operation('in_test_phase', (3, 3), (2, 2), [KTH, KTF],
                                       training=training)
            check_two_tensor_operation('in_test_phase', (2, 3), (2, 3), BACKENDS,
                                       training=training)

    def test_setfloatx_incorrect_values(self):
        # Keep track of the old value
        old_floatx = floatx()
        # Try some incorrect values
        initial = floatx()
        for value in ['', 'beerfloat', 123]:
            with pytest.raises(ValueError):
                set_floatx(value)
        assert floatx() == initial
        # Restore old value
        set_floatx(old_floatx)

    def test_setfloatx_correct_values(self):
        # Keep track of the old value
        old_floatx = floatx()
        # Check correct values
        for value in ['float16', 'float32', 'float64']:
            set_floatx(value)
            assert floatx() == value
        # Restore old value
        set_floatx(old_floatx)

    @pytest.mark.skipif((K.backend() == 'cntk'),
                        reason='cntk does not support float16')
    def test_set_floatx(self):
        """
        Make sure that changes to the global floatx are effectively
        taken into account by the backend.
        """
        # Keep track of the old value
        old_floatx = floatx()

        set_floatx('float16')
        var = variable([10])
        check_dtype(var, 'float16')

        set_floatx('float64')
        var = variable([10])
        check_dtype(var, 'float64')

        # Restore old value
        set_floatx(old_floatx)

    def test_dtype(self):
        assert K.dtype(K.variable(1, dtype='float64')) == 'float64'
        assert K.dtype(K.variable(1, dtype='float32')) == 'float32'
        assert K.dtype(K.variable(1, dtype='float16')) == 'float16'

    def test_variable_support_bool_dtype(self):
        # Github issue: 7819
        if K.backend() == 'tensorflow':
            assert K.dtype(K.variable(1, dtype='int16')) == 'int16'
            assert K.dtype(K.variable(False, dtype='bool')) == 'bool'
            with pytest.raises(TypeError):
                K.variable('', dtype='unsupported')


if __name__ == '__main__':
    pytest.main([__file__])
