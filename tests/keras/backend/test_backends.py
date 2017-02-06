import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sparse

from keras import backend as K
from keras.backend import theano_backend as KTH, floatx, set_floatx, variable
from keras.backend import tensorflow_backend as KTF
from keras.utils.np_utils import convert_kernel


def check_dtype(var, dtype):
    if K._BACKEND == 'theano':
        assert var.dtype == dtype
    else:
        assert var.dtype.name == '%s_ref' % dtype


def check_single_tensor_operation(function_name, input_shape, **kwargs):
    val = np.random.random(input_shape) - 0.5
    xth = KTH.variable(val)
    xtf = KTF.variable(val)

    zth = KTH.eval(getattr(KTH, function_name)(xth, **kwargs))
    ztf = KTF.eval(getattr(KTF, function_name)(xtf, **kwargs))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


def check_two_tensor_operation(function_name, x_input_shape,
                               y_input_shape, **kwargs):
    xval = np.random.random(x_input_shape) - 0.5

    xth = KTH.variable(xval)
    xtf = KTF.variable(xval)

    yval = np.random.random(y_input_shape) - 0.5

    yth = KTH.variable(yval)
    ytf = KTF.variable(yval)

    zth = KTH.eval(getattr(KTH, function_name)(xth, yth, **kwargs))
    ztf = KTF.eval(getattr(KTF, function_name)(xtf, ytf, **kwargs))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


def check_composed_tensor_operations(first_function_name, first_function_args,
                                     second_function_name, second_function_args,
                                     input_shape):
    ''' Creates a random tensor t0 with shape input_shape and compute
                 t1 = first_function_name(t0, **first_function_args)
                 t2 = second_function_name(t1, **second_function_args)
        with both Theano and TensorFlow backends and ensures the answers match.
    '''
    val = np.random.random(input_shape) - 0.5
    xth = KTH.variable(val)
    xtf = KTF.variable(val)

    yth = getattr(KTH, first_function_name)(xth, **first_function_args)
    ytf = getattr(KTF, first_function_name)(xtf, **first_function_args)

    zth = KTH.eval(getattr(KTH, second_function_name)(yth, **second_function_args))
    ztf = KTF.eval(getattr(KTF, second_function_name)(ytf, **second_function_args))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


class TestBackend(object):

    def test_linear_operations(self):
        check_two_tensor_operation('dot', (4, 2), (2, 4))
        check_two_tensor_operation('dot', (4, 2), (5, 2, 3))

        check_two_tensor_operation('batch_dot', (4, 2, 3), (4, 5, 3),
                                   axes=(2, 2))
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20), axes=1)
        check_two_tensor_operation('batch_dot', (32, 20), (32, 20), axes=(1, 1))
        check_single_tensor_operation('transpose', (4, 2))
        check_single_tensor_operation('reverse', (4, 3, 2), axes=1)
        check_single_tensor_operation('reverse', (4, 3, 2), axes=(1, 2))

    def test_batch_dot_shape(self):
        x_batch = KTF.ones(shape=(32, 20))
        y_batch = KTF.ones(shape=(32, 20))
        xy_batch_dot = KTF.batch_dot(x_batch, y_batch, axes=1)
        assert_allclose(KTF.eval(xy_batch_dot), np.ones((32, 1)) * 20, atol=1e-05)
        xy_batch_dot = KTF.batch_dot(x_batch, y_batch, axes=0)
        assert_allclose(KTF.eval(xy_batch_dot), np.ones((20, 1)) * 32, atol=1e-05)
        # making sure swapping axes when ndim == 2 works
        x_batch = KTF.ones(shape=(32, 20))
        y_batch = KTF.ones(shape=(20, 32))
        xy_batch_dot = KTF.batch_dot(x_batch, y_batch, axes=(0, 1))
        assert_allclose(KTF.eval(xy_batch_dot), np.ones((20, 1)) * 32, atol=1e-05)
        xy_batch_dot = KTF.batch_dot(x_batch, y_batch, axes=(1, 0))
        assert_allclose(KTF.eval(xy_batch_dot), np.ones((32, 1)) * 20, atol=1e-05)

    def test_shape_operations(self):
        # concatenate
        xval = np.random.random((4, 3))
        xth = KTH.variable(xval)
        xtf = KTF.variable(xval)
        yval = np.random.random((4, 2))
        yth = KTH.variable(yval)
        ytf = KTF.variable(yval)
        zth = KTH.eval(KTH.concatenate([xth, yth], axis=-1))
        ztf = KTF.eval(KTF.concatenate([xtf, ytf], axis=-1))
        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

        check_single_tensor_operation('reshape', (4, 2), shape=(8, 1))
        check_single_tensor_operation('permute_dimensions', (4, 2, 3),
                                      pattern=(2, 0, 1))
        check_single_tensor_operation('repeat', (4, 1), n=3)
        check_single_tensor_operation('flatten', (4, 1))
        check_single_tensor_operation('expand_dims', (4, 3), dim=-1)
        check_single_tensor_operation('expand_dims', (4, 3, 2), dim=1)
        check_single_tensor_operation('squeeze', (4, 3, 1), axis=2)
        check_single_tensor_operation('squeeze', (4, 1, 1), axis=1)
        check_composed_tensor_operations('reshape', {'shape': (4, 3, 1, 1)},
                                         'squeeze', {'axis': 2},
                                         (4, 3, 1, 1))

    def test_repeat_elements(self):
        reps = 3
        for ndims in [1, 2, 3]:
            shape = np.arange(2, 2 + ndims)
            arr = np.arange(np.prod(shape)).reshape(shape)
            arr_th = KTH.variable(arr)
            arr_tf = KTF.variable(arr)

            for rep_axis in range(ndims):
                np_rep = np.repeat(arr, reps, axis=rep_axis)
                th_rep = KTH.eval(
                    KTH.repeat_elements(arr_th, reps, axis=rep_axis))
                tf_rep = KTF.eval(
                    KTF.repeat_elements(arr_tf, reps, axis=rep_axis))

                assert th_rep.shape == np_rep.shape
                assert tf_rep.shape == np_rep.shape
                assert_allclose(np_rep, th_rep, atol=1e-05)
                assert_allclose(np_rep, tf_rep, atol=1e-05)

    def test_tile(self):
        shape = (3, 4)
        arr = np.arange(np.prod(shape)).reshape(shape)
        arr_th = KTH.variable(arr)
        arr_tf = KTF.variable(arr)

        n = (2, 1)
        th_rep = KTH.eval(KTH.tile(arr_th, n))
        tf_rep = KTF.eval(KTF.tile(arr_tf, n))
        assert_allclose(tf_rep, th_rep, atol=1e-05)

    def test_value_manipulation(self):
        val = np.random.random((4, 2))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)

        # get_value
        valth = KTH.get_value(xth)
        valtf = KTF.get_value(xtf)
        assert valtf.shape == valth.shape
        assert_allclose(valth, valtf, atol=1e-05)

        # set_value
        val = np.random.random((4, 2))
        KTH.set_value(xth, val)
        KTF.set_value(xtf, val)

        valth = KTH.get_value(xth)
        valtf = KTF.get_value(xtf)
        assert valtf.shape == valth.shape
        assert_allclose(valth, valtf, atol=1e-05)

        # count_params
        assert KTH.count_params(xth) == KTF.count_params(xtf)

        # print_tensor
        check_single_tensor_operation('print_tensor', ())
        check_single_tensor_operation('print_tensor', (2,))
        check_single_tensor_operation('print_tensor', (4, 3))
        check_single_tensor_operation('print_tensor', (1, 2, 3))

        val = np.random.random((3, 2))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)
        assert KTH.get_variable_shape(xth) == KTF.get_variable_shape(xtf)

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

        # does not work yet, wait for bool <-> int casting in TF (coming soon)
        # check_single_tensor_operation('any', (4, 2))
        # check_single_tensor_operation('any', (4, 2), axis=1, keepdims=True)
        #
        # check_single_tensor_operation('any', (4, 2))
        # check_single_tensor_operation('any', (4, 2), axis=1, keepdims=True)

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
        check_single_tensor_operation('clip', (4, 2), min_value=0.4,
                                      max_value=0.6)

        # two-tensor ops
        check_two_tensor_operation('equal', (4, 2), (4, 2))
        check_two_tensor_operation('not_equal', (4, 2), (4, 2))
        check_two_tensor_operation('greater', (4, 2), (4, 2))
        check_two_tensor_operation('greater_equal', (4, 2), (4, 2))
        check_two_tensor_operation('lesser', (4, 2), (4, 2))
        check_two_tensor_operation('lesser_equal', (4, 2), (4, 2))
        check_two_tensor_operation('maximum', (4, 2), (4, 2))
        check_two_tensor_operation('minimum', (4, 2), (4, 2))

    def test_gradient(self):
        val = np.random.random((4, 2))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)

        expth = xth * KTH.exp(xth)
        exptf = xtf * KTF.exp(xtf)
        lossth = KTH.sum(expth)
        losstf = KTF.sum(exptf)
        zero_lossth = KTH.stop_gradient(lossth)
        zero_losstf = KTF.stop_gradient(losstf)

        gradth = KTH.gradients(lossth, [expth])
        gradtf = KTF.gradients(losstf, [exptf])
        zero_gradth = KTH.gradients(lossth + zero_lossth, [expth])
        zero_gradtf = KTF.gradients(losstf + zero_losstf, [exptf])

        zth = KTH.eval(gradth[0])
        ztf = KTF.eval(gradtf[0])
        zero_zth = KTH.eval(zero_gradth[0])
        zero_ztf = KTF.eval(zero_gradtf[0])
        assert zth.shape == ztf.shape
        assert zero_zth.shape == zero_ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)
        assert_allclose(zero_zth, zero_ztf, atol=1e-05)
        assert_allclose(zero_zth, zth, atol=1e-05)
        assert_allclose(zero_ztf, ztf, atol=1e-05)

    def test_function(self):
        val = np.random.random((4, 2))
        input_val = np.random.random((4, 2))

        xth = KTH.variable(val)
        xtf = KTF.variable(val)
        yth = KTH.placeholder(ndim=2)
        ytf = KTF.placeholder(ndim=2)

        exp_th = KTH.square(xth) + yth
        exp_tf = KTF.square(xtf) + ytf

        update_th = xth * 2
        update_tf = xtf * 2
        fth = KTH.function([yth], [exp_th], updates=[(xth, update_th)])
        ftf = KTF.function([ytf], [exp_tf], updates=[(xtf, update_tf)])

        function_outputs_th = fth([input_val])[0]
        function_outputs_tf = ftf([input_val])[0]
        assert function_outputs_th.shape == function_outputs_tf.shape
        assert_allclose(function_outputs_th, function_outputs_tf, atol=1e-05)

        new_val_th = KTH.get_value(xth)
        new_val_tf = KTF.get_value(xtf)
        assert new_val_th.shape == new_val_tf.shape
        assert_allclose(new_val_th, new_val_tf, atol=1e-05)

    def test_rnn(self):
        # implement a simple RNN
        input_dim = 8
        output_dim = 4
        timesteps = 5

        input_val = np.random.random((32, timesteps, input_dim))
        init_state_val = np.random.random((32, output_dim))
        W_i_val = np.random.random((input_dim, output_dim))
        W_o_val = np.random.random((output_dim, output_dim))

        def rnn_step_fn(input_dim, output_dim, K):
            W_i = K.variable(W_i_val)
            W_o = K.variable(W_o_val)

            def step_function(x, states):
                assert len(states) == 1
                prev_output = states[0]
                output = K.dot(x, W_i) + K.dot(prev_output, W_o)
                return output, [output]
            return step_function

        # test default setup
        th_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTH)
        th_inputs = KTH.variable(input_val)
        th_initial_states = [KTH.variable(init_state_val)]
        last_output, outputs, new_states = KTH.rnn(th_rnn_step_fn, th_inputs,
                                                   th_initial_states,
                                                   go_backwards=False,
                                                   mask=None)
        th_last_output = KTH.eval(last_output)
        th_outputs = KTH.eval(outputs)
        assert len(new_states) == 1
        th_state = KTH.eval(new_states[0])

        tf_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTF)
        tf_inputs = KTF.variable(input_val)
        tf_initial_states = [KTF.variable(init_state_val)]
        last_output, outputs, new_states = KTF.rnn(tf_rnn_step_fn, tf_inputs,
                                                   tf_initial_states,
                                                   go_backwards=False,
                                                   mask=None)
        tf_last_output = KTF.eval(last_output)
        tf_outputs = KTF.eval(outputs)
        assert len(new_states) == 1
        tf_state = KTF.eval(new_states[0])

        assert_allclose(tf_last_output, th_last_output, atol=1e-04)
        assert_allclose(tf_outputs, th_outputs, atol=1e-04)
        assert_allclose(tf_state, th_state, atol=1e-04)

        # test unroll
        unrolled_last_output, unrolled_outputs, unrolled_new_states = KTH.rnn(
            th_rnn_step_fn, th_inputs,
            th_initial_states,
            go_backwards=False,
            mask=None,
            unroll=True,
            input_length=timesteps)

        unrolled_th_last_output = KTH.eval(unrolled_last_output)
        unrolled_th_outputs = KTH.eval(unrolled_outputs)
        assert len(unrolled_new_states) == 1
        unrolled_th_state = KTH.eval(unrolled_new_states[0])
        assert_allclose(th_last_output, unrolled_th_last_output, atol=1e-04)
        assert_allclose(th_outputs, unrolled_th_outputs, atol=1e-04)
        assert_allclose(th_state, unrolled_th_state, atol=1e-04)

        # test go_backwards
        th_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTH)
        th_inputs = KTH.variable(input_val)
        th_initial_states = [KTH.variable(init_state_val)]
        last_output, outputs, new_states = KTH.rnn(th_rnn_step_fn, th_inputs,
                                                   th_initial_states,
                                                   go_backwards=True,
                                                   mask=None)
        th_last_output = KTH.eval(last_output)
        th_outputs = KTH.eval(outputs)
        assert len(new_states) == 1
        th_state = KTH.eval(new_states[0])

        tf_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTF)
        tf_inputs = KTF.variable(input_val)
        tf_initial_states = [KTF.variable(init_state_val)]
        last_output, outputs, new_states = KTF.rnn(tf_rnn_step_fn, tf_inputs,
                                                   tf_initial_states,
                                                   go_backwards=True,
                                                   mask=None)
        tf_last_output = KTF.eval(last_output)
        tf_outputs = KTF.eval(outputs)
        assert len(new_states) == 1
        tf_state = KTF.eval(new_states[0])

        assert_allclose(tf_last_output, th_last_output, atol=1e-04)
        assert_allclose(tf_outputs, th_outputs, atol=1e-04)
        assert_allclose(tf_state, th_state, atol=1e-04)

        # test unroll with backwards = True
        bwd_last_output, bwd_outputs, bwd_new_states = KTH.rnn(
            th_rnn_step_fn, th_inputs,
            th_initial_states,
            go_backwards=True,
            mask=None)
        bwd_th_last_output = KTH.eval(bwd_last_output)
        bwd_th_outputs = KTH.eval(bwd_outputs)
        assert len(bwd_new_states) == 1
        bwd_th_state = KTH.eval(bwd_new_states[0])

        bwd_unrolled_last_output, bwd_unrolled_outputs, bwd_unrolled_new_states = KTH.rnn(
            th_rnn_step_fn, th_inputs,
            th_initial_states,
            go_backwards=True,
            mask=None,
            unroll=True,
            input_length=timesteps)

        bwd_unrolled_th_last_output = KTH.eval(bwd_unrolled_last_output)
        bwd_unrolled_th_outputs = KTH.eval(bwd_unrolled_outputs)
        assert len(bwd_unrolled_new_states) == 1
        bwd_unrolled_th_state = KTH.eval(bwd_unrolled_new_states[0])
        assert_allclose(bwd_th_last_output, bwd_unrolled_th_last_output, atol=1e-04)
        assert_allclose(bwd_th_outputs, bwd_unrolled_th_outputs, atol=1e-04)
        assert_allclose(bwd_th_state, bwd_unrolled_th_state, atol=1e-04)

        # test unroll with masking
        np_mask = np.random.randint(2, size=(32, timesteps))
        th_mask = KTH.variable(np_mask)

        masked_last_output, masked_outputs, masked_new_states = KTH.rnn(
            th_rnn_step_fn, th_inputs,
            th_initial_states,
            go_backwards=False,
            mask=th_mask)
        masked_th_last_output = KTH.eval(masked_last_output)
        masked_th_outputs = KTH.eval(masked_outputs)
        assert len(masked_new_states) == 1
        masked_th_state = KTH.eval(masked_new_states[0])

        unrolled_masked_last_output, unrolled_masked_outputs, unrolled_masked_new_states = KTH.rnn(
            th_rnn_step_fn, th_inputs,
            th_initial_states,
            go_backwards=False,
            mask=th_mask,
            unroll=True,
            input_length=timesteps)
        unrolled_masked_th_last_output = KTH.eval(unrolled_masked_last_output)
        unrolled_masked_th_outputs = KTH.eval(unrolled_masked_outputs)
        assert len(unrolled_masked_new_states) == 1
        unrolled_masked_th_state = KTH.eval(unrolled_masked_new_states[0])

        assert_allclose(unrolled_masked_th_last_output, masked_th_last_output, atol=1e-04)
        assert_allclose(unrolled_masked_th_outputs, masked_th_outputs, atol=1e-04)
        assert_allclose(unrolled_masked_th_state, masked_th_state, atol=1e-04)

    def test_rnn_no_states(self):
        # implement a simple RNN without states
        input_dim = 8
        output_dim = 4
        timesteps = 5

        input_val = np.random.random((32, timesteps, input_dim))
        W_i_val = np.random.random((input_dim, output_dim))

        def rnn_step_fn(input_dim, output_dim, K):
            W_i = K.variable(W_i_val)

            def step_function(x, states):
                assert len(states) == 0
                output = K.dot(x, W_i)
                return output, []
            return step_function

        # test default setup
        th_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTH)
        th_inputs = KTH.variable(input_val)
        th_initial_states = []
        last_output, outputs, new_states = KTH.rnn(th_rnn_step_fn, th_inputs,
                                                   th_initial_states,
                                                   go_backwards=False,
                                                   mask=None)
        th_last_output = KTH.eval(last_output)
        th_outputs = KTH.eval(outputs)
        assert len(new_states) == 0

        tf_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTF)
        tf_inputs = KTF.variable(input_val)
        tf_initial_states = []
        last_output, outputs, new_states = KTF.rnn(tf_rnn_step_fn, tf_inputs,
                                                   tf_initial_states,
                                                   go_backwards=False,
                                                   mask=None)
        tf_last_output = KTF.eval(last_output)
        tf_outputs = KTF.eval(outputs)
        assert len(new_states) == 0

        assert_allclose(tf_last_output, th_last_output, atol=1e-04)
        assert_allclose(tf_outputs, th_outputs, atol=1e-04)

    def test_switch(self):
        val = np.random.random()
        xth = KTH.variable(val)
        xth = KTH.switch(xth >= 0.5, xth * 0.1, xth * 0.2)

        xtf = KTF.variable(val)
        xtf = KTF.switch(xtf >= 0.5, xtf * 0.1, xtf * 0.2)

        zth = KTH.eval(xth)
        ztf = KTF.eval(xtf)

        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

    def test_nn_operations(self):
        check_single_tensor_operation('relu', (4, 2), alpha=0.1, max_value=0.5)
        check_single_tensor_operation('softmax', (4, 10))
        check_single_tensor_operation('softplus', (4, 10))
        check_single_tensor_operation('elu', (4, 10), alpha=0.5)

        check_single_tensor_operation('sigmoid', (4, 2))
        check_single_tensor_operation('hard_sigmoid', (4, 2))
        check_single_tensor_operation('tanh', (4, 2))

        # dropout
        val = np.random.random((100, 100))
        xth = KTH.variable(val)
        xtf = KTF.variable(val)
        zth = KTH.eval(KTH.dropout(xth, level=0.2))
        ztf = KTF.eval(KTF.dropout(xtf, level=0.2))
        assert zth.shape == ztf.shape
        # dropout patterns are different, only check mean
        assert np.abs(zth.mean() - ztf.mean()) < 0.05

        check_two_tensor_operation('binary_crossentropy', (4, 2), (4, 2), from_logits=True)
        check_two_tensor_operation('categorical_crossentropy', (4, 2), (4, 2), from_logits=True)
        check_two_tensor_operation('binary_crossentropy', (4, 2), (4, 2), from_logits=False)
        check_two_tensor_operation('categorical_crossentropy', (4, 2), (4, 2), from_logits=False)

        check_single_tensor_operation('l2_normalize', (4, 3), axis=-1)
        check_single_tensor_operation('l2_normalize', (4, 3), axis=1)

    def test_conv2d(self):
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)

        for input_shape in [(2, 3, 4, 5), (2, 3, 5, 6)]:
            for kernel_shape in [(4, 3, 2, 2), (4, 3, 3, 4)]:
                xval = np.random.random(input_shape)

                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)

                kernel_val = np.random.random(kernel_shape) - 0.5

                kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='th'))
                kernel_tf = KTF.variable(kernel_val)

                zth = KTH.eval(KTH.conv2d(xth, kernel_th, dim_ordering='th'))
                ztf = KTF.eval(KTF.conv2d(xtf, kernel_tf, dim_ordering='th'))

                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-05)

        input_shape = (1, 6, 5, 3)
        kernel_shape = (3, 3, 3, 2)

        xval = np.random.random(input_shape)

        xth = KTH.variable(xval)
        xtf = KTF.variable(xval)

        kernel_val = np.random.random(kernel_shape) - 0.5

        kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='tf'))
        kernel_tf = KTF.variable(kernel_val)

        zth = KTH.eval(KTH.conv2d(xth, kernel_th, dim_ordering='tf'))
        ztf = KTF.eval(KTF.conv2d(xtf, kernel_tf, dim_ordering='tf'))

        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

    def test_conv3d(self):
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        # TH kernel shape: (depth, input_depth, x, y, z)
        # TF kernel shape: (x, y, z, input_depth, depth)

        # test in dim_ordering = th
        for input_shape in [(2, 3, 4, 5, 4), (2, 3, 5, 4, 6)]:
            for kernel_shape in [(4, 3, 2, 2, 2), (4, 3, 3, 2, 4)]:
                xval = np.random.random(input_shape)

                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)

                kernel_val = np.random.random(kernel_shape) - 0.5

                kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='th'))
                kernel_tf = KTF.variable(kernel_val)

                zth = KTH.eval(KTH.conv3d(xth, kernel_th, dim_ordering='th'))
                ztf = KTF.eval(KTF.conv3d(xtf, kernel_tf, dim_ordering='th'))

                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-05)

        # test in dim_ordering = tf
        input_shape = (1, 2, 2, 2, 1)
        kernel_shape = (2, 2, 2, 1, 1)

        xval = np.random.random(input_shape)

        xth = KTH.variable(xval)
        xtf = KTF.variable(xval)

        kernel_val = np.random.random(kernel_shape) - 0.5

        kernel_th = KTH.variable(convert_kernel(kernel_val, dim_ordering='tf'))
        kernel_tf = KTF.variable(kernel_val)

        zth = KTH.eval(KTH.conv3d(xth, kernel_th, dim_ordering='tf'))
        ztf = KTF.eval(KTF.conv3d(xtf, kernel_tf, dim_ordering='tf'))

        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

    def test_pool2d(self):
        check_single_tensor_operation('pool2d', (5, 10, 12, 3), pool_size=(2, 2),
                                      strides=(1, 1), border_mode='valid')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3), pool_size=(2, 2),
                                      strides=(1, 1), border_mode='valid')

        check_single_tensor_operation('pool2d', (5, 9, 11, 3), pool_size=(2, 3),
                                      strides=(1, 1), border_mode='valid')

    def test_pool3d(self):
        check_single_tensor_operation('pool3d', (5, 10, 12, 5, 3), pool_size=(2, 2, 2),
                                      strides=(1, 1, 1), border_mode='valid')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3), pool_size=(2, 2, 2),
                                      strides=(1, 1, 1), border_mode='valid')

        check_single_tensor_operation('pool3d', (5, 9, 11, 5, 3), pool_size=(2, 3, 2),
                                      strides=(1, 1, 1), border_mode='valid')

    def test_random_normal(self):
        mean = 0.
        std = 1.
        rand = KTF.eval(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert rand.shape == (1000, 1000)
        assert np.abs(np.mean(rand) - mean) < 0.01
        assert np.abs(np.std(rand) - std) < 0.01

        rand = KTH.eval(KTH.random_normal((1000, 1000), mean=mean, std=std))
        assert rand.shape == (1000, 1000)
        assert np.abs(np.mean(rand) - mean) < 0.01
        assert np.abs(np.std(rand) - std) < 0.01

    def test_random_uniform(self):
        min_val = -1.
        max_val = 1.
        rand = KTF.eval(KTF.random_uniform((1000, 1000), min_val, max_val))
        assert rand.shape == (1000, 1000)
        assert np.abs(np.mean(rand)) < 0.01
        assert np.max(rand) <= max_val
        assert np.min(rand) >= min_val

        rand = KTH.eval(KTH.random_uniform((1000, 1000), min_val, max_val))
        assert rand.shape == (1000, 1000)
        assert np.abs(np.mean(rand)) < 0.01
        assert np.max(rand) <= max_val
        assert np.min(rand) >= min_val

    def test_random_binomial(self):
        p = 0.5
        rand = KTF.eval(KTF.random_binomial((1000, 1000), p))
        assert rand.shape == (1000, 1000)
        assert np.abs(np.mean(rand) - p) < 0.01
        assert np.max(rand) == 1
        assert np.min(rand) == 0

        rand = KTH.eval(KTH.random_binomial((1000, 1000), p))
        assert rand.shape == (1000, 1000)
        assert np.abs(np.mean(rand) - p) < 0.01
        assert np.max(rand) == 1
        assert np.min(rand) == 0

    def test_ctc(self):
        # simplified version of TensorFlow's test

        label_lens = np.expand_dims(np.asarray([5, 4]), 1)
        input_lens = np.expand_dims(np.asarray([5, 5]), 1)  # number of timesteps

        # the Theano and Tensorflow CTC code use different methods to ensure
        # numerical stability.  The Theano code subtracts out the max
        # before the final log, so the results are different but scale
        # identically and still train properly
        loss_log_probs_tf = [3.34211, 5.42262]
        loss_log_probs_th = [1.73308, 3.81351]

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

        labels_tf = KTF.variable(labels, dtype="int32")
        inputs_tf = KTF.variable(inputs, dtype="float32")
        input_lens_tf = KTF.variable(input_lens, dtype="int32")
        label_lens_tf = KTF.variable(label_lens, dtype="int32")
        res = KTF.eval(KTF.ctc_batch_cost(labels_tf, inputs_tf, input_lens_tf, label_lens_tf))
        assert_allclose(res[:, 0], loss_log_probs_tf, atol=1e-05)

        labels_th = KTH.variable(labels, dtype="int32")
        inputs_th = KTH.variable(inputs, dtype="float32")
        input_lens_th = KTH.variable(input_lens, dtype="int32")
        label_lens_th = KTH.variable(label_lens, dtype="int32")
        res = KTH.eval(KTH.ctc_batch_cost(labels_th, inputs_th, input_lens_th, label_lens_th))
        assert_allclose(res[0, :], loss_log_probs_th, atol=1e-05)

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
        nb_classes = 20
        batch_size = 30
        indices = np.random.randint(0, nb_classes, size=(batch_size, input_length))
        oh = np.eye(nb_classes)[indices]
        for K in [KTH, KTF]:
            koh = K.eval(K.one_hot(K.variable(indices, dtype='int32'), nb_classes))
            assert np.all(koh == oh)

    def test_sparse_dot(self):
        x_d = np.array([0, 7, 2, 3], dtype=np.float32)
        x_r = np.array([0, 2, 2, 3], dtype=np.int64)
        x_c = np.array([4, 3, 2, 3], dtype=np.int64)

        x_sparse = sparse.csr_matrix((x_d, (x_r, x_c)), shape=(4, 5))
        x_dense = x_sparse.toarray()

        W = np.random.random((5, 4))

        backends = [KTF]
        if KTH.th_sparse_module:
            # Theano has some dependency issues for sparse
            backends.append(KTH)

        for K in backends:
            t_W = K.variable(W)
            k_s = K.eval(K.dot(K.variable(x_sparse), t_W))
            k_d = K.eval(K.dot(K.variable(x_dense), t_W))

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

        backends = [KTF]
        if KTH.th_sparse_module:
            # Theano has some dependency issues for sparse
            backends.append(KTH)

        for K in backends:
            k_s = K.concatenate([K.variable(x_sparse_1), K.variable(x_sparse_2)])
            assert K.is_sparse(k_s)

            k_s_d = K.eval(k_s)

            k_d = K.eval(K.concatenate([K.variable(x_dense_1), K.variable(x_dense_2)]))

            assert k_s_d.shape == k_d.shape
            assert_allclose(k_s_d, k_d, atol=1e-05)

    def test_map(self):
        x = np.random.rand(10, 3).astype(np.float32)
        for K in [KTF, KTH]:
            kx = K.eval(K.map_fn(K.sum, x))

            assert (10,) == kx.shape
            assert_allclose(x.sum(axis=1), kx, atol=1e-05)

    def test_foldl(self):
        x = np.random.rand(10, 3).astype(np.float32)
        for K in [KTF, KTH]:
            kx = K.eval(K.foldl(lambda a, b: a + b, x))

            assert (3,) == kx.shape
            assert_allclose(x.sum(axis=0), kx, atol=1e-05)

    def test_foldr(self):
        # This test aims to make sure that we walk the array from right to left
        # and checks it in the following way: multiplying left to right 1e-40
        # cannot be held into a float32 so it causes an underflow while from
        # right to left we have no such problem and the result is larger
        x = np.array([1e-20, 1e-20, 10, 10, 10], dtype=np.float32)
        for K in [KTF, KTH]:
            p1 = K.eval(K.foldl(lambda a, b: a * b, x))
            p2 = K.eval(K.foldr(lambda a, b: a * b, x))

            assert p1 < p2
            assert 9e-38 < p2 <= 1e-37

    def test_arange(self):
        for test_value in (-20, 0, 1, 10):
            t_a = KTF.arange(test_value)
            a = KTF.eval(t_a)
            assert np.array_equal(a, np.arange(test_value))
            t_b = KTH.arange(test_value)
            b = KTH.eval(t_b)
            assert np.array_equal(b, np.arange(test_value))
            assert np.array_equal(a, b)
            assert KTF.dtype(t_a) == KTH.dtype(t_b)
        for start, stop, step in ((0, 5, 1), (-5, 5, 2), (0, 1, 2)):
            a = KTF.eval(KTF.arange(start, stop, step))
            assert np.array_equal(a, np.arange(start, stop, step))
            b = KTH.eval(KTH.arange(start, stop, step))
            assert np.array_equal(b, np.arange(start, stop, step))
            assert np.array_equal(a, b)
        for dtype in ('int32', 'int64', 'float32', 'float64'):
            for backend in (KTF, KTH):
                t = backend.arange(10, dtype=dtype)
                assert backend.dtype(t) == dtype

    def test_setfloatx_incorrect_values(self):
        # Keep track of the old value
        old_floatx = floatx()
        # Try some incorrect values
        initial = floatx()
        for value in ['', 'beerfloat', 123]:
            with pytest.raises(Exception):
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

if __name__ == '__main__':
    pytest.main([__file__])
