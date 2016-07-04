import sys
import pytest
from numpy.testing import assert_allclose
import numpy as np

from keras.backend import theano_backend as KTH
from keras.backend import tensorflow_backend as KTF
from keras.utils.np_utils import convert_kernel


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
        check_single_tensor_operation('transpose', (4, 2))

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
        check_composed_tensor_operations('reshape', {'shape':(4,3,1,1)}, 
                                         'squeeze', {'axis':2}, 
                                         (4, 3, 1, 1))

    def test_repeat_elements(self):
        reps = 3
        for ndims in [1, 2, 3]:
            shape = np.arange(2, 2+ndims)
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

        gradth = KTH.gradients(lossth, [expth])
        gradtf = KTF.gradients(losstf, [exptf])

        zth = KTH.eval(gradth[0])
        ztf = KTF.eval(gradtf[0])
        assert zth.shape == ztf.shape
        assert_allclose(zth, ztf, atol=1e-05)

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

                kernel_th = KTH.variable(convert_kernel(kernel_val))
                kernel_tf = KTF.variable(kernel_val)

                zth = KTH.eval(KTH.conv2d(xth, kernel_th))
                ztf = KTF.eval(KTF.conv2d(xtf, kernel_tf))

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

                kernel_th = KTH.variable(convert_kernel(kernel_val))
                kernel_tf = KTF.variable(kernel_val)

                zth = KTH.eval(KTH.conv3d(xth, kernel_th))
                ztf = KTF.eval(KTF.conv3d(xtf, kernel_tf))

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
        check_single_tensor_operation('pool2d', (5, 3, 10, 12), pool_size=(2, 2),
                                      strides=(1, 1), border_mode='valid')

        check_single_tensor_operation('pool2d', (5, 3, 9, 11), pool_size=(2, 2),
                                      strides=(1, 1), border_mode='valid')

        check_single_tensor_operation('pool2d', (5, 3, 9, 11), pool_size=(2, 3),
                                      strides=(1, 1), border_mode='valid')

    def test_pool3d(self):
        check_single_tensor_operation('pool3d', (5, 3, 10, 12, 5), pool_size=(2, 2, 2),
                                      strides=(1, 1, 1), border_mode='valid')

        check_single_tensor_operation('pool3d', (5, 3, 9, 11, 5), pool_size=(2, 2, 2),
                                      strides=(1, 1, 1), border_mode='valid')

        check_single_tensor_operation('pool3d', (5, 3, 9, 11, 5), pool_size=(2, 3, 2),
                                      strides=(1, 1, 1), border_mode='valid')

    def test_random_normal(self):
        mean = 0.
        std = 1.
        rand = KTF.eval(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

        rand = KTH.eval(KTH.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

    def test_random_uniform(self):
        min = -1.
        max = 1.
        rand = KTF.eval(KTF.random_uniform((1000, 1000), min, max))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand)) < 0.01)
        assert(np.max(rand) <= max)
        assert(np.min(rand) >= min)

        rand = KTH.eval(KTH.random_uniform((1000, 1000), min, max))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand)) < 0.01)
        assert(np.max(rand) <= max)
        assert(np.min(rand) >= min)

    def test_random_binomial(self):
        p = 0.5
        rand = KTF.eval(KTF.random_binomial((1000, 1000), p))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - p) < 0.01)
        assert(np.max(rand) == 1)
        assert(np.min(rand) == 0)

        rand = KTH.eval(KTH.random_binomial((1000, 1000), p))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - p) < 0.01)
        assert(np.max(rand) == 1)
        assert(np.min(rand) == 0)


if __name__ == '__main__':
    pytest.main([__file__])
