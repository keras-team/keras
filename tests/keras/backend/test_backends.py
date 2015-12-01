import sys
import unittest
from numpy.testing import assert_allclose
import numpy as np
import pytest

if sys.version_info.major == 2:
    from keras.backend import theano_backend as KTH
    from keras.backend import tensorflow_backend as KTF


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


@pytest.mark.skipif(sys.version_info.major != 2, reason="Requires Python 2.7")
class TestBackend(unittest.TestCase):

    def test_linear_operations(self):
        check_two_tensor_operation('dot', (4, 2), (2, 4))
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

        check_single_tensor_operation('mean', (4, 2))
        check_single_tensor_operation('mean', (4, 2), axis=1, keepdims=True)
        check_single_tensor_operation('mean', (4, 2, 3), axis=-1, keepdims=True)

        check_single_tensor_operation('std', (4, 2))
        check_single_tensor_operation('std', (4, 2), axis=1, keepdims=True)

        check_single_tensor_operation('prod', (4, 2))
        check_single_tensor_operation('prod', (4, 2), axis=1, keepdims=True)

        # does not work yet, wait for bool <-> int casting in TF (coming soon)
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
        inputs = KTH.variable(input_val)
        initial_states = [KTH.variable(init_state_val)]
        last_output, outputs, new_states = KTH.rnn(th_rnn_step_fn, inputs,
                                                   initial_states,
                                                   go_backwards=False,
                                                   masking=False)
        th_last_output = KTH.eval(last_output)
        th_outputs = KTH.eval(outputs)
        assert len(new_states) == 1
        th_state = KTH.eval(new_states[0])

        tf_rnn_step_fn = rnn_step_fn(input_dim, output_dim, KTF)
        inputs = KTF.variable(input_val)
        initial_states = [KTF.variable(init_state_val)]
        last_output, outputs, new_states = KTF.rnn(tf_rnn_step_fn, inputs,
                                                   initial_states,
                                                   go_backwards=False,
                                                   masking=False)
        tf_last_output = KTF.eval(last_output)
        tf_outputs = KTF.eval(outputs)
        assert len(new_states) == 1
        tf_state = KTF.eval(new_states[0])

        assert_allclose(tf_last_output, th_last_output, atol=1e-05)
        assert_allclose(tf_outputs, th_outputs, atol=1e-05)
        assert_allclose(tf_state, th_state, atol=1e-05)

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
        val = np.random.random((20, 20))
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

    # def test_conv2d(self):
    #     '''conv2d works "properly" with Theano and TF but outputs different
    #     values in each case. Cause unclear (input / kernel shape format?)
    #     '''
    #     # TH kernel shape: (depth, input_depth, rows, cols)
    #     check_two_tensor_operation('conv2d', (5, 3, 10, 12), (4, 3, 2, 2),
    #                                strides=(1, 1), border_mode='valid')
    #     check_two_tensor_operation('conv2d', (5, 3, 10, 12), (4, 3, 2, 2),
    #                                strides=(1, 1), border_mode='same')

    #     # TF kernel shape: (rows, cols, input_depth, depth)
    #     check_two_tensor_operation('conv2d', (5, 10, 12, 3), (2, 2, 3, 4),
    #                                strides=(1, 1), border_mode='valid', dim_ordering='tf')
    #     check_two_tensor_operation('conv2d', (5, 10, 12, 3), (2, 2, 3, 4),
    #                                strides=(1, 1), border_mode='same', dim_ordering='tf')

    #     check_two_tensor_operation('conv2d', (5, 3, 10, 12), (4, 3, 3, 3),
    #                                strides=(1, 1), border_mode='valid')
    #     check_two_tensor_operation('conv2d', (5, 3, 10, 12), (4, 3, 3, 3),
    #                                strides=(1, 1), border_mode='same')

    #     check_two_tensor_operation('conv2d', (5, 3, 10, 12), (4, 3, 3, 3),
    #                                strides=(2, 2), border_mode='valid')

    # def test_maxpool2d(self):
    #     '''maxpool2d works "properly" with Theano and TF but outputs different
    #     values in each case. Cause unclear (input shape format?)
    #     '''
    #     check_single_tensor_operation('maxpool2d', (5, 3, 10, 12), pool_size=(2, 2),
    #                                   strides=(1, 1), border_mode='valid')

    #     check_single_tensor_operation('maxpool2d', (5, 3, 9, 11), pool_size=(2, 2),
    #                                   strides=(1, 1), border_mode='valid')

    #     check_single_tensor_operation('maxpool2d', (5, 3, 9, 11), pool_size=(2, 3),
    #                                   strides=(1, 1), border_mode='valid')

    def test_random_normal(self):
        mean = 0.
        std = 1.
        rand = KTF.get_value(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

        rand = KTF.get_value(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

    def test_random_uniform(self):
        mean = 0.
        std = 1.
        rand = KTF.get_value(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)

        rand = KTF.get_value(KTF.random_normal((1000, 1000), mean=mean, std=std))
        assert(rand.shape == (1000, 1000))
        assert(np.abs(np.mean(rand) - mean) < 0.01)
        assert(np.abs(np.std(rand) - std) < 0.01)


if __name__ == '__main__':
    unittest.main()
