import pytest
import numpy as np
from numpy.testing import assert_allclose
import keras
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
import time


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_cudnn_rnn_canonical_to_params_lstm():
    units = 1
    input_size = 1
    layer = keras.layers.CuDNNLSTM(units)
    layer.build((None, None, input_size))

    params = layer._canonical_to_params(
        weights=[
            layer.kernel_i,
            layer.kernel_f,
            layer.kernel_c,
            layer.kernel_o,
            layer.recurrent_kernel_i,
            layer.recurrent_kernel_f,
            layer.recurrent_kernel_c,
            layer.recurrent_kernel_o,
        ],
        biases=[
            layer.bias_i_i,
            layer.bias_f_i,
            layer.bias_c_i,
            layer.bias_o_i,
            layer.bias_i,
            layer.bias_f,
            layer.bias_c,
            layer.bias_o,
        ],
    )
    ref_params = layer._cudnn_lstm.canonical_to_params(
        weights=[
            layer.kernel_i,
            layer.kernel_f,
            layer.kernel_c,
            layer.kernel_o,
            layer.recurrent_kernel_i,
            layer.recurrent_kernel_f,
            layer.recurrent_kernel_c,
            layer.recurrent_kernel_o,
        ],
        biases=[
            layer.bias_i_i,
            layer.bias_f_i,
            layer.bias_c_i,
            layer.bias_o_i,
            layer.bias_i,
            layer.bias_f,
            layer.bias_c,
            layer.bias_o,
        ],
    )
    ref_params_value = keras.backend.get_value(ref_params)
    params_value = keras.backend.get_value(params)
    diff = np.mean(ref_params_value - params_value)
    assert diff < 1e-8


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_cudnn_rnn_canonical_to_params_gru():
    units = 7
    input_size = 9
    layer = keras.layers.CuDNNGRU(units)
    layer.build((None, None, input_size))

    ref_params = layer._cudnn_gru.canonical_to_params(
        weights=[
            layer.kernel_r,
            layer.kernel_z,
            layer.kernel_h,
            layer.recurrent_kernel_r,
            layer.recurrent_kernel_z,
            layer.recurrent_kernel_h,
        ],
        biases=[
            layer.bias_r_i,
            layer.bias_z_i,
            layer.bias_h_i,
            layer.bias_r,
            layer.bias_z,
            layer.bias_h,
        ],
    )
    params = layer._canonical_to_params(
        weights=[
            layer.kernel_r,
            layer.kernel_z,
            layer.kernel_h,
            layer.recurrent_kernel_r,
            layer.recurrent_kernel_z,
            layer.recurrent_kernel_h,
        ],
        biases=[
            layer.bias_r_i,
            layer.bias_z_i,
            layer.bias_h_i,
            layer.bias_r,
            layer.bias_z,
            layer.bias_h,
        ],
    )
    ref_params_value = keras.backend.get_value(ref_params)
    params_value = keras.backend.get_value(params)
    diff = np.mean(ref_params_value - params_value)
    assert diff < 1e-8


@keras_test
@pytest.mark.parametrize('rnn_type', ['lstm', 'gru'], ids=['LSTM', 'GRU'])
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_cudnn_rnn_timing(rnn_type):
    input_size = 1000
    timesteps = 60
    units = 256
    num_samples = 10000

    times = []
    for use_cudnn in [True, False]:
        start_time = time.time()
        inputs = keras.layers.Input(shape=(None, input_size))
        if use_cudnn:
            if rnn_type == 'lstm':
                layer = keras.layers.CuDNNLSTM(units)
            else:
                layer = keras.layers.CuDNNGRU(units)
        else:
            if rnn_type == 'lstm':
                layer = keras.layers.LSTM(units)
            else:
                layer = keras.layers.GRU(units)
        outputs = layer(inputs)

        model = keras.models.Model(inputs, outputs)
        model.compile('sgd', 'mse')

        x = np.random.random((num_samples, timesteps, input_size))
        y = np.random.random((num_samples, units))
        model.fit(x, y, epochs=4, batch_size=32)

        times.append(time.time() - start_time)

    speedup = times[1] / times[0]
    print(rnn_type, 'speedup', speedup)
    assert speedup > 3


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_cudnn_rnn_basics():
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
        for return_sequences in [True, False]:
            with keras.utils.CustomObjectScope(
                    {'keras.layers.CuDNNGRU': keras.layers.CuDNNGRU,
                     'keras.layers.CuDNNLSTM': keras.layers.CuDNNLSTM}):
                layer_test(
                    layer_class,
                    kwargs={'units': units,
                            'return_sequences': return_sequences},
                    input_shape=(num_samples, timesteps, input_size))
        for go_backwards in [True, False]:
            with keras.utils.CustomObjectScope(
                    {'keras.layers.CuDNNGRU': keras.layers.CuDNNGRU,
                     'keras.layers.CuDNNLSTM': keras.layers.CuDNNLSTM}):
                layer_test(
                    layer_class,
                    kwargs={'units': units,
                            'go_backwards': go_backwards},
                    input_shape=(num_samples, timesteps, input_size))


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_trainability():
    input_size = 10
    units = 2
    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
        layer = layer_class(units)
        layer.build((None, None, input_size))
        assert len(layer.weights) == 3
        assert len(layer.trainable_weights) == 3
        assert len(layer.non_trainable_weights) == 0
        layer.trainable = False
        assert len(layer.weights) == 3
        assert len(layer.non_trainable_weights) == 3
        assert len(layer.trainable_weights) == 0
        layer.trainable = True
        assert len(layer.weights) == 3
        assert len(layer.trainable_weights) == 3
        assert len(layer.non_trainable_weights) == 0


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_regularizer():
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
        layer = layer_class(units, return_sequences=False,
                            input_shape=(timesteps, input_size),
                            kernel_regularizer=keras.regularizers.l1(0.01),
                            recurrent_regularizer=keras.regularizers.l1(0.01),
                            bias_regularizer='l2')
        layer.build((None, None, input_size))
        assert len(layer.losses) == 3

        layer = layer_class(units, return_sequences=False,
                            input_shape=(timesteps, input_size),
                            activity_regularizer='l2')
        assert layer.activity_regularizer
        x = keras.backend.variable(np.ones((num_samples,
                                            timesteps,
                                            input_size)))
        layer(x)
        assert len(layer.get_losses_for(x)) == 1


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_return_state():
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32

    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
        num_states = 2 if layer_class is keras.layers.CuDNNLSTM else 1

        inputs = keras.Input(batch_shape=(num_samples, timesteps, input_size))
        layer = layer_class(units, return_state=True, stateful=True)
        outputs = layer(inputs)
        output, state = outputs[0], outputs[1:]
        assert len(state) == num_states
        model = keras.models.Model(inputs, state[0])

        inputs = np.random.random((num_samples, timesteps, input_size))
        state = model.predict(inputs)
        np.testing.assert_allclose(
            keras.backend.eval(layer.states[0]), state, atol=1e-4)


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_specify_initial_state_keras_tensor():
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
        num_states = 2 if layer_class is keras.layers.CuDNNLSTM else 1

        inputs = keras.Input((timesteps, input_size))
        initial_state = [keras.Input((units,)) for _ in range(num_states)]
        layer = layer_class(units)
        if len(initial_state) == 1:
            output = layer(inputs, initial_state=initial_state[0])
        else:
            output = layer(inputs, initial_state=initial_state)
        assert initial_state[0] in layer._inbound_nodes[0].input_tensors

        model = keras.models.Model([inputs] + initial_state, output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        inputs = np.random.random((num_samples, timesteps, input_size))
        initial_state = [np.random.random((num_samples, units))
                         for _ in range(num_states)]
        targets = np.random.random((num_samples, units))
        model.fit([inputs] + initial_state, targets)


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_statefulness():
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32

    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(10, input_size,
                                         input_length=timesteps,
                                         batch_input_shape=(num_samples,
                                                            timesteps)))
        layer = layer_class(units,
                            return_sequences=False,
                            stateful=True,
                            weights=None)
        model.add(layer)
        model.compile(optimizer='sgd', loss='mse')
        out1 = model.predict(np.ones((num_samples, timesteps)))
        assert(out1.shape == (num_samples, units))

        # train once so that the states change
        model.train_on_batch(np.ones((num_samples, timesteps)),
                             np.ones((num_samples, units)))
        out2 = model.predict(np.ones((num_samples, timesteps)))

        # if the state is not reset, output should be different
        assert(out1.max() != out2.max())

        # check that output changes after states are reset
        # (even though the model itself didn't change)
        layer.reset_states()
        out3 = model.predict(np.ones((num_samples, timesteps)))
        assert(out2.max() != out3.max())

        # check that container-level reset_states() works
        model.reset_states()
        out4 = model.predict(np.ones((num_samples, timesteps)))
        assert_allclose(out3, out4, atol=1e-5)

        # check that the call to `predict` updated the states
        out5 = model.predict(np.ones((num_samples, timesteps)))
        assert(out4.max() != out5.max())


@keras_test
@pytest.mark.parametrize('implementation', [1, 2], ids=['impl1', 'impl2'])
@pytest.mark.parametrize('bidirectional', [False, True], ids=['single', 'bidirectional'])
@pytest.mark.parametrize('to_cudnn', [False, True], ids=['from_cudnn', 'to_cudnn'])
@pytest.mark.parametrize('rnn_type', ['LSTM', 'GRU'], ids=['LSTM', 'GRU'])
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_load_weights_between_noncudnn_rnn(rnn_type, to_cudnn, bidirectional, implementation):
    input_size = 10
    timesteps = 6
    input_shape = (timesteps, input_size)
    units = 2
    num_samples = 32
    inputs = np.random.random((num_samples, timesteps, input_size))

    rnn_layer_kwargs = {
        'recurrent_activation': 'sigmoid',
        # ensure biases are non-zero and properly converted
        'bias_initializer': 'random_uniform',
        'implementation': implementation
    }
    if rnn_type == 'LSTM':
        rnn_layer_class = keras.layers.LSTM
        cudnn_rnn_layer_class = keras.layers.CuDNNLSTM
    else:
        rnn_layer_class = keras.layers.GRU
        cudnn_rnn_layer_class = keras.layers.CuDNNGRU
        rnn_layer_kwargs['reset_after'] = True

    def convert_weights(source_layer, target_layer):
        weights = source_layer.get_weights()
        weights = keras.engine.topology.preprocess_weights_for_loading(target_layer, weights)
        target_layer.set_weights(weights)

    input_layer = keras.layers.InputLayer(input_shape)

    layer = rnn_layer_class(units, **rnn_layer_kwargs)
    if bidirectional:
        layer = keras.layers.Bidirectional(layer)

    cudnn_layer = cudnn_rnn_layer_class(units)
    if bidirectional:
        cudnn_layer = keras.layers.Bidirectional(cudnn_layer)

    model = keras.models.Sequential([input_layer, layer])
    cudnn_model = keras.models.Sequential([input_layer, cudnn_layer])

    if to_cudnn:
        convert_weights(layer, cudnn_layer)
    else:
        convert_weights(cudnn_layer, layer)

    assert_allclose(model.predict(inputs), cudnn_model.predict(inputs), atol=1e-4)


@keras_test
@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_cudnnrnn_bidirectional():
    rnn = keras.layers.CuDNNGRU
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    mode = 'concat'

    x = np.random.random((samples, timesteps, dim))
    target_dim = 2 * output_dim if mode == 'concat' else output_dim
    y = np.random.random((samples, target_dim))

    # test with Sequential model
    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(rnn(output_dim),
                                         merge_mode=mode,
                                         input_shape=(None, dim)))
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, y, epochs=1, batch_size=1)

    # test config
    model.get_config()
    model = keras.models.model_from_json(model.to_json())
    model.summary()

    # test stacked bidirectional layers
    model = keras.Sequential()
    model.add(keras.layers.Bidirectional(rnn(output_dim,
                                             return_sequences=True),
                                         merge_mode=mode,
                                         input_shape=(None, dim)))
    model.add(keras.layers.Bidirectional(rnn(output_dim), merge_mode=mode))
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, y, epochs=1, batch_size=1)

    # test with functional API
    inputs = keras.Input((timesteps, dim))
    outputs = keras.layers.Bidirectional(rnn(output_dim),
                                         merge_mode=mode)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, y, epochs=1, batch_size=1)

    # Bidirectional and stateful
    inputs = keras.Input(batch_shape=(1, timesteps, dim))
    outputs = keras.layers.Bidirectional(rnn(output_dim, stateful=True),
                                         merge_mode=mode)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, y, epochs=1, batch_size=1)


@pytest.mark.skipif((keras.backend.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
@pytest.mark.skipif(not keras.backend.tensorflow_backend._get_available_gpus(),
                    reason='Requires GPU')
def test_preprocess_weights_for_loading_gru_incompatible():
    """
    Loading weights between incompatible layers should fail fast with an exception.
    """
    def gru(cudnn=False, **kwargs):
        layer_class = keras.layers.CuDNNGRU if cudnn else keras.layers.GRU
        return layer_class(2, input_shape=[3, 5], **kwargs)

    def initialize_weights(layer):
        # A model is needed to initialize weights.
        _ = keras.models.Sequential([layer])
        return layer

    def assert_not_compatible(src, dest, message):
        with pytest.raises(ValueError) as ex:
            keras.engine.topology.preprocess_weights_for_loading(
                dest, initialize_weights(src).get_weights())
        assert message in ex.value.message

    assert_not_compatible(gru(), gru(cudnn=True),
                          'GRU(reset_after=False) is not compatible with CuDNNGRU')
    assert_not_compatible(gru(cudnn=True), gru(),
                          'CuDNNGRU is not compatible with GRU(reset_after=False)')
    assert_not_compatible(gru(), gru(reset_after=True),
                          'GRU(reset_after=False) is not compatible with GRU(reset_after=True)')
    assert_not_compatible(gru(reset_after=True), gru(),
                          'GRU(reset_after=True) is not compatible with GRU(reset_after=False)')


if __name__ == '__main__':
    pytest.main([__file__])
