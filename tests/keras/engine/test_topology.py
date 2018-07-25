import pytest
import json
import numpy as np

from keras.layers import Dense, Dropout, Conv2D, InputLayer
from keras import layers
from keras.engine import Input, Layer, saving, get_source_inputs
from keras.models import Model, Sequential
from keras import backend as K
from keras.models import model_from_json, model_from_yaml
from keras.utils.test_utils import keras_test
from keras.initializers import Constant


skipif_no_tf_gpu = pytest.mark.skipif(
    (K.backend() != 'tensorflow') or (not K.tensorflow_backend._get_available_gpus()),
    reason='Requires TensorFlow backend and a GPU')


@keras_test
def test_get_updates_for():
    a = Input(shape=(2,))
    dense_layer = Dense(1)
    dense_layer.add_update(0, inputs=a)
    dense_layer.add_update(1, inputs=None)

    assert dense_layer.get_updates_for(a) == [0]
    assert dense_layer.get_updates_for(None) == [1]


@keras_test
def test_get_losses_for():
    a = Input(shape=(2,))
    dense_layer = Dense(1)
    dense_layer.add_loss(0, inputs=a)
    dense_layer.add_loss(1, inputs=None)

    assert dense_layer.get_losses_for(a) == [0]
    assert dense_layer.get_losses_for(None) == [1]


@keras_test
def test_trainable_weights():
    a = Input(shape=(2,))
    b = Dense(1)(a)
    model = Model(a, b)

    weights = model.weights
    assert model.trainable_weights == weights
    assert model.non_trainable_weights == []

    model.trainable = False
    assert model.trainable_weights == []
    assert model.non_trainable_weights == weights

    model.trainable = True
    assert model.trainable_weights == weights
    assert model.non_trainable_weights == []

    model.layers[1].trainable = False
    assert model.trainable_weights == []
    assert model.non_trainable_weights == weights

    # sequential model
    model = Sequential()
    model.add(Dense(1, input_dim=2))
    weights = model.weights

    assert model.trainable_weights == weights
    assert model.non_trainable_weights == []

    model.trainable = False
    assert model.trainable_weights == []
    assert model.non_trainable_weights == weights

    model.trainable = True
    assert model.trainable_weights == weights
    assert model.non_trainable_weights == []

    model.layers[0].trainable = False
    assert model.trainable_weights == []
    assert model.non_trainable_weights == weights


def test_valid_compute_mask():
    model = Sequential()
    model.add(Dense(1, input_dim=2))
    assert model.layers[0].supports_masking is True
    assert model.layers[0].compute_mask([model.input], [0., 1.]) == [0., 1.]


def test_invalid_compute_mask():
    model = Sequential()
    model.add(Conv2D(1, [2, 2], input_shape=[3, 3, 1]))
    assert model.layers[0].supports_masking is False
    assert model.layers[0].compute_mask([model.input], [None]) is None

    mask = np.array([[0., 1.], [1., 0.]])
    with pytest.raises(TypeError):
        model.layers[0].compute_mask([model.input], [mask])
    with pytest.raises(TypeError):
        model.layers[0].compute_mask([model.input], mask)


def test_get_layer():
    model = Sequential()
    model.add(Dense(1, input_dim=2))
    with pytest.raises(ValueError):
        model.get_layer(index=5)
    with pytest.raises(ValueError):
        model.get_layer(index=None)
    with pytest.raises(ValueError):
        model.get_layer(name='conv')


def test_learning_phase():
    a = Input(shape=(32,), name='input_a')
    b = Input(shape=(32,), name='input_b')

    a_2 = Dense(16, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    assert not a_2._uses_learning_phase
    assert b_2._uses_learning_phase

    # test merge
    m = layers.concatenate([a_2, b_2])
    assert m._uses_learning_phase

    # Test recursion
    model = Model([a, b], [a_2, b_2])
    print(model.input_spec)
    assert model.uses_learning_phase

    c = Input(shape=(32,), name='input_c')
    d = Input(shape=(32,), name='input_d')

    c_2, b_2 = model([c, d])
    assert c_2._uses_learning_phase
    assert b_2._uses_learning_phase

    # try actually running graph
    fn = K.function(model.inputs + [K.learning_phase()], model.outputs)
    input_a_np = np.random.random((10, 32))
    input_b_np = np.random.random((10, 32))
    fn_outputs_no_dp = fn([input_a_np, input_b_np, 0])
    fn_outputs_dp = fn([input_a_np, input_b_np, 1])
    # output a: nothing changes
    assert fn_outputs_no_dp[0].sum() == fn_outputs_dp[0].sum()
    # output b: dropout applied
    assert fn_outputs_no_dp[1].sum() != fn_outputs_dp[1].sum()


@keras_test
def test_layer_call_arguments():
    # Test the ability to pass and serialize arguments to `call`.
    inp = layers.Input(shape=(2,))
    x = layers.Dense(3)(inp)
    x = layers.Dropout(0.5)(x, training=True)
    model = Model(inp, x)
    assert not model.uses_learning_phase

    # Test that argument is kept when applying the model
    inp2 = layers.Input(shape=(2,))
    out2 = model(inp2)
    assert not out2._uses_learning_phase

    # Test that argument is kept after loading a model
    config = model.get_config()
    model = Model.from_config(config)
    assert not model.uses_learning_phase


@keras_test
def test_node_construction():
    ####################################################
    # test basics

    a = Input(shape=(32,), name='input_a')
    b = Input(shape=(32,), name='input_b')

    assert a._keras_shape == (None, 32)
    a_layer, a_node_index, a_tensor_index = a._keras_history
    b_layer, b_node_index, b_tensor_index = b._keras_history
    assert len(a_layer._inbound_nodes) == 1
    assert a_tensor_index is 0
    node = a_layer._inbound_nodes[a_node_index]
    assert node.outbound_layer == a_layer

    assert isinstance(node.inbound_layers, list)
    assert node.inbound_layers == []
    assert isinstance(node.input_tensors, list)
    assert node.input_tensors == [a]
    assert isinstance(node.input_masks, list)
    assert node.input_masks == [None]
    assert isinstance(node.input_shapes, list)
    assert node.input_shapes == [(None, 32)]

    assert isinstance(node.output_tensors, list)
    assert node.output_tensors == [a]
    assert isinstance(node.output_shapes, list)
    assert node.output_shapes == [(None, 32)]
    assert isinstance(node.output_masks, list)
    assert node.output_masks == [None]

    dense = Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    assert len(dense._inbound_nodes) == 2
    assert len(dense._outbound_nodes) == 0
    assert dense._inbound_nodes[0].inbound_layers == [a_layer]
    assert dense._inbound_nodes[0].outbound_layer == dense
    assert dense._inbound_nodes[1].inbound_layers == [b_layer]
    assert dense._inbound_nodes[1].outbound_layer == dense

    assert dense._inbound_nodes[0].input_tensors == [a]
    assert dense._inbound_nodes[1].input_tensors == [b]

    assert dense._inbound_nodes[0].get_config()['inbound_layers'] == ['input_a']
    assert dense._inbound_nodes[1].get_config()['inbound_layers'] == ['input_b']

    # test layer properties
    test_layer = Dense(16, name='test_layer')
    a_test = test_layer(a)
    assert K.int_shape(test_layer.kernel) == (32, 16)
    assert test_layer.input == a
    assert test_layer.output == a_test
    assert test_layer.input_mask is None
    assert test_layer.output_mask is None
    assert test_layer.input_shape == (None, 32)
    assert test_layer.output_shape == (None, 16)

    with pytest.raises(AttributeError):
        dense.input
    with pytest.raises(AttributeError):
        dense.output
    with pytest.raises(AttributeError):
        dense.input_mask
    with pytest.raises(AttributeError):
        dense.output_mask

    assert dense.get_input_at(0) == a
    assert dense.get_input_at(1) == b
    assert dense.get_output_at(0) == a_2
    assert dense.get_output_at(1) == b_2
    assert dense.get_input_shape_at(0) == (None, 32)
    assert dense.get_input_shape_at(1) == (None, 32)
    assert dense.get_output_shape_at(0) == (None, 16)
    assert dense.get_output_shape_at(1) == (None, 16)
    assert dense.get_input_mask_at(0) is None
    assert dense.get_input_mask_at(1) is None
    assert dense.get_output_mask_at(0) is None
    assert dense.get_output_mask_at(1) is None


@keras_test
def test_multi_input_layer():
    ####################################################
    # test multi-input layer
    a = Input(shape=(32,), name='input_a')
    b = Input(shape=(32,), name='input_b')

    dense = Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)

    merged = layers.concatenate([a_2, b_2], name='merge')
    assert merged._keras_shape == (None, 16 * 2)
    merge_layer, merge_node_index, merge_tensor_index = merged._keras_history

    assert merge_node_index == 0
    assert merge_tensor_index == 0

    assert len(merge_layer._inbound_nodes) == 1
    assert len(merge_layer._outbound_nodes) == 0

    assert len(merge_layer._inbound_nodes[0].input_tensors) == 2
    assert len(merge_layer._inbound_nodes[0].inbound_layers) == 2

    c = Dense(64, name='dense_2')(merged)
    d = Dense(5, name='dense_3')(c)

    model = Model(inputs=[a, b], outputs=[c, d], name='model')
    assert len(model.layers) == 6
    assert model.compute_output_shape([(None, 32), (None, 32)]) == [(None, 64), (None, 5)]
    assert model.compute_mask([a, b], [None, None]) == [None, None]
    assert model.compute_output_shape([(None, 32), (None, 32)]) == [(None, 64), (None, 5)]

    # we don't check names of first 2 layers (inputs) because
    # ordering of same-level layers is not fixed
    assert [l.name for l in model.layers][2:] == ['dense_1', 'merge', 'dense_2', 'dense_3']
    assert [l.name for l in model._input_layers] == ['input_a', 'input_b']
    assert [l.name for l in model._output_layers] == ['dense_2', 'dense_3']

    # actually run model
    fn = K.function(model.inputs, model.outputs)
    input_a_np = np.random.random((10, 32))
    input_b_np = np.random.random((10, 32))
    fn_outputs = fn([input_a_np, input_b_np])
    assert [x.shape for x in fn_outputs] == [(10, 64), (10, 5)]

    # test get_source_inputs
    assert get_source_inputs(c) == [a, b]

    # serialization / deserialization
    json_config = model.to_json()
    recreated_model = model_from_json(json_config)
    recreated_model.compile('rmsprop', 'mse')

    assert [l.name for l in recreated_model.layers][2:] == ['dense_1', 'merge', 'dense_2', 'dense_3']
    assert [l.name for l in recreated_model._input_layers] == ['input_a', 'input_b']
    assert [l.name for l in recreated_model._output_layers] == ['dense_2', 'dense_3']

    fn = K.function(recreated_model.inputs, recreated_model.outputs)
    input_a_np = np.random.random((10, 32))
    input_b_np = np.random.random((10, 32))
    fn_outputs = fn([input_a_np, input_b_np])
    assert [x.shape for x in fn_outputs] == [(10, 64), (10, 5)]


@keras_test
def test_recursion():
    ####################################################
    # test recursion

    a = Input(shape=(32,), name='input_a')
    b = Input(shape=(32,), name='input_b')

    dense = Dense(16, name='dense_1')
    a_2 = dense(a)
    b_2 = dense(b)
    merged = layers.concatenate([a_2, b_2], name='merge')
    c = Dense(64, name='dense_2')(merged)
    d = Dense(5, name='dense_3')(c)

    model = Model(inputs=[a, b], outputs=[c, d], name='model')

    e = Input(shape=(32,), name='input_e')
    f = Input(shape=(32,), name='input_f')
    g, h = model([e, f])

    # g2, h2 = model([e, f])

    assert g._keras_shape == c._keras_shape
    assert h._keras_shape == d._keras_shape

    # test separate manipulation of different layer outputs
    i = Dense(7, name='dense_4')(h)

    final_model = Model(inputs=[e, f], outputs=[i, g], name='final')
    assert len(final_model.inputs) == 2
    assert len(final_model.outputs) == 2
    assert len(final_model.layers) == 4

    # we don't check names of first 2 layers (inputs) because
    # ordering of same-level layers is not fixed
    assert [layer.name for layer in final_model.layers][2:] == ['model', 'dense_4']
    assert model.compute_mask([e, f], [None, None]) == [None, None]
    assert final_model.compute_output_shape([(10, 32), (10, 32)]) == [(10, 7), (10, 64)]

    # run recursive model
    fn = K.function(final_model.inputs, final_model.outputs)
    input_a_np = np.random.random((10, 32))
    input_b_np = np.random.random((10, 32))
    fn_outputs = fn([input_a_np, input_b_np])
    assert [x.shape for x in fn_outputs] == [(10, 7), (10, 64)]

    # test serialization
    model_config = final_model.get_config()
    print(json.dumps(model_config, indent=4))
    recreated_model = Model.from_config(model_config)

    fn = K.function(recreated_model.inputs, recreated_model.outputs)
    input_a_np = np.random.random((10, 32))
    input_b_np = np.random.random((10, 32))
    fn_outputs = fn([input_a_np, input_b_np])
    assert [x.shape for x in fn_outputs] == [(10, 7), (10, 64)]

    ####################################################
    # test multi-input multi-output

    j = Input(shape=(32,), name='input_j')
    k = Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    o = Input(shape=(32,), name='input_o')
    p = Input(shape=(32,), name='input_p')
    q, r = model([o, p])

    assert n._keras_shape == (None, 5)
    assert q._keras_shape == (None, 64)
    s = layers.concatenate([n, q], name='merge_nq')
    assert s._keras_shape == (None, 64 + 5)

    # test with single output as 1-elem list
    multi_io_model = Model([j, k, o, p], [s])

    fn = K.function(multi_io_model.inputs, multi_io_model.outputs)
    fn_outputs = fn([np.random.random((10, 32)), np.random.random((10, 32)),
                     np.random.random((10, 32)), np.random.random((10, 32))])
    assert [x.shape for x in fn_outputs] == [(10, 69)]

    # test with single output as tensor
    multi_io_model = Model([j, k, o, p], s)

    fn = K.function(multi_io_model.inputs, multi_io_model.outputs)
    fn_outputs = fn([np.random.random((10, 32)), np.random.random((10, 32)),
                     np.random.random((10, 32)), np.random.random((10, 32))])
    # note that the output of the K.function will still be a 1-elem list
    assert [x.shape for x in fn_outputs] == [(10, 69)]

    # test serialization
    model_config = multi_io_model.get_config()
    recreated_model = Model.from_config(model_config)

    fn = K.function(recreated_model.inputs, recreated_model.outputs)
    fn_outputs = fn([np.random.random((10, 32)), np.random.random((10, 32)),
                     np.random.random((10, 32)), np.random.random((10, 32))])
    # note that the output of the K.function will still be a 1-elem list
    assert [x.shape for x in fn_outputs] == [(10, 69)]

    config = model.get_config()
    Model.from_config(config)

    model.summary()
    json_str = model.to_json()
    model_from_json(json_str)

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str)

    ####################################################
    # test invalid graphs

    # input is not an Input tensor
    j = Input(shape=(32,), name='input_j')
    j = Dense(32)(j)
    k = Input(shape=(32,), name='input_k')
    m, n = model([j, k])

    with pytest.raises(ValueError):
        Model([j, k], [m, n])

    # disconnected graph
    j = Input(shape=(32,), name='input_j')
    k = Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with pytest.raises(ValueError):
        Model([j], [m, n])

    # redundant outputs
    j = Input(shape=(32,), name='input_j')
    k = Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    # this should work with a warning
    Model([j, k], [m, n, n])

    # redundant inputs
    j = Input(shape=(32,), name='input_j')
    k = Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with pytest.raises(ValueError):
        Model([j, k, j], [m, n])

    # i have not idea what I'm doing: garbage as inputs/outputs
    j = Input(shape=(32,), name='input_j')
    k = Input(shape=(32,), name='input_k')
    m, n = model([j, k])
    with pytest.raises(ValueError):
        Model([j, k], [m, n, 0])

    ####################################################
    # test calling layers/models on TF tensors

    if K._BACKEND == 'tensorflow':
        import tensorflow as tf
        j = Input(shape=(32,), name='input_j')
        k = Input(shape=(32,), name='input_k')
        m, n = model([j, k])
        tf_model = Model([j, k], [m, n])

        j_tf = tf.placeholder(dtype=K.floatx())
        k_tf = tf.placeholder(dtype=K.floatx())
        m_tf, n_tf = tf_model([j_tf, k_tf])
        assert m_tf.get_shape().as_list() == [None, 64]
        assert n_tf.get_shape().as_list() == [None, 5]

        # test merge
        layers.concatenate([j_tf, k_tf], axis=1)
        layers.add([j_tf, k_tf])

        # test tensor input
        x = tf.placeholder(shape=(None, 2), dtype=K.floatx())
        InputLayer(input_tensor=x)

        x = Input(tensor=x)
        Dense(2)(x)


@keras_test
def test_load_layers():
    from keras.layers import ConvLSTM2D, TimeDistributed, Bidirectional, Conv2D, Input
    from keras.models import Model

    if K.backend() == 'tensorflow' or K.backend() == 'cntk':
        inputs = Input(shape=(10, 20, 20, 1))
    else:
        inputs = Input(shape=(10, 1, 20, 20))
    td_conv = TimeDistributed(Conv2D(15, (5, 5)))(inputs)
    bi_convlstm2d = Bidirectional(ConvLSTM2D(10, (3, 3)), merge_mode='concat')(td_conv)
    model = Model(inputs=inputs, outputs=bi_convlstm2d)

    weight_value_tuples = []

    # TimeDistributed Conv2D layer
    # use 'channels_first' data format to check that the function is being called correctly for Conv2D
    # old: (filters, stack_size, kernel_rows, kernel_cols)
    # new: (kernel_rows, kernel_cols, stack_size, filters)
    weight_tensor_td_conv_old = list()
    weight_tensor_td_conv_old.append(np.zeros((15, 1, 5, 5)))
    weight_tensor_td_conv_old.append(np.zeros((15,)))
    td_conv_layer = model.layers[1]
    td_conv_layer.layer.data_format = 'channels_first'
    weight_tensor_td_conv_new = saving.preprocess_weights_for_loading(
        td_conv_layer,
        weight_tensor_td_conv_old,
        original_keras_version='1')
    symbolic_weights = td_conv_layer.weights
    assert (len(symbolic_weights) == len(weight_tensor_td_conv_new))
    weight_value_tuples += zip(symbolic_weights, weight_tensor_td_conv_new)

    # Bidirectional ConvLSTM2D layer
    # old ConvLSTM2D took a list of 12 weight tensors, returns a list of 3 concatenated larger tensors.
    weight_tensor_bi_convlstm_old = []
    for j in range(2):  # bidirectional
        for i in range(4):
            weight_tensor_bi_convlstm_old.append(np.zeros((3, 3, 15, 10)))  # kernel
            weight_tensor_bi_convlstm_old.append(np.zeros((3, 3, 10, 10)))  # recurrent kernel
            weight_tensor_bi_convlstm_old.append(np.zeros((10,)))  # bias

    bi_convlstm_layer = model.layers[2]
    weight_tensor_bi_convlstm_new = saving.preprocess_weights_for_loading(
        bi_convlstm_layer,
        weight_tensor_bi_convlstm_old,
        original_keras_version='1')

    symbolic_weights = bi_convlstm_layer.weights
    assert (len(symbolic_weights) == len(weight_tensor_bi_convlstm_new))
    weight_value_tuples += zip(symbolic_weights, weight_tensor_bi_convlstm_new)

    K.batch_set_value(weight_value_tuples)

    assert np.all(K.eval(model.layers[1].weights[0]) == weight_tensor_td_conv_new[0])
    assert np.all(K.eval(model.layers[1].weights[1]) == weight_tensor_td_conv_new[1])
    assert np.all(K.eval(model.layers[2].weights[0]) == weight_tensor_bi_convlstm_new[0])
    assert np.all(K.eval(model.layers[2].weights[1]) == weight_tensor_bi_convlstm_new[1])
    assert np.all(K.eval(model.layers[2].weights[2]) == weight_tensor_bi_convlstm_new[2])
    assert np.all(K.eval(model.layers[2].weights[3]) == weight_tensor_bi_convlstm_new[3])
    assert np.all(K.eval(model.layers[2].weights[4]) == weight_tensor_bi_convlstm_new[4])
    assert np.all(K.eval(model.layers[2].weights[5]) == weight_tensor_bi_convlstm_new[5])


def convert_weights(layer, weights):
    if layer.__class__.__name__ == 'GRU':
        W = [np.split(w, 3, axis=-1) for w in weights]
        return sum(map(list, zip(*W)), [])
    elif layer.__class__.__name__ in ('LSTM', 'ConvLSTM2D'):
        W = [np.split(w, 4, axis=-1) for w in weights]
        for w in W:
            w[2], w[1] = w[1], w[2]
        return sum(map(list, zip(*W)), [])
    elif layer.__class__.__name__ == 'Conv2DTranspose':
        return [np.transpose(weights[0], (2, 3, 0, 1)), weights[1]]
    return weights


@keras_test
@pytest.mark.parametrize("layer", [
    layers.GRU(2, input_shape=[3, 5]),
    layers.LSTM(2, input_shape=[3, 5]),
    layers.ConvLSTM2D(5, (3, 3),
                      input_shape=[6, 6, 6, 6],
                      data_format='channels_first'),
], ids=['GRU', 'LSTM', 'ConvLSTM2D'])
def test_preprocess_weights_for_loading(layer):
    # A model is needed to initialize weights.
    _ = Sequential([layer])
    weights1 = layer.get_weights()
    weights2 = saving.preprocess_weights_for_loading(
        layer, convert_weights(layer, weights1),
        original_keras_version='1')
    assert all([np.allclose(x, y, 1e-5)
                for (x, y) in zip(weights1, weights2)])


@keras_test
@pytest.mark.parametrize("layer", [
    layers.Conv2D(2, (3, 3), input_shape=[5, 5, 3]),
    layers.Conv2DTranspose(2, (5, 5),
                           input_shape=[7, 7, 3],
                           data_format='channels_first'),
], ids=['Conv2D', 'Conv2DTranspose'])
def test_preprocess_weights_for_loading_for_model(layer):
    model = Sequential([layer])
    weights1 = model.get_weights()
    weights2 = saving.preprocess_weights_for_loading(
        model, convert_weights(layer, weights1),
        original_keras_version='1')
    assert all([np.allclose(x, y, 1e-5)
                for (x, y) in zip(weights1, weights2)])


@keras_test
@pytest.mark.parametrize('layer_class,layer_args', [
    (layers.GRU, {'units': 2, 'input_shape': [3, 5]}),
    (layers.GRU, {'units': 2, 'input_shape': [3, 5], 'reset_after': True}),
    (layers.LSTM, {'units': 2, 'input_shape': [3, 5]}),
])
def test_preprocess_weights_for_loading_rnn_should_be_idempotent(layer_class, layer_args):
    """
    Loading weights from a RNN class to itself should not convert the weights.
    """
    # layer can be instantiated only for supported backends
    layer = layer_class(**layer_args)
    # A model is needed to initialize weights.
    _ = Sequential([layer])
    weights1 = layer.get_weights()
    weights2 = saving.preprocess_weights_for_loading(layer, weights1)
    assert all([np.allclose(x, y, 1e-5) for (x, y) in zip(weights1, weights2)])


@keras_test
@pytest.mark.parametrize('layer_class,layer_args', [
    (layers.CuDNNGRU, {'units': 2, 'input_shape': [3, 5]}),
    (layers.CuDNNLSTM, {'units': 2, 'input_shape': [3, 5]}),
])
@skipif_no_tf_gpu
def test_preprocess_weights_for_loading_cudnn_rnn_should_be_idempotent(layer_class, layer_args):
    test_preprocess_weights_for_loading_rnn_should_be_idempotent(layer_class, layer_args)


@keras_test
def test_recursion_with_bn_and_loss():
    model1 = Sequential([
        layers.Dense(5, input_dim=5, activity_regularizer='l1'),
        layers.BatchNormalization(),
        layers.Dense(5),
    ])

    print('NEW MODEL')
    inputs = layers.Input(shape=(5,))
    outputs = model1(inputs)
    model2 = Model(inputs=inputs, outputs=outputs)

    assert len(model1.updates) == 2
    assert len(model2.updates) == 2
    assert len(model1.losses) == 1
    assert len(model2.losses) == 1, model2.layers[1]._per_input_losses

    model1.compile(optimizer='sgd', loss='categorical_crossentropy')
    model2.compile(optimizer='sgd', loss='categorical_crossentropy')

    x = np.ones((3, 5))
    y = np.ones((3, 5))
    model1.fit(x, y, verbose=0, epochs=1)
    model2.fit(x, y, verbose=0, epochs=1)


@keras_test
def test_activity_regularization_with_model_composition():

    def reg(x):
        return K.sum(x)

    net_a_input = Input((2,))
    net_a = net_a_input
    net_a = Dense(2, kernel_initializer='ones',
                  use_bias=False,
                  activity_regularizer=reg)(net_a)
    model_a = Model([net_a_input], [net_a])

    net_b_input = Input((2,))
    net_b = model_a(net_b_input)
    model_b = Model([net_b_input], [net_b])

    model_b.compile(optimizer='sgd', loss=None)
    x = np.ones((1, 2))
    loss = model_b.evaluate(x)
    assert loss == 4


@keras_test
def test_shared_layer_depth_is_correct():
    # Basic outline here: we have a shared embedding layer, and two inputs that go through
    # different depths of computation in the graph before the final output.  We need the computed
    # depth of the input layers to be the same, because they both pass through the embedding layer
    # before anything else happens.  That's what we're testing.
    from keras.layers import Embedding, Input, Dense, Concatenate
    from keras.models import Model
    input1 = Input(shape=(10,), name='input1')
    input2 = Input(shape=(10,), name='input2')
    embedding_layer = Embedding(name='embedding', input_dim=5, output_dim=10)
    embedded_input1 = embedding_layer(input1)
    embedded_input2 = embedding_layer(input2)
    transformed_input2 = Dense(6)(Dense(5)(Dense(3)(embedded_input2)))
    final_output = Dense(2)(Concatenate()([embedded_input1, transformed_input2]))
    model = Model(inputs=[input1, input2], outputs=final_output)
    input1_depth = -1
    input2_depth = -1
    for depth, layers in model._layers_by_depth.items():
        for layer in layers:
            if layer.name == 'input1':
                input1_depth = depth
            if layer.name == 'input2':
                input2_depth = depth
    assert input1_depth != -1
    assert input1_depth == input2_depth


@keras_test
def test_layer_sharing_at_heterogeneous_depth():
    x_val = np.random.random((10, 5))

    x = Input(shape=(5,))
    A = Dense(5, name='A')
    B = Dense(5, name='B')
    output = A(B(A(B(x))))
    M = Model(x, output)

    output_val = M.predict(x_val)

    config = M.get_config()
    weights = M.get_weights()

    M2 = Model.from_config(config)
    M2.set_weights(weights)

    output_val_2 = M2.predict(x_val)
    np.testing.assert_allclose(output_val, output_val_2, atol=1e-6)


@keras_test
def test_layer_sharing_at_heterogeneous_depth_with_concat():
    input_shape = (16, 9, 3)
    input_layer = Input(shape=input_shape)

    A = Dense(3, name='dense_A')
    B = Dense(3, name='dense_B')
    C = Dense(3, name='dense_C')

    x1 = B(A(input_layer))
    x2 = A(C(input_layer))
    output = layers.concatenate([x1, x2])

    M = Model(inputs=input_layer, outputs=output)

    x_val = np.random.random((10, 16, 9, 3))
    output_val = M.predict(x_val)

    config = M.get_config()
    weights = M.get_weights()

    M2 = Model.from_config(config)
    M2.set_weights(weights)

    output_val_2 = M2.predict(x_val)
    np.testing.assert_allclose(output_val, output_val_2, atol=1e-6)


@keras_test
def test_multi_output_mask():
    """Fixes #7589"""
    class ArbitraryMultiOutputLayer(Layer):
        def __init__(self, **kwargs):
            super(ArbitraryMultiOutputLayer, self).__init__(**kwargs)

        def call(self, inputs, **kwargs):
            return [K.abs(inputs), K.abs(inputs)]

        def compute_output_shape(self, input_shape):
            out_shape = super(ArbitraryMultiOutputLayer, self).compute_output_shape(input_shape)
            return [out_shape, out_shape]

    class ArbitraryMultiInputLayer(Layer):
        def __init__(self, **kwargs):
            super(ArbitraryMultiInputLayer, self).__init__(**kwargs)

        def call(self, inputs, **kwargs):
            negative, positive = inputs
            return negative + positive

    input_layer = Input(shape=(16, 16, 3))
    x, y = ArbitraryMultiOutputLayer()(input_layer)
    z = ArbitraryMultiInputLayer()([x, y])
    _ = Model(inputs=input_layer, outputs=z)
    assert K.int_shape(z)[1:] == (16, 16, 3)


@keras_test
def test_constant_initializer_with_numpy():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,), kernel_initializer=Constant(np.ones((3, 2)))))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    json_str = model.to_json()
    model_from_json(json_str).summary()

    yaml_str = model.to_yaml()
    model_from_yaml(yaml_str).summary()


if __name__ == '__main__':
    pytest.main([__file__])
