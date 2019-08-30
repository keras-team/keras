import pytest
import keras
import numpy as np
from keras import layers
from keras import backend as K


def test_sublayer_tracking():
    # basic case
    class MyLayer(layers.Layer):

        def __init__(self):
            super(MyLayer, self).__init__()
            self._input_shape = (2, 4)
            self.dense = layers.Dense(3)
            self.bidir = layers.Bidirectional(keras.layers.LSTM(2))

        def call(self, inputs):
            return self.dense(self.bidir(inputs))

    layer = MyLayer()
    assert len(layer._layers) == 2
    layer(K.constant(np.random.random((2,) + layer._input_shape)))
    assert len(layer.weights) == 2 + 3 + 3
    assert len(layer._layers[0].weights) == 2
    assert len(layer._layers[1].weights) == 6

    # recursive case
    class MyRecursiveLayer(layers.Layer):

        def __init__(self):
            super(MyRecursiveLayer, self).__init__()
            self._input_shape = (2, 4)
            self.my_layer = MyLayer()
            self.dense = layers.Dense(3)
            self.bidir = layers.Bidirectional(
                keras.layers.LSTM(2, return_sequences=True))

        def call(self, inputs):
            return self.my_layer(self.dense(self.bidir(inputs)))

    layer = MyRecursiveLayer()
    assert len(layer._layers) == 3
    layer(K.constant(np.random.random((2,) + layer._input_shape)))
    assert len(layer.weights) == 16

    # subnetwork case
    class MyLayerWithSubnetwork(keras.layers.Layer):

        def __init__(self):
            super(MyLayerWithSubnetwork, self).__init__()
            self._input_shape = (2,)
            self.dense = layers.Dense(3)
            self.sequential = keras.Sequential(
                [layers.Dense(5), layers.Dense(1)], name='seq')
            inputs = keras.Input((1,))
            outputs = layers.Dense(1)(inputs)
            self.functional = keras.Model(inputs, outputs, name='func')

        def call(self, inputs):
            x = self.dense(inputs)
            x = self.sequential(x)
            return self.functional(x)

    layer = MyLayerWithSubnetwork()
    assert len(layer._layers) == 3
    layer(K.constant(np.random.random((2,) + layer._input_shape)))
    assert len(layer.weights) == 2 + (2 + 2) + 2
    assert len(layer._layers[0].weights) == 2
    assert len(layer._layers[1].weights) == 4
    assert len(layer._layers[2].weights) == 2


def test_weight_tracking():

    class MyLayer(layers.Layer):

        def __init__(self):
            super(MyLayer, self).__init__()
            self._input_shape = (2,)
            self.dense = layers.Dense(3)
            self.w1 = K.variable(0, name='w1')

        def build(self, input_shape):
            self.w2 = K.variable(1, name='w2')
            self.w3 = self.add_weight(
                'w3', shape=(), trainable=False, initializer='zeros')

        def call(self, inputs):
            return self.dense(inputs) + self.w1 + self.w2

    layer = MyLayer()
    layer(K.constant(np.random.random((2,) + layer._input_shape)))
    assert len(layer.weights) == 5
    assert len(layer.trainable_weights) == 4
    assert len(layer.non_trainable_weights) == 1
    assert len(layer._trainable_weights) == 2
    assert layer._trainable_weights[0] is layer.w1
    assert layer._trainable_weights[1] is layer.w2
    assert len(layer._non_trainable_weights) == 1
    assert layer._non_trainable_weights[0] is layer.w3


def test_loss_tracking():
    # basic case
    class MyLayer(layers.Layer):

        def __init__(self):
            super(MyLayer, self).__init__()
            self.dense = layers.Dense(
                3, kernel_regularizer='l2', activity_regularizer='l2')

        def call(self, inputs):
            return self.dense(inputs)

    inputs = keras.Input((2,))
    outputs = MyLayer()(inputs)
    model = keras.Model(inputs, outputs)

    assert len(model.layers) == 2  # includes input layer
    assert len(model.weights) == 2
    assert len(model.losses) == 2
    assert len(model.get_losses_for(None)) == 1
    assert len(model.get_losses_for(inputs)) == 1


@pytest.mark.skipif(K.backend() != 'tensorflow',
                    reason='Requires TF symbols')
def test_tf_keras_guide():
    import tensorflow as tf

    class Linear(layers.Layer):

        def __init__(self, units=32, input_dim=32):
            super(Linear, self).__init__()
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                      dtype='float32'),
                                 trainable=True)
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                      dtype='float32'),
                                 trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    x = tf.ones((2, 2))
    linear_layer = Linear(4, 2)
    y = linear_layer(x)

    assert len(linear_layer.trainable_weights) == 2

    class Linear(layers.Layer):

        def __init__(self, units=32):
            super(Linear, self).__init__()
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer='random_normal',
                                     trainable=True)
            self.b = self.add_weight(shape=(self.units,),
                                     initializer='random_normal',
                                     trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    class MLPBlock(layers.Layer):

        def __init__(self):
            super(MLPBlock, self).__init__()
            self.linear_1 = Linear(32)
            self.linear_2 = Linear(32)
            self.linear_3 = Linear(1)

        def call(self, inputs):
            x = self.linear_1(inputs)
            x = tf.nn.relu(x)
            x = self.linear_2(x)
            x = tf.nn.relu(x)
            return self.linear_3(x)

    mlp = MLPBlock()
    y = mlp(tf.ones(shape=(3, 64)))
    assert len(mlp.weights) == 6
    assert len(mlp.trainable_weights) == 6

    class OuterLayer(layers.Layer):

        def __init__(self):
            super(OuterLayer, self).__init__()
            self.dense = layers.Dense(
                32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

        def call(self, inputs):
            return self.dense(inputs)

    layer = OuterLayer()
    _ = layer(tf.zeros((1, 1)))
    assert len(layer.losses) == 1
