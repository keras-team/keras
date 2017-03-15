import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import models
from keras import losses
from keras.layers import Activation, BatchNormalization, Dense, Input
from keras.utils.test_utils import layer_test, keras_test
from keras.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


@keras_test
def basic_batchnorm_test():
    from keras import regularizers
    layer_test(normalization.BatchNormalization,
               kwargs={'momentum': 0.9,
                       'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(normalization.BatchNormalization,
               kwargs={'gamma_initializer': 'ones',
                       'beta_initializer': 'ones',
                       'moving_mean_initializer': 'zeros',
                       'moving_variance_initializer': 'ones'},
               input_shape=(3, 4, 2))
    layer_test(normalization.BatchNormalization,
               kwargs={'scale': False, 'center': False},
               input_shape=(3, 3))


@keras_test
def test_batchnorm_correctness():
    model = Sequential()
    norm = normalization.BatchNormalization(input_shape=(10,), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= K.eval(norm.beta)
    out /= K.eval(norm.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_batchnorm_mode_twice():
    # This is a regression test for issue #4881 with the old
    # batch normalization functions in the Theano backend.
    model = Sequential()
    model.add(normalization.BatchNormalization(input_shape=(10, 5, 5), axis=1))
    model.add(normalization.BatchNormalization(input_shape=(10, 5, 5), axis=1))
    model.compile(loss='mse', optimizer='sgd')

    x = np.random.normal(loc=5.0, scale=10.0, size=(20, 10, 5, 5))
    model.fit(x, x, epochs=1, verbose=0)
    model.predict(x)


@keras_test
def test_batchnorm_convnet():
    model = Sequential()
    norm = normalization.BatchNormalization(axis=1, input_shape=(3, 4, 4), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(K.eval(norm.beta), (1, 3, 1, 1))
    out /= np.reshape(K.eval(norm.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras_test
def test_shared_batchnorm():
    '''Test that a BN layer can be shared
    across different data streams.
    '''
    # Test single layer reuse
    bn = normalization.BatchNormalization(input_shape=(10,))
    x1 = Input(shape=(10,))
    bn(x1)

    x2 = Input(shape=(10,))
    y2 = bn(x2)

    x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
    model = Model(x2, y2)
    assert len(model.updates) == 2
    model.compile('sgd', 'mse')
    model.train_on_batch(x, x)

    # Test model-level reuse
    x3 = Input(shape=(10,))
    y3 = model(x3)
    new_model = Model(x3, y3)
    assert len(model.updates) == 2
    new_model.compile('sgd', 'mse')
    new_model.train_on_batch(x, x)


def get_gan_models(generator_batch_normalization=False, discriminator_batch_normalization=False):
    """
    Get the built and compiled generator, discriminator and combined models of a dummy GAN.

    :param generator_batch_normalization: If True, a `BatchNormalization` layer is added to the generator model.
    :param discriminator_batch_normalization: If True, a `BatchNormalization` layer is added to the discriminator model.
    :return: The built and compiled (generator_model, discriminator_model, combined_model).
    """
    input_dim = 10

    def get_generator_discriminator_model(x, batch_normalization=False):
        """
        Use `keras's` functional API to build a dummy generator or discriminator.

        :param x: Model's input tensor.
        :param batch_normalization: If True, a `BatchNormalization` layer is added to the model.
        :return: Model's output tensor.
        """
        x = Dense(16)(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        return Dense(1, activation='sigmoid')(x)

    generator_input = Input(shape=(input_dim,))
    generator_output = get_generator_discriminator_model(generator_input,
                                                         batch_normalization=generator_batch_normalization)
    generator_model = models.Model(inputs=[generator_input], outputs=[generator_output])

    discriminator_input = Input(shape=(1,))
    discriminator_output = get_generator_discriminator_model(discriminator_input,
                                                             batch_normalization=discriminator_batch_normalization)
    discriminator_model = models.Model(inputs=[discriminator_input], outputs=[discriminator_output])

    combined_output = discriminator_model(generator_model(generator_input))
    combined_model = models.Model(inputs=[generator_input], outputs=[combined_output])

    generator_model.compile('sgd', loss='mse')
    discriminator_model.compile('sgd', loss='mse')
    discriminator_model.trainable = False
    combined_model.compile('sgd', loss='mse')

    return generator_model, discriminator_model, combined_model


@keras_test
def test_batchnorm_generator():
    batch_size = 32
    input_dim = 10

    # this test uses batch normalization in the generator model but not the discriminator
    generator_model, discriminator_model, combined_model = get_gan_models(generator_batch_normalization=True)

    # there is some randomness in this test so do it a few times to be sure
    for _ in range(10):
        x = np.random.uniform(low=0.0, high=1.0, size=(batch_size, input_dim))
        y = np.ones(shape=batch_size)

        combined_preds = combined_model.predict_on_batch(x)
        # make sure `combined_preds` is the same shape as `y` for the objective function
        combined_preds = np.reshape(combined_preds, newshape=batch_size)

        loss_validate = K.eval(losses.mse(y, combined_preds))
        loss = combined_model.train_on_batch(x, y)

        assert '{0:.4f}'.format(loss_validate) == '{0:.4f}'.format(loss)


@keras_test
def test_batchnorm_discriminator():
    batch_size = 32

    # this test uses batch normalization in the discriminator model but not the generator
    generator_model, discriminator_model, combined_model = get_gan_models(discriminator_batch_normalization=True)

    # verify that we are able to train the discriminator model without an exception being raised
    discriminator_model.train_on_batch(np.random.uniform(low=0.0, high=1.0, size=(batch_size, )),
                                       np.ones(shape=batch_size))


if __name__ == '__main__':
    pytest.main([__file__])
