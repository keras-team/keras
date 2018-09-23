from __future__ import print_function

import os
import tempfile
import pytest
import keras
from keras import layers
from keras.utils.test_utils import get_test_data


@pytest.mark.skipif(keras.backend.backend() != 'tensorflow',
                    reason='Requires TF backend')
def test_tf_optimizer():
    import tensorflow as tf

    num_hidden = 10
    output_dim = 2
    input_dim = 10
    target = 0.8
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate=1., rho=0.95, epsilon=1e-08)

    (x_train, y_train), (x_test, y_test) = get_test_data(
        num_train=1000, num_test=200,
        input_shape=(input_dim,),
        classification=True, num_classes=output_dim)

    model = keras.Sequential()
    model.add(layers.Dense(num_hidden,
                           activation='relu',
                           input_shape=(input_dim,)))
    model.add(layers.Dense(output_dim, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=8, batch_size=16,
                        validation_data=(x_test, y_test), verbose=2)
    assert history.history['val_acc'][-1] >= target

    # Test saving.
    _, fname = tempfile.mkstemp('.h5')
    model.save(fname)
    model = keras.models.load_model(fname)
    assert len(model.weights) == 4
    os.remove(fname)


if __name__ == '__main__':
    pytest.main([__file__])
