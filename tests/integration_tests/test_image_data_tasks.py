from __future__ import print_function
import numpy as np
import pytest

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras import layers
from keras.utils.np_utils import to_categorical


def test_image_classification():
    np.random.seed(1337)
    input_shape = (16, 16, 3)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=500,
                                                         num_test=200,
                                                         input_shape=input_shape,
                                                         classification=True,
                                                         num_classes=4)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        layers.Conv2D(filters=8, kernel_size=3,
                      activation='relu',
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters=4, kernel_size=(3, 3),
                      activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(y_test.shape[-1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=12, batch_size=16,
                        validation_data=(x_test, y_test),
                        verbose=0)
    assert history.history['val_accuracy'][-1] > 0.75
    config = model.get_config()
    model = Sequential.from_config(config)


def test_image_data_generator_training():
    np.random.seed(1337)
    img_gen = ImageDataGenerator(rescale=1.)  # Dummy ImageDataGenerator
    input_shape = (16, 16, 3)
    (x_train, y_train), (x_test, y_test) = get_test_data(num_train=500,
                                                         num_test=200,
                                                         input_shape=input_shape,
                                                         classification=True,
                                                         num_classes=4)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        layers.Conv2D(filters=8, kernel_size=3,
                      activation='relu',
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters=4, kernel_size=(3, 3),
                      activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(y_test.shape[-1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = model.fit_generator(img_gen.flow(x_train, y_train, batch_size=16),
                                  epochs=15,
                                  validation_data=img_gen.flow(x_test, y_test,
                                                               batch_size=16),
                                  verbose=0)
    assert history.history['val_accuracy'][-1] > 0.70
    model.evaluate_generator(img_gen.flow(x_train, y_train, batch_size=16))


if __name__ == '__main__':
    pytest.main([__file__])
