'''MNIST classification with TensorFlow's Dataset API.

Introduced in TensorFlow 1.3, the Dataset API is now the
standard method for loading data into TensorFlow models.
A Dataset is a sequence of elements, which are themselves
composed of tf.Tensor components. For more details, see:
https://www.tensorflow.org/programmers_guide/datasets

To use this with Keras, we make a dataset out of elements
of the form (input batch, output batch). From there, we
create a one-shot iterator and a graph node corresponding
to its get_next() method. Its components are then provided
to the network's Input layer and the Model.compile() method,
respectively.

Note that from TensorFlow 1.4, tf.contrib.data is deprecated
and tf.data is preferred. See the release notes for details.

This example is intended to closely follow the
mnist_tfrecord.py example.
'''
import numpy as np
import os
import tempfile

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist

import tensorflow as tf
from tensorflow.contrib.data import Dataset


if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the TensorFlow backend,'
                       ' because it requires the Datset API, which is not'
                       ' supported on other platforms.')


def cnn_layers(inputs):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return predictions


batch_size = 128
buffer_size = 10000
steps_per_epoch = int(np.ceil(60000 / float(batch_size)))  # = 469
epochs = 5
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_train = np.expand_dims(x_train, -1)
y_train = tf.one_hot(y_train, num_classes)

# Create the dataset and its associated one-shot iterator.
dataset = Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size)
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()

# Model creation using tensors from the get_next() graph node.
inputs, targets = iterator.get_next()
model_input = layers.Input(tensor=inputs)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[targets])
train_model.summary()

train_model.fit(epochs=epochs,
                steps_per_epoch=steps_per_epoch)

# Save the model weights.
weight_path = os.path.join(tempfile.gettempdir(), 'saved_wt.h5')
train_model.save_weights(weight_path)

# Clean up the TF session.
K.clear_session()

# Second session to test loading trained model without tensors.
x_test = x_test.astype(np.float32)
x_test = np.expand_dims(x_test, -1)

x_test_inp = layers.Input(shape=x_test.shape[1:])
test_out = cnn_layers(x_test_inp)
test_model = keras.models.Model(inputs=x_test_inp, outputs=test_out)

test_model.load_weights(weight_path)
test_model.compile(optimizer='rmsprop',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(x_test, y_test, num_classes)
print('\nTest accuracy: {0}'.format(acc))
