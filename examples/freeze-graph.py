'''Train a CNN and freeze parameters for tensorflow (works only for tensorflow backend)

For demo reason, just train for 10 epochs.

Use `python freeze-graph.py -t` to train the model.
Use `python freeze-graph.py -r airplane.png` to evaluate the image with trained model.
(loaded into tensorflow graph)
'''
import numpy as np
import keras
import keras.backend as K
import keras.datasets.cifar10 as cifar10
import keras.layers as kl
import keras.losses as kls
import keras.models as km
import keras.optimizers as ko
import keras.utils as ku
import tensorflow as tf
from PIL import Image

import argparse
import math
import os
import sys

sess = tf.Session()
K.set_session(sess)

cifar10classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

parser = argparse.ArgumentParser(description='TF Freeze Weights Demo')
parser.add_argument('-t', '--train', action='store_true', required=False, help='train the network')
parser.add_argument('-r', '--run', nargs=1, type=str, required=False, help='run prediction on an image')
args = parser.parse_args()

if len(sys.argv) <= 1:
    parser.print_help()
    sys.exit(0)

# Train data from the beginning
epochs = 10
batch_size = 128
categories = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = ku.to_categorical(y_train, num_classes=categories)
y_test  = ku.to_categorical(y_test,  num_classes=categories)

inputs = kl.Input(shape=(32, 32, 3))

hidden = kl.Conv2D( 64, (3, 3), padding='same')(inputs)
hidden = kl.Activation('relu')(hidden)
hidden = kl.MaxPooling2D(2, strides=(2, 2))(hidden)
hidden = kl.BatchNormalization()(hidden)
hidden = kl.Dropout(rate=0.2)(hidden)

hidden = kl.Conv2D(128, (3, 3), padding='same')(hidden)
hidden = kl.Activation('relu')(hidden)
hidden = kl.MaxPooling2D(2, strides=(2, 2))(hidden)
hidden = kl.BatchNormalization()(hidden)
hidden = kl.Dropout(rate=0.2)(hidden)

hidden = kl.Conv2D(256, (3, 3), padding='same')(hidden)
hidden = kl.Activation('relu')(hidden)
hidden = kl.MaxPooling2D(2, strides=(2, 2))(hidden)
hidden = kl.BatchNormalization()(hidden)
hidden = kl.Dropout(rate=0.2)(hidden)

hidden = kl.Flatten()(hidden)
hidden = kl.Dense(1024, activation='relu')(hidden)
output = kl.Dense(categories, activation='softmax')(hidden)

M = km.Model(inputs=inputs, outputs=output)
M.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


if args.train:
    # Write a custom callback to save tensorflow checkpoint every epoch
    class TFCheckpointCallback(keras.callbacks.Callback):
        def __init__(self, saver, sess):
            self.saver = saver
            self.sess = sess

        def on_epoch_end(self, epoch, logs=None):
            self.saver.save(self.sess, 'freeze/checkpoint', global_step=epoch)


    tf_graph = sess.graph
    # Ref: https://www.tensorflow.org/api_docs/python/tf/train/Saver
    tf_saver = tf.train.Saver()
    tfckptcb = TFCheckpointCallback(tf_saver, sess)

    # Write the protobuf graph, ref: https://www.tensorflow.org/api_docs/python/tf/train/write_graph
    tf.train.write_graph(tf_graph.as_graph_def(), 'freeze', 'graph.pbtxt', as_text=True)
    tf.train.write_graph(tf_graph.as_graph_def(), 'freeze', 'graph.pb',    as_text=False)

    # Train it!
    M.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=[tfckptcb],
          validation_data=(X_test, y_test))
elif args.run:
    filepath = args.run[0]
    if not os.path.exists(filepath):
        print('Image path specified is invalid: {}'.format(filepath))
        sys.exit(1)

    image = Image.open(filepath).convert('RGB')
    image = image.resize((32, 32), Image.ANTIALIAS)
    image_arr = np.array(image).reshape((1, 32, 32, 3))

    tf_sess = K.get_session()
    tf_graph = tf_sess.graph
    tf_saver = tf.train.Saver()
    tf_saver.restore(tf_sess, tf.train.latest_checkpoint('freeze'))

    pred = M.predict(image_arr)
    pred_txt = sorted(zip(cifar10classes, pred[0].tolist()), key=lambda pair: pair[1], reverse=True)
    for pair in pred_txt[:4]:
        print('{:10s}: {:.3f}%'.format(pair[0], pair[1] * 100))
