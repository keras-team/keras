'''
This is the Keras implementation of
'Domain-Adversarial Training of Neural Networks' by Y. Ganin

This allows domain adaptation (when you want to train on a dataset
with different statistics than a target dataset) in an unsupervised manner
by using the adversarial paradigm to punish features that help discriminate
between the datasets during backpropagation.

This is achieved by usage of the 'gradient reversal' layer to form
a domain invariant embedding for classification by an MLP.

The example here uses the 'MNIST-M' dataset as described in the paper.

Credits:
- Clayton Mellina (https://github.com/pumpikano/tf-dann) for providing
  a sketch of implementation (in TF) and utility functions.
- Yusuke Iwasawa (https://github.com/fchollet/keras/issues/3119#issuecomment-230289301)
  for Theano implementation (op) for gradient reversal.

Author: Vanush Vaswani (vanush@gmail.com)
'''

from __future__ import print_function
from keras.layers import Input, Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.datasets import mnist
import keras.backend as K

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.manifold import TSNE

from keras.layers import GradientReversal
from keras.engine.training import make_batches
from keras.datasets import mnist_m


# Helper functions

def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        # The AxesGrid object work as a list of axes.
        grid[i].imshow(np.swapaxes(np.swapaxes(images[i], 0, 2), 0, 1))


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def batch_gen(batches, id_array, data, labels):
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = id_array[batch_start:batch_end]
        if labels is not None:
            yield data[batch_ids], labels[batch_ids]
        else:
            yield data[batch_ids]
        np.random.shuffle(id_array)


def evaluate_dann(num_batches, size):
    acc = 0
    for i in range(0, num_batches):
        _, prob = dann_model.predict_on_batch(XT_test[i * size:i * size + size])
        predictions = np.argmax(prob, axis=1)
        actual = np.argmax(y_test[i * size:i * size + size], axis=1)
        acc += float(np.sum((predictions == actual))) / size
    return acc / num_batches


# Model parameters

batch_size = 128
nb_epoch = 15
nb_classes = 10
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 5

_TRAIN = K.variable(1, dtype='uint8')

# Prep source data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Prep target data
mnistm = mnist_m.load_data()
XT_test = np.swapaxes(np.swapaxes(mnistm['test'], 1, 3), 2, 3)
XT_train = np.swapaxes(np.swapaxes(mnistm['train'], 1, 3), 2, 3)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = np.concatenate([X_train, X_train, X_train], axis=1)
X_test = np.concatenate([X_test, X_test, X_test], axis=1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

XT_train = XT_train.astype('float32')
XT_test = XT_test.astype('float32')
XT_train /= 255
XT_test /= 255

domain_labels = np.vstack([np.tile([0, 1], [batch_size / 2, 1]),
                           np.tile([1., 0.], [batch_size / 2, 1])])

# Created mixed dataset for TSNE visualization
num_test = 500
combined_test_imgs = np.vstack([X_test[:num_test], XT_test[:num_test]])
combined_test_labels = np.vstack([y_test[:num_test], y_test[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                 np.tile([0., 1.], [num_test, 1])])


class DANNBuilder(object):
    def __init__(self):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        self.opt = SGD()

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            activation='relu')(model_input)
        net = Convolution2D(nb_filters, nb_conv, nb_conv,
                            activation='relu')(net)
        net = MaxPooling2D(pool_size=(nb_pool, nb_pool))(net)
        net = Dropout(0.5)(net)
        net = Flatten()(net)
        self.domain_invariant_features = net
        return net

    def _build_classifier(self, model_input):
        net = Dense(128, activation='relu')(model_input)
        net = Dropout(0.5)(net)
        net = Dense(nb_classes, activation='softmax',
                    name='classifier_output')(net)
        return net

    def build_source_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        net = self._build_classifier(net)
        model = Model(input=main_input, output=net)
        if plot_model:
            plot(model, show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_dann_model(self, main_input, plot_model=False):
        net = self._build_feature_extractor(main_input)
        self.grl = GradientReversal(1.0)
        branch = self.grl(net)
        branch = Dense(128, activation='relu')(branch)
        branch = Dropout(0.1)(branch)
        branch = Dense(2, activation='softmax', name='domain_output')(branch)

        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        net = Lambda(lambda x: K.switch(K.learning_phase(),
                     x[:int(batch_size / 2), :], x, lazy=True),
                     output_shape=lambda x: ((batch_size / 2,) +
                     x[1:]) if _TRAIN else x[0:])(net)

        net = self._build_classifier(net)
        model = Model(input=main_input, output=[branch, net])
        if plot_model:
            plot(model, show_shapes=True)
        model.compile(loss={'classifier_output': 'categorical_crossentropy',
                      'domain_output': 'categorical_crossentropy'},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(input=main_input,
                           output=self.domain_invariant_features)
        return tsne_model


main_input = Input(shape=(3, img_rows, img_cols), name='main_input')

builder = DANNBuilder()
src_model = builder.build_source_model(main_input)
src_vis = builder.build_tsne_model(main_input)

dann_model = builder.build_dann_model(main_input)
dann_vis = builder.build_tsne_model(main_input)
print('Training source only model')
src_model.fit(X_train, y_train, batch_size=64, nb_epoch=10, verbose=1,
              validation_data=(X_test, y_test))
print('Evaluating target samples on source-only model')
print('Accuracy: ', src_model.evaluate(XT_test, y_test)[1])

# Broken out training loop for a DANN model.
src_index_arr = np.arange(X_train.shape[0])
target_index_arr = np.arange(XT_train.shape[0])

batches_per_epoch = len(X_train) / batch_size
num_steps = nb_epoch * batches_per_epoch
j = 0

print('Training DANN model')

for i in range(nb_epoch):

    batches = make_batches(X_train.shape[0], batch_size / 2)
    target_batches = make_batches(XT_train.shape[0], batch_size / 2)

    src_gen = batch_gen(batches, src_index_arr, X_train, y_train)
    target_gen = batch_gen(target_batches, target_index_arr, XT_train, None)

    losses = list()
    acc = list()

    print('Epoch ', i)

    for (xb, yb) in src_gen:

        # Update learning rate and gradient multiplier as described in
        # the paper.
        p = float(j) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        builder.grl.l = l
        builder.opt.lr = lr

        if xb.shape[0] != batch_size / 2:
            continue

        try:
            xt = target_gen.next()
        except:
            # Regeneration
            target_gen = target_gen(target_batches, target_index_arr, XT_train,
                                    None)

        # Concatenate source and target batch
        xb = np.vstack([xb, xt])

        metrics = dann_model.train_on_batch({'main_input': xb},
                                            {'classifier_output': yb,
                                            'domain_output': domain_labels},
                                            check_batch_dim=False)
        j += 1

print('Evaluating target samples on DANN model')
size = batch_size / 2
nb_testbatches = XT_test.shape[0] / size
acc = evaluate_dann(nb_testbatches, size)
print('Accuracy:', acc)
print('Visualizing output of domain invariant features')

# Plot both MNIST and MNIST-M
imshow_grid(X_train)
imshow_grid(XT_train)

src_embedding = src_vis.predict([combined_test_imgs])
src_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = src_tsne.fit_transform(src_embedding)

plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'Source only')

dann_embedding = dann_vis.predict([combined_test_imgs])
dann_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = dann_tsne.fit_transform(dann_embedding)

plot_embedding(tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'DANN')

plt.show()
