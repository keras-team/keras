# -*- coding: utf-8 -*-
""" layers for Transfer Learning
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine import InputSpec
from keras.layers import Dense
from .. import backend as K
from ..engine import Layer

from six.moves import zip_longest


def gaussian_kernel(source, target, sigma=None):
    # n = source.shape[0] + target.shape[0]
    # print("type of n: ", type(n))
    # print("value of n: ", n)
    total = K.concatenate([source, target], axis=0)
    square = K.reshape(K.sum(K.square(total), axis=-1), [-1, 1])
    distance = square - 2 * K.dot(total, K.transpose(total)) + K.transpose(square)
    # TODO: what should the default value of \sigma be?
    # temporarily use d_ij / |D|
    bandwidth = K.stop_gradient(
        # K.sum(distance) / K.count_params(distance) if sigma is None
        K.sum(distance) if sigma is None
        else float(sigma))
    return K.exp(-distance / bandwidth)


class MMD(Layer):
    """
    The MMD loss layer
    """
    def __init__(self, n_classes, loss_weights, **kwargs):
        super(MMD, self).__init__(**kwargs)
        # self.input_spec = InputSpec(ndim=4)
        # self.is_placeholder = True
        self.n_classes = n_classes
        self.loss_weights = loss_weights

    @staticmethod
    def mmd_loss(source, target, sigma=None):
        n_source, n_target = source.shape[0], target.shape[0]
        kernels = gaussian_kernel(source, target, sigma=sigma)
        n_source_float = K.cast(n_source, dtype='float32')
        n_target_float = K.cast(n_target, dtype='float32')
        return K.sum(kernels[:n_source, :n_source]) / K.square(n_source_float) \
            + K.sum(kernels[n_source:, n_source:]) / K.square(n_target_float) \
            - 2.0 * K.sum(kernels[:n_source, n_source:]) / n_source_float / n_target_float

    def union_loss(self, source_feats, target_feats, source_labels):
        fc = Dense(self.n_classes)
        source_logits = fc(source_feats)
        target_logits = fc(target_feats)

        cross_entropy_loss = K.sparse_categorical_crossentropy(target=source_labels,
                                                               output=source_logits,
                                                               from_logits=True)
        mmd_losses = [
            self.mmd_loss(source_feats, target_feats),
            self.mmd_loss(source_logits, target_logits)
        ]

        loss = K.sum([w * l if w is not None else l
                      for w, l in zip_longest(self.loss_weights, mmd_losses)]) + cross_entropy_loss

        return loss

    def call(self, inputs):
        source_feats = inputs[0]   # features for training data
        target_feats = inputs[1]   # features for testing data
        source_labels = inputs[2]  # labels for training data
        loss = self.union_loss(source_feats, target_feats, source_labels)
        self.add_loss(loss, inputs=inputs)

        total_feats = K.concatenate([source_feats, target_feats], axis=0)
        outputs = Dense(self.n_classes, activation="softmax", name='predictions')(total_feats)
        # output the predictions
        return outputs
