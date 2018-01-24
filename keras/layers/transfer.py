# -*- coding: utf-8 -*-
""" layers for Transfer Learning
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense
from .. import backend as K
from ..engine import Layer

from six.moves import zip_longest


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, sigma=None):
    n = source.shape[0] + target.shape[0]
    total = K.concatenate([source, target])
    square = K.reshape(K.sum(K.square(total), axis=-1), [-1, 1])
    distance = square - 2 * K.dot(total, K.transpose(total)) + K.transpose(square)
    bandwidth = K.stop_gradient(K.sum(distance) / K.cast(n * (n - 1), dtype='float32')
                                if sigma is None else K.constant(sigma, dtype='float32'))
    bandwidth_list = [bandwidth * (kernel_mul ** (i - kernel_num // 2))
                      for i in range(kernel_num)]
    return K.sum([K.exp(-distance / i) for i in bandwidth_list])


class MMD(Layer):
    """
    The MMD loss layer
    """
    def __init__(self, n_classes, loss_weights, **kwargs):
        super(MMD, self).__init__(**kwargs)
        self.is_placeholder = True
        self.n_classes = n_classes
        self.loss_weights = loss_weights

    @staticmethod
    def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, sigma=None):
        n_source, n_target = source.shape[0], target.shape[0]
        kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, sigma=sigma)
        n_source_float = K.cast(n_source, dtype='float32')
        n_target_float = K.cast(n_target, dtype='float32')
        return K.sum(kernels[:n_source, :n_source]) / K.square(n_source_float) \
            + K.sum(kernels[n_source:, n_source:]) / K.square(n_target_float) \
            - 2.0 * K.sum(kernels[:n_source, n_source:]) / n_source_float / n_target_float

    def union_loss(self, feats, labels):
        logits = Dense(self.n_classes)(feats)

        train_size = labels.shape[0]
        source_feats, target_feats = feats[:train_size, :], feats[train_size:, :]
        source_logits, target_logits = logits[:train_size, :], feats[train_size:, :]

        cross_entropy_loss = K.sparse_categorical_crossentropy(target=labels,
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
        feats = inputs[0]   # features for training and testing data
        labels = inputs[1]  # labels for training data
        loss = self.union_loss(feats, labels)
        self.add_loss(loss, inputs=inputs)

        outputs = Dense(self.n_classes, activation="softmax", name='predictions')(feats)
        # output the predictions
        return outputs
