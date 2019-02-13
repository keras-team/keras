"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from . import backend as K
from .losses import mean_squared_error
from .losses import mean_absolute_error
from .losses import mean_absolute_percentage_error
from .losses import mean_squared_logarithmic_error
from .losses import hinge
from .losses import logcosh
from .losses import squared_hinge
from .losses import categorical_crossentropy
from .losses import sparse_categorical_crossentropy
from .losses import binary_crossentropy
from .losses import kullback_leibler_divergence
from .losses import poisson
from .losses import cosine_proximity
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    # If the shape of y_true is (num_samples, 1), flatten to (num_samples,)
    return K.mean(K.in_top_k(y_pred, K.cast(K.flatten(y_true), 'int32'), k),
                  axis=-1)



def iou_metric_bbox(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, x2, y1, y2]

    # AOG = Area of Groundtruth box
    AG = K.abs(K.transpose(y_true)[1] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose
                                                                             (y_true)[3] - K.transpose(y_true)[2] + 1)

    # AOP = Area of Predicted box
    AP = K.abs(K.transpose(y_pred)[1] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose
                                                                             (y_pred)[3] - K.transpose(y_pred)[2] + 1)

    # overlaps are the co-ordinates of intersection box
    area_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    area_1 = K.minimum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    area_2 = K.maximum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    area_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (area_1 -area_0 + 1) * (area_3 - area_2 + 1)

    # area of union of both boxes
    union = AG + AP - intersection

    # iou calculation
    iou = intersection / union

    #avoiding divide by zero
    if union == 0:
        union = 0.0001

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return (iou)


def iou_bbox(y_true, y_pred):
    num_images = K.int_shape(y_pred)[-1]
    # print(y_pred.shape)
    if y_pred.shape[1] != 4:
        raise Exception(
            'BBox metric takes columns in the format. (x1,x2,y1,y2).'
            'Target shape should have 4 values in column. No of columns found: {} .Please consider changing metric function for this problem.'.format(
                y_pred.shape[1]))
    if y_true.shape[1] != 4:
        raise Exception(
            'BBox metric takes columns in the format. (x1,x2,y1,y2).'
            'Source shape should have 4 values in column. No of columns found: {} .Please consider changing metric function for this problem.'.format(
                y_pred.shape[1]))
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_images):
        total_iou = total_iou + iou_metric_bbox(y_true, y_pred)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_images

# Aliases

intersection_over_union_bbox = bbox = iou_bbox
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def serialize(metric):
    return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, dict):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif isinstance(identifier, six.string_types):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
