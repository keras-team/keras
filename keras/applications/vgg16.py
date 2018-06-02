from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import vgg16

VGG16 = vgg16.VGG16
decode_predictions = vgg16.decode_predictions
preprocess_input = vgg16.preprocess_input
