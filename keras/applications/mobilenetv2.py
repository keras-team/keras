from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import mobilenet_v2

MobileNetV2 = mobilenet_v2.MobileNetV2
decode_predictions = mobilenet_v2.decode_predictions
preprocess_input = mobilenet_v2.preprocess_input
