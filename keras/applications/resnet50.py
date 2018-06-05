from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import resnet50

ResNet50 = resnet50.ResNet50
decode_predictions = resnet50.decode_predictions
preprocess_input = resnet50.preprocess_input
