from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend
from .. import layers
from .. import models
from .. import utils

import keras_applications

if not hasattr(keras_applications, 'get_submodules_from_kwargs'):
    keras_applications.set_keras_submodules(
        backend=backend,
        layers=layers,
        models=models,
        utils=utils)


def keras_modules_injection(base_fun):

    def wrapper(*args, **kwargs):
        if hasattr(keras_applications, 'get_submodules_from_kwargs'):
            kwargs['backend'] = backend
            kwargs['layers'] = layers
            kwargs['models'] = models
            kwargs['utils'] = utils
        return base_fun(*args, **kwargs)

    return wrapper


from .vgg16 import VGG16
from .vgg19 import VGG19
from .resnet50 import ResNet50
from .inception_v3 import InceptionV3
from .inception_resnet_v2 import InceptionResNetV2
from .xception import Xception
from .mobilenet import MobileNet
from .mobilenet_v2 import MobileNetV2
from .densenet import DenseNet121, DenseNet169, DenseNet201
from .nasnet import NASNetMobile, NASNetLarge
