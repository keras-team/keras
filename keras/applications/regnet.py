"""Regnet models in Keras.

References:
  - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
    (CVPR 2020)
  - [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
    (CVPR 2021)
"""

from keras.applications.efficientnet import BASE_DOCSTRING, BASE_WEIGHTS_PATH
import tensorflow as tf

from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers

layers = VersionAwareLayers()

BASE_WEIGHTS_PATH = ""

WEIGHTS_HASHES = {
    "x2": ("", ""),
    "x4": ("", ""),
    "x6": ("", ""),
    "x8": ("", ""),
    "x16": ("", ""),
    "x32": ("", ""),
    "x40": ("", ""),
    "x64": ("", ""),
    "x80": ("", ""),
    "x120": ("", ""),
    "x160": ("", ""),
    "x320": ("", ""),
    "y2": ("", ""),
    "y4": ("", ""),
    "y6": ("", ""),
    "y8": ("", ""),
    "y16": ("", ""),
    "y32": ("", ""),
    "y40": ("", ""),
    "y64": ("", ""),
    "y80": ("", ""),
    "y120": ("", ""),
    "y160": ("", ""),
    "y320": ("", ""),
    "z2": ("", ""),
    "z4": ("", ""),
    "z6": ("", ""),
    "z8": ("", ""),
    "z16": ("", ""),
    "z32": ("", ""),
    "z40": ("", ""),
    "z64": ("", ""),
    "z80": ("", ""),
    "z120": ("", ""),
    "z160": ("", ""),
    "z320": ("", "")
}

layers = VersionAwareLayers()

BASE_DOCSTRING = """ Instantiates Regnet architecture.

  References:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
    (CVPR 2020)
    - [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
    (CVPR 2021)

  For image classification use cases, see
  [this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For Regnets, preprocessing is included in the model using a `Rescaling` layer.
  RegNet models expect their inputs to be float or uint8 tensors of pixels with 
  values in the [0-255] range.

  The naming of models is as follows: `RegNet{block_type}{flops}` where 
  `block_type` is one of `{X, Y, Z}` and `flops` signifies hundred million 
  floating point operations. For example RegNetY64 corresponds to RegNet with 
  Y block and 6.4 giga flops (64 hundred million flops). 

  Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded. Defaults to 'imagenet'.
    input_tensor: Optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`. Defaults to None.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified. Defaults to 1000 (number of
        ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to 'softmax'.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""

def RegNet(
    depths,
    widths,
    group_width,
    block_type,
    model_name='regnet',
    activation='relu',
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
    """ Instantiates RegNet architecture given specific configuration.

    Args:
      depths: An iterable containing depths for each individual stages. 
      widths: An iterable containing output channel width of each individual 
        stages
      group_width: Number of channels to be used in each group. See grouped 
        convolutions for more information.
      block_type: Must be one of {'x', 'y', 'z'}. For more details see the
        papers 'Designing network design spaces' and 'Fast and Accurate Model 
        Scaling'
      model_name: An optional name for the model.
      activation: A string or callable denoting the activation to be used. 
        Defaults to 'relu'.
      include_top: Boolean denoting whether to include classification head to 
        the model.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.        
    
    Returns:
    A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
          using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000.
        ValueError: if `block_type` is not one of `{'x', 'y', 'z'}`

    
    """
    pass