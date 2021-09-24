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
  `block_type` is one of `(X, Y, Z)` and `flops` signifies hundred million 
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


def Stem(x):
  """Implementation of Regnet stem. (Common to all models)
  
  Args:
    x: Input tensor. Should be 224x224, rescaled and normalized to [0,1].   

  Returns:
    Output tensor of the Stem
  """
  x = layers.Conv2D(32, (3, 3), strides=2, use_bias=False)(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = layers.ReLU()(x)

  return x


def XBlock(inputs,
           filters_in,
           filters_out,
           group_width,
           stride=1):
  """
  Implementation of X Block. 
  Reference: [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

  Args:
    inputs: input tensor
    filters_in: Filters in the input tensor
    filters_out: Filters in the output tensor
    group_width: Group width
    stride: Stride

  Return:
    Output tensor of the block 
  """

  if filters_in != filters_out and stride == 1:
    raise ValueError("""Input filters and output filters are not equal for stride
                         1. Please check inputs and try again.""")

  # Declare layers
  groups = filters_out // group_width

  relu1x1 = layers.ReLU()
  relu3x3 = layers.ReLU()
  relu = layers.ReLU()
  
  if stride != 1:
    skip = layers.Conv2D(filters_out, (1,1), strides=stride, use_bias=False)(inputs)
    skip = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(skip)
    conv_3x3 = layers.Conv2D(filters_out, (3, 3), use_bias=False, strides=stride,
        groups=groups)
  else:
    skip = inputs
    conv_3x3 = layers.Conv2D(filters_out, (3, 3), use_bias=False,
                             groups=groups)
  
  # Build block
  # conv_1x1_1
  x = layers.Conv2D(filters_out, (1,1), use_bias=False)(inputs)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = relu1x1(x)
  
  # conv_3x3
  x = conv_3x3(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = relu3x3(x)
  
  # conv_1x1_2
  x = layers.Conv2D(filters_out, (1, 1), use_bias=False)(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

  x = relu(x + skip)

  return x


def YBlock(inputs,
           filters_in,
           filters_out,
           group_width,
           stride=1,
           se_ratio=0.25):
  """
  Implementation of Y Block. 
  Reference: [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

  Args:
    inputs: input tensor
    filters_in: Filters in the input tensor
    filters_out: Filters in the output tensor
    group_width: Group width
    stride: Stride
    se_ratio: Expansion ration for Squeeze and Excite block

  Return:
    Output tensor of the block 
  """

  if filters_in != filters_out and stride == 1:
    raise ValueError("""Input filters and output filters are not equal for stride
                         1. Please check inputs and try again.""")

  groups = filters_out // group_width
  se_filters = int(filters_out * se_ratio)

  relu1x1 = layers.ReLU()
  relu3x3 = layers.ReLU()
  relu = layers.ReLU()



  if stride != 1:
    skip = layers.Conv2D(filters_out, (1, 1),
                         strides=stride, use_bias=False)(inputs)
    skip = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(skip)
    conv_3x3 = layers.Conv2D(filters_out, (3, 3), use_bias=False, strides=stride,
                             groups=groups)
  else:
    skip = inputs
    conv_3x3 = layers.Conv2D(filters_out, (3, 3), use_bias=False,
                             groups=groups)

  # Build block
  # conv_1x1_1
  x = layers.Conv2D(filters_out, (1, 1), use_bias=False)(inputs)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = relu1x1(x)

  # # conv_3x3
  x = conv_3x3(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = relu3x3(x)

  # SE 
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Reshape((1, 1, filters_out))(x)
  x = layers.Conv2D(se_filters, (1, 1), activation="relu")(x)
  x = layers.Conv2D(filters_out, (1, 1), activation="sigmoid")(x)


  # conv_1x1_2
  x = layers.Conv2D(filters_out, (1, 1), use_bias=False)(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

  x = relu(x + skip)

  return x


def ZBlock(inputs,
           filters_in,
           filters_out,
           group_width,
           stride=1,
           se_ratio=0.25,
           b=0.25):
  """Implementation of Z block
  Reference: [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
  Note that Z block can be completely 
  
  Args:
    inputs: input tensor
    filters_in: Filters in the input tensor
    filters_out: Filters in the output tensor
    group_width: Group width
    stride: Stride
    se_ratio: Expansion ration for Squeeze and Excite block
    b: inverted bottleneck ratio 
  Return:
    Output tensor of the block 
  """

  if filters_in != filters_out and stride == 1:
    raise ValueError("""Input filters and output filters are not equal for stride
                         1. Please check inputs and try again.""")
  
  groups = filters_out // group_width
  se_filters = int(filters_out * se_ratio)

  inv_btlneck_filters = int(filters_out / b)
  if stride != 1:
    conv_3x3 = layers.Conv2D(inv_btlneck_filters, (3, 3), use_bias=False, strides=stride,
                             groups=groups)
  else:
    conv_3x3 = layers.Conv2D(inv_btlneck_filters, (3, 3), use_bias=False,
                             groups=groups)

  # Build block
  # conv_1x1_1
  x = layers.Conv2D(inv_btlneck_filters, (1, 1), use_bias=False)(inputs)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = tf.nn.silu(x)

  # # conv_3x3
  x = conv_3x3(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
  x = tf.nn.silu(x)

  # SE
  x = layers.GlobalAveragePooling2D(name="Y_GlobalAvgPool")(x)
  x = layers.Reshape((1, 1, inv_btlneck_filters))(x)
  x = layers.Conv2D(se_filters, (1, 1), activation=tf.nn.silu)(x)
  x = layers.Conv2D(inv_btlneck_filters, (1, 1), activation="sigmoid")(x)

  # conv_1x1_2
  x = layers.Conv2D(filters_out, (1, 1), use_bias=False)(x)
  x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

  if stride != 1:
    return x
  else:
    return x + inputs


def Stage(inputs,  block_type, depth, group_width, filters_in, filters_out):
  """Implementation of Stage in RegNet.

  Args:
    inputs: Input tensor
    block_type: Must be one of "X", "Y", "Z"
    depth: Depth of stage, number of blocks to use
    group_width: Group width of all blocks in  this stage
    filters_in: Input filters to this stage
    filters_out: Output filters from this stage

  Returns:
    Output tensor of Stage
  """
  x = inputs
  if block_type == "X":
    x = XBlock(x, filters_in, filters_out, group_width, stride=2)
    for _ in range(depth - 1):
      x = XBlock(x, filters_out, filters_out, group_width)
  elif block_type == "Y":
    x = YBlock(x, filters_in, filters_out, group_width, stride=2)
    for _ in range(depth - 1):
      x = YBlock(x, filters_out, filters_out, group_width)
  elif block_type == "Z":
    x = ZBlock(x, filters_in, filters_out, group_width, stride=2)
    for _ in range(depth - 1):
      x = ZBlock(x, filters_out, filters_out, group_width)
  else:
    raise NotImplementedError(f"""Block type {block_type} not implemented. 
                              block_type must be one of ("X", "Y", "Z"). """)
  return x


def Head(x, num_classes=1000):
  """Implementation of classification head of RegNet
  
  Args:
  x: Input tensor
    num_classes: Classes for Dense layer
  
  Returns:
    Output logits tensor. 
  """

  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(num_classes)(x)

  return x


def RegNet(
  depths,
  widths,
  group_width,
  block_type,
  model_name='regnet',
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
