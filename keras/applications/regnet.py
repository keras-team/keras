"""Regnet models in Keras.

References:
  - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
    (CVPR 2020)
  - [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
    (CVPR 2021)
"""

from keras.applications.efficientnet import BASE_DOCSTRING, BASE_WEIGHTS_PATH
import tensorflow as tf

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import layer_utils
from keras.utils import data_utils

layers = VersionAwareLayers()

BASE_WEIGHTS_PATH = ""

WEIGHTS_HASHES = {
    "x002": ("", ""),
    "x004": ("", ""),
    "x006": ("", ""),
    "x008": ("", ""),
    "x016": ("", ""),
    "x032": ("", ""),
    "x040": ("", ""),
    "x064": ("", ""),
    "x080": ("", ""),
    "x120": ("", ""),
    "x160": ("", ""),
    "x320": ("", ""),
    "y002": ("", ""),
    "y004": ("", ""),
    "y006": ("", ""),
    "y008": ("", ""),
    "y016": ("", ""),
    "y032": ("", ""),
    "y040": ("", ""),
    "y064": ("", ""),
    "y080": ("", ""),
    "y120": ("", ""),
    "y160": ("", ""),
    "y320": ("", ""),
    "z002": ("", ""),
    "z004": ("", ""),
    "z006": ("", ""),
    "z008": ("", ""),
    "z016": ("", ""),
    "z032": ("", ""),
    "z040": ("", ""),
    "z064": ("", ""),
    "z080": ("", ""),
    "z120": ("", ""),
    "z160": ("", ""),
    "z320": ("", "")
}

MODEL_CONFIGS= {
    "x002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224
    },
    "x004": {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "group_width": 16,
        "default_size": 224
    },
    "x006": {
        "depths": [1, 3, 5, 7],
        "widths": [48, 96, 240, 528],
        "group_width": 24,
        "default_size": 224
    },
    "x008": {
        "depths": [1, 3, 7, 5],
        "widths": [64, 128, 288, 672],
        "group_width": 16,
        "default_size": 224
    },
    "x016": {
        "depths": [2, 4, 10, 2],
        "widths": [72, 168, 408, 912],
        "group_width": 24,
        "default_size": 224
    },
    "x032": {
        "depths": [2, 6, 15, 2],
        "widths": [96, 192, 432, 1008],
        "group_width": 48,
        "default_size": 224
    },
    "x040": {
        "depths": [2, 5, 14, 2],
        "widths": [80, 240, 560, 1360],
        "group_width": 40,
        "default_size": 224
    },
    "x064": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 392, 784, 1624],
        "group_width": 56,
        "default_size": 224
    },
    "x080": {
        "depths": [2, 5, 15, 1],
        "widths": [80, 240, 720, 1920],
        "group_width": 120,
        "default_size": 224
    },
    "x120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224
    },
    "x160": {
        "depths": [2, 6, 13, 1],
        "widths": [256, 512, 896, 2048],
        "group_width": 128,
        "default_size": 224
    },
    "x320": {
        "depths": [2, 7, 13, 1],
        "widths": [336, 672, 1344, 2520],
        "group_width": 168,
        "default_size": 224
    },
    "y002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224
    },
    "y004": {
        "depths": [1, 3, 6, 6],
        "widths": [48, 104, 208, 440],
        "group_width": 8,
        "default_size": 224
    },
    "y006": {
        "depths": [1, 3, 7, 4],
        "widths": [48, 112, 256, 608],
        "group_width": 16,
        "default_size": 224
    },
    "y008": {
        "depths": [1, 3, 8, 2],
        "widths": [64, 128, 320, 768],
        "group_width": 16,
        "default_size": 224
    },
    "y016": {
        "depths": [2, 6, 17, 2],
        "widths": [48, 120, 336, 888],
        "group_width": 24,
        "default_size": 224
    },
    "y032": {
        "depths": [2, 5, 13, 1],
        "widths": [72, 216, 576, 1512],
        "group_width": 24,
        "default_size": 224
    },
    "y040": {
        "depths": [2, 6, 12, 2],
        "widths": [128, 192, 512, 1088],
        "group_width": 64,
        "default_size": 224
    },
    "y064": {
        "depths": [2, 7, 14, 2],
        "widths": [144, 288, 576, 1296],
        "group_width": 72,
        "default_size": 224
    },
    "y080": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 448, 896, 2016],
        "group_width": 56,
        "default_size": 224
    },
    "y120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224
    },
    "y160": {
        "depths": [2, 4, 11, 1],
        "widths": [224, 448, 1232, 3024],
        "group_width": 112,
        "default_size": 224
    },
    "y320": {
        "depths": [2, 5, 12, 1],
        "widths": [232, 696, 1392, 3712],
        "group_width": 232,
        "default_size": 224
    }
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
  default_size,
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
    default_size: Default input image size. 
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
  if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  
  x = inputs
  x = Stem(x)

  in_channels = 32 # Output from Stem

  for num_stage in range(4):
    depth = depths[num_stage]
    out_channels = widths[num_stage]

    x = Stage(x, block_type, depth, group_width, 
              in_channels, out_channels) 
    in_channels = out_channels

  if include_top:
    x = Head(x, num_classes=classes)
    imagenet_utils.validate_activation(classifier_activation, weights)
    
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)
  
  model = training.Model(inputs, x, name=model_name)

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      file_suffix = '.h5'
      file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
    else:
      file_suffix = '_notop.h5'
      file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
    file_name = model_name + file_suffix
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)
  
  return model


## Instantiating variants ##

def RegNetX002(model_name="regnetx002",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x002"]["depths"],
      MODEL_CONFIGS["x002"]["widths"],
      MODEL_CONFIGS["x002"]["group_width"],
      "X",
      MODEL_CONFIGS["x002"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX004(model_name="regnetx004",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x004"]["depths"],
      MODEL_CONFIGS["x004"]["widths"],
      MODEL_CONFIGS["x004"]["group_width"],
      "X",
      MODEL_CONFIGS["x004"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX006(model_name="regnetx006",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x006"]["depths"],
      MODEL_CONFIGS["x006"]["widths"],
      MODEL_CONFIGS["x006"]["group_width"],
      "X",
      MODEL_CONFIGS["x006"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX008(model_name="regnetx008",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x008"]["depths"],
      MODEL_CONFIGS["x008"]["widths"],
      MODEL_CONFIGS["x008"]["group_width"],
      "X",
      MODEL_CONFIGS["x008"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX016(model_name="regnetx016",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x016"]["depths"],
      MODEL_CONFIGS["x016"]["widths"],
      MODEL_CONFIGS["x016"]["group_width"],
      "X",
      MODEL_CONFIGS["x016"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX032(model_name="regnetx032",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x032"]["depths"],
      MODEL_CONFIGS["x032"]["widths"],
      MODEL_CONFIGS["x032"]["group_width"],
      "X",
      MODEL_CONFIGS["x032"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX040(model_name="regnetx040",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x040"]["depths"],
      MODEL_CONFIGS["x040"]["widths"],
      MODEL_CONFIGS["x040"]["group_width"],
      "X",
      MODEL_CONFIGS["x040"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX064(model_name="regnetx064",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x064"]["depths"],
      MODEL_CONFIGS["x064"]["widths"],
      MODEL_CONFIGS["x064"]["group_width"],
      "X",
      MODEL_CONFIGS["x064"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX080(model_name="regnetx080",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x080"]["depths"],
      MODEL_CONFIGS["x080"]["widths"],
      MODEL_CONFIGS["x080"]["group_width"],
      "X",
      MODEL_CONFIGS["x080"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX120(model_name="regnetx120",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x120"]["depths"],
      MODEL_CONFIGS["x120"]["widths"],
      MODEL_CONFIGS["x120"]["group_width"],
      "X",
      MODEL_CONFIGS["x120"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX160(model_name="regnetx160",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x160"]["depths"],
      MODEL_CONFIGS["x160"]["widths"],
      MODEL_CONFIGS["x160"]["group_width"],
      "X",
      MODEL_CONFIGS["x160"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetX320(model_name="regnetx320",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["x320"]["depths"],
      MODEL_CONFIGS["x320"]["widths"],
      MODEL_CONFIGS["x320"]["group_width"],
      "X",
      MODEL_CONFIGS["x320"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY002(model_name="regnety002",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y002"]["depths"],
      MODEL_CONFIGS["y002"]["widths"],
      MODEL_CONFIGS["y002"]["group_width"],
      "Y",
      MODEL_CONFIGS["y002"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY004(model_name="regnety004",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y004"]["depths"],
      MODEL_CONFIGS["y004"]["widths"],
      MODEL_CONFIGS["y004"]["group_width"],
      "Y",
      MODEL_CONFIGS["y004"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY006(model_name="regnety006",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y006"]["depths"],
      MODEL_CONFIGS["y006"]["widths"],
      MODEL_CONFIGS["y006"]["group_width"],
      "Y",
      MODEL_CONFIGS["y006"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY008(model_name="regnety008",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y008"]["depths"],
      MODEL_CONFIGS["y008"]["widths"],
      MODEL_CONFIGS["y008"]["group_width"],
      "Y",
      MODEL_CONFIGS["y008"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY016(model_name="regnety016",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y016"]["depths"],
      MODEL_CONFIGS["y016"]["widths"],
      MODEL_CONFIGS["y016"]["group_width"],
      "Y",
      MODEL_CONFIGS["y016"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY032(model_name="regnety032",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y032"]["depths"],
      MODEL_CONFIGS["y032"]["widths"],
      MODEL_CONFIGS["y032"]["group_width"],
      "Y",
      MODEL_CONFIGS["y032"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY040(model_name="regnety040",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y040"]["depths"],
      MODEL_CONFIGS["y040"]["widths"],
      MODEL_CONFIGS["y040"]["group_width"],
      "Y",
      MODEL_CONFIGS["y040"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY064(model_name="regnety064",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y064"]["depths"],
      MODEL_CONFIGS["y064"]["widths"],
      MODEL_CONFIGS["y064"]["group_width"],
      "Y",
      MODEL_CONFIGS["y064"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY080(model_name="regnety080",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y080"]["depths"],
      MODEL_CONFIGS["y080"]["widths"],
      MODEL_CONFIGS["y080"]["group_width"],
      "Y",
      MODEL_CONFIGS["y080"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY120(model_name="regnety120",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y120"]["depths"],
      MODEL_CONFIGS["y120"]["widths"],
      MODEL_CONFIGS["y120"]["group_width"],
      "Y",
      MODEL_CONFIGS["y120"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY160(model_name="regnety160",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y160"]["depths"],
      MODEL_CONFIGS["y160"]["widths"],
      MODEL_CONFIGS["y160"]["group_width"],
      "Y",
      MODEL_CONFIGS["y160"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )


def RegNetY320(model_name="regnety320",
               include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  return RegNet(
      MODEL_CONFIGS["y320"]["depths"],
      MODEL_CONFIGS["y320"]["widths"],
      MODEL_CONFIGS["y320"]["group_width"],
      "Y",
      MODEL_CONFIGS["y320"]["default_size"],
      model_name=model_name,
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation
  )
