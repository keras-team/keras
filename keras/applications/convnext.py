# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=g-classes-have-attributes
# pylint: disable=g-direct-tensorflow-import
"""ConvNeXt models for Keras.

References:

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  (CVPR 2022)
"""

from keras import backend
from keras import layers
from keras import utils
from keras import Model
from keras.applications import imagenet_utils
from keras.engine import training

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHTS_PATH = "https://storage.googleapis.com/convnext-tf/keras-applications/convnext/"

WEIGHTS_HASHES = {
  "tiny":
    ("dec324e40ebe943afc7b75b72484646eeb092c04bb079df35911d7080364f9a8",
      "4d4f0e079db2cc0e627b55f7d0d76c367145d14f2c90674415373457cd822346"),
  "small":
    ("f964ea5cd5618a1e64902a74ca5ccff3797a4fa5dba11a14f2c4d1a562b72f08",
      "fd8f0ac74faa4e364d7cb5b2d32af9ae35b54ce5e80525b5beb7b7571320065a"),
  "base":
    ("d30e0c509f4e1abe2784d33765d4391ce8fbff259b0bd79f4a63684b20db87d2",
      "736f7a96cd933ee568611e29f334737fb9aebaaea021ea7adfe4d2f5cbb4a9aa"),
  "large":
    ("8a304c66deb782b0d59837bc13127068901adaaa280cfac604d3341aaf44b2cf",
      "b02b623b3c28586423e6be4aa214e2f5619280b97b4ef6b35ffb686e83235f01"),
  "xlarge":
    ("da65d1294d386c71aebd81bc2520b8d42f7f60eee4414806c60730cd63eb15cb",
      "2bfbf5f0c2b3f004f1c32e9a76661e11a9ac49014ed2a68a49ecd0cd6c88d377"),
}


MODEL_CONFIGS = {
  "tiny": {
    "depths": [3, 3, 9, 3],
    "projection_dims": [96, 192, 384, 768],
    "default_size": 224,
  },
  "small": {
    "depths": [3, 3, 27, 3],
    "projection_dims": [96, 192, 384, 768],
    "default_size": 224,
  },
  "base": {
    "depths": [3, 3, 27, 3],
    "projection_dims": [128, 256, 512, 1024],
    "default_size": 224,
  },
  "large": {
    "depths": [3, 3, 27, 3],
    "projection_dims": [192, 384, 768, 1536],
    "default_size": 224,
  },
  "xlarge": {
    "depths": [3, 3, 27, 3],
    "projection_dims": [256, 512, 1024, 2048],
    "default_size": 224,
  },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    (CVPR 2022)

  For image classification use cases, see
  [this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  The `base`, `large`, and `xlarge` models were first pre-trained on the
  ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
  pre-trained parameters of the models were assembled from the
  [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
  sense of how these parameters were converted to Keras compatible parameters,
  please refer to [this repository](https://github.com/sayakpaul/keras-convnext-conversion).

  Note: Each Keras Application expects a specific kind of input preprocessing.
  For ConvNeXt, preprocessing is included in the model using a `Normalization` layer.
  ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with
  values in the [0-255] range.

  When calling the `summary()` method after instantiating a ConvNeXt model, prefer
  setting the `expand_nested` argument `summary()` to `True` to better investigate
  the instantiated model.

  Args:
    include_top: Whether to include the fully-connected
      layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
      `"imagenet"` (pre-training on ImageNet-1k), or the path to the weights file
      to be loaded. Defaults to `"imagenet"`.
    input_tensor: Optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False.
      It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`. Defaults to None.
      - `None` means that the output of the model will be
        the 4D tensor output of the last convolutional layer.
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
      Defaults to `"softmax"`.
      When loading pretrained weights, `classifier_activation` can only
      be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""

class StochasticDepth(layers.Layer):
  """Stochastic Depth module. It performs batch-wise dropping rather than
    sample-wise. In libraries like `timm`, it's similar to `DropPath` layers
    that drops residual paths sample-wise.

  Reference:
    - https://github.com.rwightman/pytorch-image-models

  Args:
    drop_path (float): Probability of dropping paths. Should be within [0, 1].
  
  Returns:
    Tensor either with the residual path dropped or kept.

  """
  def __init__(self, drop_path, **kwargs):
    super().__init__(**kwargs)
    self.drop_path = drop_path

  def call(self, x, training=False):
    if training:
      keep_prob = 1 - self.drop_path
      shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
      random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
      random_tensor = tf.floor(random_tensor)
      return (x / keep_prob) * random_tensor
    return x

  def get_config(self):
    config = super().get_config()
    config.update({"drop_path": self.drop_path})
    return config


class ConvNeXtBlock(Model):
  """ConvNeXt block.
  
  References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

  Notes:
    In the original ConvNeXt implementation (linked above), the authors use
    `Dense` layers for pointwise convolutions for increased efficiency. 
    Following that, this implementation also uses the same.

  Args:
    projection_dim (int): Number of filters for convolution layers. In the
    ConvNeXt paper, this is referred to as projection dimension.
    drop_path (float): Probability of dropping paths. Should be within [0, 1].
    layer_scale_init_value (float): Layer scale value. Should be a small float
      number.

  Returns:
    A keras.Model instance.
  """
  def __init__(self, projection_dim, drop_path=0.0, 
    layer_scale_init_value=1e-6, **kwargs):
    super().__init__(**kwargs)
    self.projection_dim = projection_dim
    name = kwargs["name"]
    
    if layer_scale_init_value > 0.0:
      self.gamma = tf.Variable(
        layer_scale_init_value * tf.ones((projection_dim,)),
        name=name + "_layer_scale_gamma")
    else:
      self.gamma = None
    
    self.dw_conv_1 = layers.Conv2D(
      filters=projection_dim, kernel_size=7, padding="same",
      groups=projection_dim, name=name + "_depthwise_conv")
    self.layer_norm = layers.LayerNormalization(epsilon=1e-6, 
      name=name + "_layernorm")
    self.pw_conv_1 = layers.Dense(4 * projection_dim,
      name=name + "_pointwise_conv_1")
    self.act_fn = layers.Activation("gelu", name=name + "_gelu")
    self.pw_conv_2 = layers.Dense(projection_dim, 
      name=name + "_pointwise_conv_2")
    self.drop_path = (
      StochasticDepth(drop_path, name=name + "_stochastic_depth")
      if drop_path > 0.0
      else layers.Activation("linear", name=name + "_identity")
    )

  def call(self, inputs):
    x = inputs

    x = self.dw_conv_1(x)
    x = self.layer_norm(x)
    x = self.pw_conv_1(x)
    x = self.act_fn(x)
    x = self.pw_conv_2(x)

    if self.gamma is not None:
      x = self.gamma * x

    return inputs + self.drop_path(x)


def PreStem(name=None):
  """Normalizes inputs with ImageNet-1k mean and std.

  Args:
    name (str): Name prefix.

  Returns:
    Normalized tensor.
  """
  if name is None:
    name = "prestem" + str(backend.get_uid("prestem"))

  def apply(x):
    x = layers.Normalization(
      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
      variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
      name=name + "_prestem_normalization"
    )(x)
    return x

  return apply


def Head(num_classes=1000, name=None):
  """Implementation of classification head of RegNet.

  Args:
    num_classes: number of classes for Dense layer
    name: name prefix

  Returns:
    Output logits tensor.
  """
  if name is None:
    name = str(backend.get_uid("head"))

  def apply(x):
    x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
    x = layers.LayerNormalization(
      epsilon=1e-6, name=name + "_head_layernorm")(x)
    x = layers.Dense(num_classes, name=name + "head_dense")(x)
    return x

  return apply


def ConvNeXt(depths,
  projection_dims,
  drop_path_rate=0.0,
  layer_scale_init_value=1e-6,
  default_size=224,
  model_name="convnext",
  include_preprocessing=True,
  include_top=True,
  weights=None,
  input_tensor=None,
  input_shape=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax"):
  """Instantiates ConvNeXt architecture given specific configuration.

  Args:
    depths: An iterable containing depths for each individual stages.
    projection_dims: An iterable containing output number of channels of
    each individual stages.
    drop_path_rate: Stochastic depth probability. If 0.0, then stochastic depth
      won't be used.
    layer_scale_init_value: Layer scale coefficient. If 0.0, layer scaling won't
      be used.
    default_size: Default input image size.
    model_name: An optional name for the model.
    include_preprocessing: boolean denoting whther to include preprocessing in
      the model.
    include_top: Boolean denoting whether to include classification head to the
      model.
    weights: one of `None` (random initialization), `"imagenet"` (pre-training 
      on ImageNet-1k), or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
      as image input for the model.
    input_shape: optional shape tuple, only to be specified if `include_top` is
      False. It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction when `include_top` is
      `False`. - `None` means that the output of the model will be the 4D tensor
      output of the last convolutional layer. - `avg` means that global average
      pooling will be applied to the output of the last convolutional layer, and
      thus the output of the model will be a 2D tensor. - `max` means that
      global max pooling will be applied.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True, and if no `weights` argument is
      specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

  Returns:
    A `keras.Model` instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax`, or `None` 
        when using a pretrained top layer.
      ValueError: if `include_top` is True but `num_classes` is not 1000 
        when using ImageNet.
  """
  if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
    raise ValueError("The `weights` argument should be either "
                     "`None` (random initialization), `imagenet` "
                     "(pre-training on ImageNet), "
                     "or the path to the weights file to be loaded.")

  if weights == "imagenet" and include_top and classes != 1000:
    raise ValueError("If using `weights` as `'imagenet'` with `include_top`"
                     " as true, `classes` should be 1000")

  # Determine proper input shape.
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
    inputs = utils.layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  x = inputs
  if include_preprocessing:
    x = PreStem(name=model_name)(x)
  
  # Stem block.
  stem = tf.keras.Sequential(
    [
      layers.Conv2D(projection_dims[0], kernel_size=4, strides=4,
        name=model_name + "_stem_conv"),
      layers.LayerNormalization(epsilon=1e-6, 
        name=model_name + "_stem_layernorm"),
    ],
    name=model_name + "_stem",
  )

  # Downsampling blocks.
  downsample_layers = []
  downsample_layers.append(stem)
  for i in range(3):
    downsample_layer = tf.keras.Sequential(
      [
        layers.LayerNormalization(epsilon=1e-6, 
          name=model_name + "_downsampling_layernorm_" + str(i)),
        layers.Conv2D(projection_dims[i + 1], kernel_size=2, strides=2,
          name=model_name + "_downsampling_conv_" + str(i)),
      ],
      name=model_name + "_downsampling_block_" + str(i),
    )
    downsample_layers.append(downsample_layer)
  
  # Stochastic depth schedule.
  dp_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

  # ConvNeXt stages.
  stages = []
  cur = 0
  for i in range(4):
    stage = tf.keras.Sequential(
      [
        *[
            ConvNeXtBlock(
              projection_dim=projection_dims[i],
              drop_path=dp_rates[cur + j],
              layer_scale_init_value=layer_scale_init_value,
              name=model_name + f"_stage_{i}_block_{j}",
            )
            for j in range(depths[i])
        ]
      ],
      name=model_name + f"_stage_{i}",
    )
    stages.append(stage)
    cur += depths[i]

  # Apply the stages.
  for i in range(len(stages)):
    x = downsample_layers[i](x)
    x = stages[i](x)

  if include_top:
    x = Head(num_classes=classes)(x)
    imagenet_utils.validate_activation(classifier_activation, weights)

  else:
    if pooling == "avg":
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == "max":
      x = layers.GlobalMaxPooling2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

  model = training.Model(inputs=inputs, outputs=x, name=model_name)

  # Load weights.
  if weights == "imagenet":
    if include_top:
      file_suffix = ".h5"
      file_hash = WEIGHTS_HASHES[model_name][0]
    else:
      file_suffix = "_notop.h5"
      file_hash = WEIGHTS_HASHES[model_name][1]
    file_name = model_name + file_suffix
    weights_path = utils.data_utils.get_file(
      file_name,
      BASE_WEIGHTS_PATH + file_name,
      cache_subdir="models",
      file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


## Instantiating variants ##

@keras_export("keras.applications.convnext.ConvNeXtTiny",
              "keras.applications.ConvNeXtTiny")
def ConvNeXtTiny(model_name="convnext_tiny",
  include_top=True,
  include_preprocessing=True,
  weights="imagenet",
  input_tensor=None,
  input_shape=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax"):
  return ConvNeXt(
    depths=MODEL_CONFIGS["tiny"]["depths"],
    projection_dims=["tiny"]["projection_dims"],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=["tiny"]["default_size"],
    model_name=model_name,
    include_top=include_top,
    include_preprocessing=include_preprocessing,
    weights=weights,
    input_tensor=input_tensor,
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    classifier_activation=classifier_activation)


@keras_export("keras.applications.convnext.ConvNeXtSmall",
              "keras.applications.ConvNeXtSmall")
def ConvNeXtSmall(model_name="convnext_small",
  include_top=True,
  include_preprocessing=True,
  weights="imagenet",
  input_tensor=None,
  input_shape=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax"):
  return ConvNeXt(
    depths=MODEL_CONFIGS["small"]["depths"],
    projection_dims=["small"]["projection_dims"],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=["small"]["default_size"],
    model_name=model_name,
    include_top=include_top,
    include_preprocessing=include_preprocessing,
    weights=weights,
    input_tensor=input_tensor,
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    classifier_activation=classifier_activation)


@keras_export("keras.applications.convnext.ConvNeXtBase",
              "keras.applications.ConvNeXtBase")
def ConvNeXtBase(model_name="convnext_base",
  include_top=True,
  include_preprocessing=True,
  weights="imagenet",
  input_tensor=None,
  input_shape=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax"):
  return ConvNeXt(
    depths=MODEL_CONFIGS["base"]["depths"],
    projection_dims=["base"]["projection_dims"],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=["base"]["default_size"],
    model_name=model_name,
    include_top=include_top,
    include_preprocessing=include_preprocessing,
    weights=weights,
    input_tensor=input_tensor,
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    classifier_activation=classifier_activation)


@keras_export("keras.applications.convnext.ConvNeXtLarge",
              "keras.applications.ConvNeXtLarge")
def ConvNeXtLarge(model_name="convnext_large",
  include_top=True,
  include_preprocessing=True,
  weights="imagenet",
  input_tensor=None,
  input_shape=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax"):
  return ConvNeXt(
    depths=MODEL_CONFIGS["large"]["depths"],
    projection_dims=["large"]["projection_dims"],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=["large"]["default_size"],
    model_name=model_name,
    include_top=include_top,
    include_preprocessing=include_preprocessing,
    weights=weights,
    input_tensor=input_tensor,
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    classifier_activation=classifier_activation)


@keras_export("keras.applications.convnext.ConvNeXtXLarge",
              "keras.applications.ConvNeXtXLarge")
def ConvNeXtXLarge(model_name="convnext_xlarge",
  include_top=True,
  include_preprocessing=True,
  weights="imagenet",
  input_tensor=None,
  input_shape=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax"):
  return ConvNeXt(
    depths=MODEL_CONFIGS["xlarge"]["depths"],
    projection_dims=["xlarge"]["projection_dims"],
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=["xlarge"]["default_size"],
    model_name=model_name,
    include_top=include_top,
    include_preprocessing=include_preprocessing,
    weights=weights,
    input_tensor=input_tensor,
    input_shape=input_shape,
    pooling=pooling,
    classes=classes,
    classifier_activation=classifier_activation)


ConvNeXtTiny.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtTiny")
ConvNeXtSmall.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtSmall")
ConvNeXtBase.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtBase")
ConvNeXtLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtLarge")
ConvNeXtXLarge.__doc__ = BASE_DOCSTRING.format(name="ConvNeXtXLarge")


@keras_export("keras.applications.convnext.preprocess_input")
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
  """A placeholder method for backward compatibility.

  The preprocessing logic has been included in the efficientnet model
  implementation. Users are no longer required to call this method to normalize
  the input data. This method does nothing and only kept as a placeholder to
  align the API surface between old and new version of model.

  Args:
    x: A floating point `numpy.array` or a `tf.Tensor`.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it, it
      defaults to "channels_last").{mode}

  Returns:
    Unchanged `numpy.array` or `tf.Tensor`.
  """
  return x


@keras_export("keras.applications.convnext.decode_predictions")
def decode_predictions(preds, top=5):
  return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
