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

import tensorflow.compat.v2 as tf
from keras import backend, layers
from keras.applications import imagenet_utils
from keras.engine import training
from keras.utils import data_utils, layer_utils
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHTS_PATH = "https://storage.googleapis.com/convnext-tf/keras-applications/convnext/"

WEIGHTS_HASHES = {
    "x002":
        ("49fb46e56cde07fdaf57bffd851461a86548f6a3a4baef234dd37290b826c0b8",
         "5445b66cd50445eb7ecab094c1e78d4d3d29375439d1a7798861c4af15ffff21"),
    "x004":
        ("3523c7f5ac0dbbcc2fd6d83b3570e7540f7449d3301cc22c29547302114e4088",
         "de139bf07a66c9256f2277bf5c1b6dd2d5a3a891a5f8a925a10c8a0a113fd6f3"),
    "x006":
        ("340216ef334a7bae30daac9f414e693c136fac9ab868704bbfcc9ce6a5ec74bb",
         "a43ec97ad62f86b2a96a783bfdc63a5a54de02eef54f26379ea05e1bf90a9505"),
    "x008":
        ("8f145d6a5fae6da62677bb8d26eb92d0b9dfe143ec1ebf68b24a57ae50a2763d",
         "3c7e4b0917359304dc18e644475c5c1f5e88d795542b676439c4a3acd63b7207"),
    "x016":
        ("31c386f4c7bfef4c021a583099aa79c1b3928057ba1b7d182f174674c5ef3510",
         "1b8e3d545d190271204a7b2165936a227d26b79bb7922bac5ee4d303091bf17a"),
    "x032":
        ("6c025df1409e5ea846375bc9dfa240956cca87ef57384d93fef7d6fa90ca8c7f",
         "9cd4522806c0fcca01b37874188b2bd394d7c419956d77472a4e072b01d99041"),
    "x040":
        ("ba128046c588a26dbd3b3a011b26cb7fa3cf8f269c184c132372cb20b6eb54c1",
         "b4ed0ca0b9a98e789e05000e830403a7ade4d8afa01c73491c44610195198afe"),
    "x064":
        ("0f4489c3cd3ad979bd6b0324213998bcb36dc861d178f977997ebfe53c3ba564",
         "3e706fa416a18dfda14c713423eba8041ae2509db3e0a611d5f599b5268a46c4"),
    "x080":
        ("76320e43272719df648db37271a247c22eb6e810fe469c37a5db7e2cb696d162",
         "7b1ce8e29ceefec10a6569640ee329dba7fbc98b5d0f6346aabade058b66cf29"),
    "x120":
        ("5cafc461b78897d5e4f24e68cb406d18e75f31105ef620e7682b611bb355eb3a",
         "36174ddd0299db04a42631d028abcb1cc7afec2b705e42bd28fcd325e5d596bf"),
    "x160":
        ("8093f57a5824b181fb734ea21ae34b1f7ee42c5298e63cf6d587c290973195d2",
         "9d1485050bdf19531ffa1ed7827c75850e0f2972118a996b91aa9264b088fd43"),
    "x320":
        ("91fb3e6f4e9e44b3687e80977f7f4412ee9937c0c704232664fc83e4322ea01e",
         "9db7eacc37b85c98184070e1a172e6104c00846f44bcd4e727da9e50d9692398"),
    "y002":
        ("1e8091c674532b1a61c04f6393a9c570113e0197f22bd1b98cc4c4fe800c6465",
         "f63221f63d625b8e201221499682587bfe29d33f50a4c4f4d53be00f66c0f12c"),
    "y004":
        ("752fdbad21c78911bf1dcb8c513e5a0e14697b068e5d9e73525dbaa416d18d8e",
         "45e6ba8309a17a77e67afc05228454b2e0ee6be0dae65edc0f31f1da10cc066b"),
    "y006":
        ("98942e07b273da500ff9699a1f88aca78dfad4375faabb0bab784bb0dace80a9",
         "b70261cba4e60013c99d130cc098d2fce629ff978a445663b6fa4f8fc099a2be"),
    "y008":
        ("1b099377cc9a4fb183159a6f9b24bc998e5659d25a449f40c90cbffcbcfdcae4",
         "b11f5432a216ee640fe9be6e32939defa8d08b8d136349bf3690715a98752ca1"),
    "y016":
        ("b7ce1f5e223f0941c960602de922bcf846288ce7a4c33b2a4f2e4ac4b480045b",
         "d7404f50205e82d793e219afb9eb2bfeb781b6b2d316a6128c6d7d7dacab7f57"),
    "y032":
        ("6a6a545cf3549973554c9b94f0cd40e25f229fffb1e7f7ac779a59dcbee612bd",
         "eb3ac1c45ec60f4f031c3f5180573422b1cf7bebc26c004637517372f68f8937"),
    "y040":
        ("98d00118b335162bbffe8f1329e54e5c8e75ee09b2a5414f97b0ddfc56e796f6",
         "b5be2a5e5f072ecdd9c0b8a437cd896df0efa1f6a1f77e41caa8719b7dfcb05d"),
    "y064":
        ("65c948c7a18aaecaad2d1bd4fd978987425604ba6669ef55a1faa0069a2804b7",
         "885c4b7ed7ea339daca7dafa1a62cb7d41b1068897ef90a5a3d71b4a2e2db31a"),
    "y080":
        ("7a2c62da2982e369a4984d3c7c3b32d6f8d3748a71cb37a31156c436c37f3e95",
         "3d119577e1e3bf8d153b895e8ea9e4ec150ff2d92abdca711b6e949c3fd7115d"),
    "y120":
        ("a96ab0d27d3ae35a422ee7df0d789069b3e3217a99334e0ce861a96595bc5986",
         "4a6fa387108380b730b71feea2ad80b5224b5ea9dc21dc156c93fe3c6186485c"),
    "y160":
        ("45067240ffbc7ca2591313fee2f80dbdda6d66ec1a7451446f9a6d00d8f7ac6e",
         "ead1e6b568be8f34447ec8941299a9df4368736ba9a8205de5427fa20a1fb316"),
    "y320": ("b05e173e4ae635cfa22d06392ee3741284d17dadfee68f2aa6fd8cb2b7561112",
             "cad78f74a586e24c61d38be17f3ae53bb9674380174d2585da1a526b8c20e1fd")
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

  `base`, `large`, and `xlarge` models were first pre-trained on the ImageNet-21k
  dataset and were then fine-tuned on the ImageNet-1k dataset. The pre-trained 
  parameters of the models were assembled from the [official repository](https://github.com/facebookresearch/ConvNeXt).
  To get a sense of how these parameters were converted to Keras compatible parameters,
  please refer to [this repository](https://github.com/sayakpaul/ConvNeXt-TF).

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
          `"imagenet"` (pre-training on ImageNet), or the path to the weights
          file to be loaded. Defaults to `"imagenet"`.
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


class Block(tf.keras.Model):
    """ConvNeXt block.
    
    References:
      - https://arxiv.org/abs/2201.03545
      - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Notes:
      In the original ConvNeXt implementation (linked above), the authors use
      `Dense` layers for pointwise convolutions for increased efficiency. Following
      that, this implementation also uses the same.

    Args:
      projection_dim (int): Number of filters for convolution layers. In the ConvNeXt paper, this is
        referred to as projection dimension.
      drop_path (float): Probability of dropping paths. Should be within [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float number.

    Returns:
      A keras.Model instance.
    """
    def __init__(self, projection_dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
      super().__init__(**kwargs)
      self.projection_dim = projection_dim
      self.name = kwargs["name"]
      
      if layer_scale_init_value > 0.0:
        self.gamma = tf.Variable(layer_scale_init_value * tf.ones((projection_dim,)), name=self.name + "_layer_scale_gamma")
      else:
        self.gamma = None
      
      self.dw_conv_1 = layers.Conv2D(
        filters=projection_dim, kernel_size=7, padding="same", groups=projection_dim,
        name=self.name + "_depthwise_conv"
      )
      self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name=self.name + "_layernorm")
      self.pw_conv_1 = layers.Dense(4 * projection_dim, name=self.name + "_pointwise_conv_1")
      self.act_fn = layers.Activation("gelu", name=self.name + "_gelu")
      self.pw_conv_2 = layers.Dense(projection_dim, name=self.name + "_pointwise_conv_2")
      self.drop_path = (
        StochasticDepth(drop_path, name=self.name + "_stochastic_depth")
        if drop_path > 0.0
        else layers.Activation("linear", name=self.name + "_identity")
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
    x = layers.LayerNormalization(epsilon=1e-6, name=name + "_head_layernorm")(x)
    x = layers.Dense(num_classes, name=name + "head_dense")(x)
    return x

  return apply


def ConvNeXt(depths,
           projection_dims,
           drop_path_rate,
           layer_scale_init_value,
           default_size,
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
    projection_dims: An iterable containing output number of channels of each individual
      stages.
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
    weights: one of `None` (random initialization), "imagenet" (pre-training on
      ImageNet), or the path to the weights file to be loaded.
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
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
      ValueError: if `include_top` is True but `num_classes` is not 1000.
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
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  x = inputs
  if include_preprocessing:
    x = PreStem(name=model_name)(x)
  
  # Stem block.
  stem = tf.keras.Sequential(
    [
      layers.Conv2D(projection_dims[0], kernel_size=4, strides=4, name=model_name + "_stem_conv"),
      layers.LayerNormalization(epsilon=1e-6, name=model_name + "_stem_layernorm"),
    ],
    name=model_name + "_stem",
  )

  # Downsampling blocks.
  downsample_layers = []
  downsample_layers.append(stem)
  for i in range(3):
    downsample_layer = tf.keras.Sequential(
        [
          layers.LayerNormalization(epsilon=1e-6, name=model_name + "_downsampling_layernorm_" + str(i+1)),
          layers.Conv2D(projection_dims[i + 1], kernel_size=2, strides=2, name=model_name + "_downsampling_conv_" + str(i+1)),
        ],
        name=model_name + "_downsampling_block_" + str(i+1),
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
            Block(
              projection_dim=projection_dims[i],
              drop_path=dp_rates[cur + j],
              layer_scale_init_value=layer_scale_init_value,
              name=model_name + f"stage_{i}_block_{j}",
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
      file_hash = WEIGHTS_HASHES[model_name[-4:]][0]
    else:
      file_suffix = "_notop.h5"
      file_hash = WEIGHTS_HASHES[model_name[-4:]][1]
    file_name = model_name + file_suffix
    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir="models",
        file_hash=file_hash)
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


## Instantiating variants ##


## TODO

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
