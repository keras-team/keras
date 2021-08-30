"""Keras kervolutional layers and image transformation layers with kernel function."""

import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import deserialize

from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

class Kervo(Layer):
  """
  This layer creates a convolution kernel with a kernel function that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).

  Args:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution). Could be "None", eg in the case of
      depth wise convolution.
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    kernel_function: A layer takes the columnized feature and the kernel as its inputs.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros
      evenly to the left/right or up/down of the input such that output has the
      same height/width dimension as the input. `"causal"` results in causal
      (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved
      separately with `filters / groups` filters. The output is the
      concatenation of all the `groups` results along the channel axis.
      Input channels and `filters` must both be divisible by `groups`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel. If None, the
      default initializer (glorot_uniform) will be used.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer (zeros) will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               kernel_function,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               conv_op=None,
               **kwargs):
    super(Kervo, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
      filters = int(filters)
    if filters is not None and filters < 0:
      raise ValueError(f'Received a negative value for `filters`.'
                       f'Was expecting a positive value. Received {filters}.')
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.kernel_function = kernel_function
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

  def _validate_init(self):
    if self.filters is not None and self.filters % self.groups != 0:
      raise ValueError(
          'The number of filters must be evenly divisible by the number of '
          'groups. Received: groups={}, filters={}'.format(
              self.groups, self.filters))

    if not all(self.kernel_size):
      raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

    if not all(self.strides):
      raise ValueError('The argument `strides` cannot contains 0(s). '
                       'Received: %s' % (self.strides,))

    if (self.padding == 'causal' and not isinstance(self,
                                                    (Kerv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and `SeparableConv1D`.')

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))
    kernel_shape = self.kernel_size + (input_channel // self.groups,
                                       self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                axes={channel_axis: input_channel})
    self.kernel_function.build([input_shape, kernel_shape])
    self.built = True

  def convolution_op(self, inputs, kernel):
    if self.padding == 'causal':
      tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, str):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding

    return tf.nn.convolution(
        inputs,
        kernel,
        strides=list(self.strides),
        padding=tf_padding,
        dilations=list(self.dilation_rate),
        data_format=self._tf_data_format,
        name=self.__class__.__name__)

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
        inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
    kernel = K.reshape(self.kernel, (-1, self.filters))
    outputs = self.kernel_function([inputs, kernel])

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _spatial_output_shape(self, spatial_input_shape):
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    batch_rank = len(input_shape) - self.rank - 1
    if self.data_format == 'channels_last':
      return tf.TensorShape(
          input_shape[:batch_rank]
          + self._spatial_output_shape(input_shape[batch_rank:-1])
          + [self.filters])
    else:
      return tf.TensorShape(
          input_shape[:batch_rank] + [self.filters] +
          self._spatial_output_shape(input_shape[batch_rank + 1:]))

  def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
    return False

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'kernel_function': {
                'class_name': self.kernel_function.__class__.__name__,
                'config': self.kernel_function.get_config(),
            },
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'dilation_rate':
            self.dilation_rate,
        'groups':
            self.groups,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(Kervo, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


  def from_config(cls, config, custom_objects=None):
        config['kernel_function'] = deserialize(
            config.pop('kernel_function'),
            custom_objects=custom_objects,
        )
        return cls(**config)

  def _compute_causal_padding(self, inputs):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if getattr(inputs.shape, 'ndims', None) is None:
      batch_rank = 1
    else:
      batch_rank = len(inputs.shape) - 2
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return -1 - self.rank
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs should be defined. '
                       f'The input_shape received is {input_shape}, '
                       f'where axis {channel_axis} (0-based) '
                       'is the channel dimension, which found to be `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding


@keras_export('keras.layers.Kerv1D', 'keras.layers.Kervolution1D')
class Kerv1D(Kervo):
  """1D Kervolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel with a kernel function that is convolved
  with the layer input over a single spatial (or temporal) dimension
  to produce a tensor of outputs.
  If `use_bias` is True, a bias vector is created and added to the outputs.
  Finally, if `activation` is not `None`,
  it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide an `input_shape` argument
  (tuple of integers or `None`, e.g.
  `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
  or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

  Examples:

  >>> # The inputs are 128-length vectors with 10 timesteps, and the batch size
  >>> # is 4.
  >>> input_shape = (4, 10, 128)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv1D(
  ... 32, 3, activation='relu',input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 8, 32)

  >>> # With extended batch shape [4, 7] (e.g. weather data where batch
  >>> # dimensions correspond to spatial location and the third dimension
  >>> # corresponds to time.)
  >>> input_shape = (4, 7, 10, 128)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv1D(
  ... 32, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 8, 32)

  Args:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer,
      specifying the length of the 1D convolution window.
    kernel_function: A layer takes the columnized feature and the kernel as its inputs.  
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`, `"same"` or `"causal"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input such that output has the same
      height/width dimension as the input.
      `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
      does not depend on `input[t+1:]`. Useful when modeling temporal data
      where the model should not violate the temporal order.
      See [WaveNet: A Generative Model for Raw Audio, section
        2.1](https://arxiv.org/abs/1609.03499).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
    dilation_rate: an integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved
      separately with `filters / groups` filters. The output is the
      concatenation of all the `groups` results along the channel axis.
      Input channels and `filters` must both be divisible by `groups`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
      (see `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector
      (see `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector
      (see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")
      (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix
      (see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector
      (see `keras.constraints`).

  Input shape:
    3+D tensor with shape: `batch_shape + (steps, input_dim)`

  Output shape:
    3+D tensor with shape: `batch_shape + (new_steps, filters)`
      `steps` value might have changed due to padding or strides.

  Returns:
    A tensor of rank 3 representing
    `activation(conv1d(inputs, kernel) + bias)`.

  Raises:
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

  def __init__(self,
               filters,
               kernel_size,
               kernel_function,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Kerv1D, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        kernel_function=kernel_function,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


@keras_export('keras.layers.Kervo2D', 'keras.layers.Kervolution2D')
class Kervo2D(Kervo):
  """2D convolution layer with kernel function (e.g. spatial convolution over images).

  This layer creates a convolution kernel with kernel function that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers or `None`, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`. You can use `None` when
  a dimension has variable size.

  Examples:

  >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
  >>> # size is 4.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 26, 26, 2)

  >>> # With `dilation_rate` as 2.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 24, 24, 2)

  >>> # With `padding` as "same".
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
  >>> print(y.shape)
  (4, 28, 28, 2)

  >>> # With extended batch shape [4, 7]:
  >>> input_shape = (4, 7, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
  >>> print(y.shape)
  (4, 7, 26, 26, 2)


  Args:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    kernel_function: A layer takes the columnized feature and the kernel as its inputs.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
      `"valid"` means no padding. `"same"` results in padding with zeros evenly
      to the left/right or up/down of the input. When `padding="same"` and
      `strides=1`, the output has the same size as the input.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch_size, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch_size, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be `channels_last`.
    dilation_rate: an integer or tuple/list of 2 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    groups: A positive integer specifying the number of groups in which the
      input is split along the channel axis. Each group is convolved separately
      with `filters / groups` filters. The output is the concatenation of all
      the `groups` results along the channel axis. Input channels and `filters`
      must both be divisible by `groups`.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (see
      `keras.initializers`). Defaults to 'glorot_uniform'.
    bias_initializer: Initializer for the bias vector (see
      `keras.initializers`). Defaults to 'zeros'.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (see
      `keras.regularizers`).
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (see
      `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (see
      `keras.constraints`).

  Input shape:
    4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
      `data_format='channels_first'`
    or 4+D tensor with shape: `batch_shape + (rows, cols, channels)` if
      `data_format='channels_last'`.

  Output shape:
    4+D tensor with shape: `batch_shape + (filters, new_rows, new_cols)` if
    `data_format='channels_first'` or 4+D tensor with shape: `batch_shape +
      (new_rows, new_cols, filters)` if `data_format='channels_last'`.  `rows`
      and `cols` values might have changed due to padding.

  Returns:
    A tensor of rank 4+ representing
    `activation(conv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is `"causal"`.
    ValueError: when both `strides > 1` and `dilation_rate > 1`.
  """

  def __init__(self,
               filters,
               kernel_size,
               kernel_function,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Kervo2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        kernel_function=kernel_function,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)