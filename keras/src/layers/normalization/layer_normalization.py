import warnings

from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.LayerNormalization")
class LayerNormalization(Layer):
    """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

    If `scale` or `center` are enabled, the layer will scale the normalized
    outputs by broadcasting them with a trainable variable `gamma`, and center
    the outputs by broadcasting with a trainable variable `beta`. `gamma` will
    default to a ones tensor and `beta` will default to a zeros tensor, so that
    centering and scaling are no-ops before training has begun.

    So, with scaling and centering enabled the normalization equations
    are as follows:

    Let the intermediate activations for a mini-batch to be the `inputs`.

    For each sample `x_i` in `inputs` with `k` features, we compute the mean and
    variance of the sample:

    ```python
    mean_i = sum(x_i[j] for j in range(k)) / k
    var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
    ```

    and then compute a normalized `x_i_normalized`, including a small factor
    `epsilon` for numerical stability.

    ```python
    x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
    ```

    And finally `x_i_normalized ` is linearly transformed by `gamma` and `beta`,
    which are learned parameters:

    ```python
    output_i = x_i_normalized * gamma + beta
    ```

    `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
    this part of the inputs' shape must be fully defined.

    For example:

    >>> layer = keras.layers.LayerNormalization(axis=[1, 2, 3])
    >>> layer.build([5, 20, 30, 40])
    >>> print(layer.beta.shape)
    (20, 30, 40)
    >>> print(layer.gamma.shape)
    (20, 30, 40)

    Note that other implementations of layer normalization may choose to define
    `gamma` and `beta` over a separate set of axes from the axes being
    normalized across. For example, Group Normalization
    ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
    corresponds to a Layer Normalization that normalizes across height, width,
    and channel and has `gamma` and `beta` span only the channel dimension.
    So, this Layer Normalization implementation will not match a Group
    Normalization layer with group size set to 1.

    Args:
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. `-1` is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.
        center: If True, add offset of `beta` to normalized tensor. If False,
            `beta` is ignored. Defaults to `True`.
        scale: If True, multiply by `gamma`. If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight.
            None by default.
        gamma_regularizer: Optional regularizer for the gamma weight.
            None by default.
        beta_constraint: Optional constraint for the beta weight.
            None by default.
        gamma_constraint: Optional constraint for the gamma weight.
            None by default.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).


    Reference:

    - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
    """

    def __init__(
        self,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        rms_scaling = kwargs.pop("rms_scaling", False)
        if rms_scaling:
            warnings.warn(
                "You passed `rms_scaling=True`, which is deprecated. This "
                "argument incorrectly scales the input by the variance, not "
                "the root mean square. To correctly use RMS Normalization, "
                "please use `keras.layers.RMSNormalization` instead."
            )

        super().__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.rms_scaling = rms_scaling
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.supports_masking = True
        self.autocast = False

    def build(self, input_shape):
        if isinstance(self.axis, list):
            shape = tuple([input_shape[dim] for dim in self.axis])
        else:
            shape = (input_shape[self.axis],)
            self.axis = [self.axis]
        if self.scale or self.rms_scaling:
            self.gamma = self.add_weight(
                name="gamma",
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                autocast=False,
            )
        else:
            self.gamma = None

        if self.center and not self.rms_scaling:
            self.beta = self.add_weight(
                name="beta",
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                autocast=False,
            )
        else:
            self.beta = None

    def call(self, inputs):
        outputs = ops.layer_normalization(
            inputs,
            self.gamma,
            self.beta,
            self.axis,
            self.epsilon,
            rms_scaling=self.rms_scaling,
        )
        return ops.cast(outputs, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = self.axis

        for axis in axes:
            if axis >= len(input_shape) or axis < -len(input_shape):
                raise ValueError(
                    f"Axis {axis} is out of bounds for "
                    f"input shape {input_shape}. "
                    f"Received: axis={self.axis}"
                )
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "rms_scaling": self.rms_scaling,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
