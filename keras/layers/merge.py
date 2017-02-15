from ..engine.topology import Layer
from .. import backend as K


class _Merge(Layer):
    """TODO
    """

    def __init__(self, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.supports_masking = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        # TODO: handle shapes with None entries.
        input_shapes_set = set(input_shape)
        if len(input_shapes_set) > 1:
            raise ValueError('Only tensors of same shape can '
                             'be merged by layer' + self.name +
                             ' Got input shapes: %s' % input_shape)

    def call(self, inputs):
        return self._merge_function(inputs)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError('`mask` should be a list.')
            if not isinstance(inputs, list):
                raise ValueError('`inputs` should be a list.')
            if len(mask) != len(inputs):
                raise ValueError('The lists `inputs` and `mask` '
                                 'should have the same length.')
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


class Sum(_Merge):
    """TODO
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output


class Multiply(_Merge):
    """TODO
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output *= inputs[i]
        return output


class Average(_Merge):
    """TODO
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output *= inputs[i]
        return output / len(inputs)


class Maximum(_Merge):
    """TODO
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = K.maximum(output, inputs[i])
        return output


class Concatenate(_Merge):
    """TODO
    """

    def __init__(self, axis=-1, **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('`Concatenate` layer should be called '
                             'on a list of 2 inputs.')
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('`Concatenate` layer requires '
                             'inputs with matching shapes '
                             'except for the concat axis. '
                             'Got inputs shapes: %s' % (input_shape))

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of 2 inputs.')
        return K.concatenate(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of 2 inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(inputs, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of 2 inputs.')
        if not isinstance(mask, list):
            raise ValueError('TODO')
        if len(inputs) != len(mask):
            raise ValueError('TODO')
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                # but cast it to uint8 first
                masks.append(K.cast(K.ones_like(input_i), 'uint8'))
            elif K.ndim(mask_i) < K.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(K.expand_dims(mask_i))
            else:
                masks.append(mask_i)
        concatenated = K.concatenate(masks, axis=self.axis)
        return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Dot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dot(_Merge):
    """TODO
    """

    def __init__(self, axes, normalize=False, **kwargs):
        super(Dot, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError('Invalid type for `axes` - '
                                'should be a list or an int.')
            if len(axes) != 2:
                raise ValueError('Invalid format for `axes` - '
                                 'should contain two elements.')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError('Invalid format for `axes` - '
                                 'list elements should be "int".')
        self.axes = axes
        self.normalize = normalize
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                'Layer shapes: %s, %s' % (shape1, shape2))

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=axes[0])
            x2 = K.l2_normalize(x2, axis=axes[1])
        output = K.batch_dot(x1, x2, axes)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        shape1.pop(self.axes[0])
        shape2.pop(self.axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'axes': self.axes,
            'normalize': self.normalize,
        }
        base_config = super(Dot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sum(inputs):
    """TODO
    """
    return Sum()(inputs)


def multiply(inputs):
    """TODO
    """
    return Multiply()(inputs)


def average(inputs):
    """TODO
    """
    return Average()(inputs)


def maximum(inputs):
    """TODO
    """
    return Maximum()(inputs)


def concatenate(inputs, axis=-1):
    """TODO
    """
    return Concatenate(axis=axis)(inputs)


def dot(inputs, axes, normalize=False):
    """TODO
    """
    return Dot(axes=axes, normalize=normalize)(inputs)
