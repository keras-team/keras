"""Interface converters for Keras 1 support in Keras 2.
"""
import six
import warnings


def generate_legacy_interface(allowed_positional_args=None,
                              conversions=None):
    allowed_positional_args = allowed_positional_args or []
    conversions = conversions or []

    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            layer_name = args[0].__class__.__name__
            converted = []
            if len(args) > len(allowed_positional_args) + 1:
                raise TypeError('Layer `' + layer_name +
                                '` can accept only ' +
                                str(len(allowed_positional_args)) +
                                ' positional arguments (' +
                                str(allowed_positional_args) + '), but '
                                'you passed the following '
                                'positional arguments: ' +
                                str(args[1:]))
            for old_name, new_name in conversions:
                if old_name in kwargs:
                    value = kwargs.pop(old_name)
                    if new_name in kwargs:
                        raise_duplicate_arg_error(old_name, new_name)
                    kwargs[new_name] = value
                    converted.append((new_name, old_name))
            if converted:
                signature = '`' + layer_name + '('
                for value in args[1:]:
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        signature += str(value)
                    signature += ', '
                for i, (name, value) in enumerate(kwargs.items()):
                    signature += name + '='
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        signature += str(value)
                    if i < len(kwargs) - 1:
                        signature += ', '
                signature += ')`'
                warnings.warn('Update your `' + layer_name +
                              '` layer call to the Keras 2 API: ' + signature)
            return func(*args, **kwargs)
        return wrapper
    return legacy_support


def raise_duplicate_arg_error(old_arg, new_arg):
    raise TypeError('For the `' + new_arg + '` argument, '
                    'the layer received both '
                    'the legacy keyword argument '
                    '`' + old_arg + '` and the Keras 2 keyword argument '
                    '`' + new_arg + '`. Stick to the latter!')

legacy_dense_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')])

legacy_dropout_support = generate_legacy_interface(
    allowed_positional_args=['rate', 'noise_shape', 'seed'],
    conversions=[('p', 'rate')])

legacy_pooling1d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('pool_length', 'pool_size'),
                 ('stride', 'strides'),
                 ('border_mode', 'padding')])
