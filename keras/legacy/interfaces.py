"""Interface converters for Keras 1 support in Keras 2.
"""
import six
import warnings


def raise_duplicate_arg_error(old_arg, new_arg):
    raise TypeError('For the `' + new_arg + '` argument, '
                    'the layer received both '
                    'the legacy keyword argument '
                    '`' + old_arg + '` and the Keras 2 keyword argument '
                    '`' + new_arg + '`. Stick with the latter!')


def legacy_dense_support(func):
    """Function wrapper to convert the `Dense` constructor from Keras 1 to 2.

    # Arguments
        func: `__init__` method of `Dense`.

    # Returns
        A constructor conversion wrapper.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 2:
            # The first entry in `args` is `self`.
            raise TypeError('The `Dense` layer can have at most '
                            'one positional argument (the `units` argument).')

        # output_dim
        if 'output_dim' in kwargs:
            if len(args) > 1:
                raise TypeError('Got both a positional argument '
                                'and keyword argument for argument '
                                '`units` '
                                '(`output_dim` in the legacy interface).')
            if 'units' in kwargs:
                raise_duplicate_arg_error('output_dim', 'units')
            output_dim = kwargs.pop('output_dim')
            args = (args[0], output_dim)

        converted = []

        # Remaining kwargs.
        conversions = [
            ('init', 'kernel_initializer'),
            ('W_regularizer', 'kernel_regularizer'),
            ('b_regularizer', 'bias_regularizer'),
            ('W_constraint', 'kernel_constraint'),
            ('b_constraint', 'bias_constraint'),
            ('bias', 'use_bias'),
        ]

        for old_arg, new_arg in conversions:
            if old_arg in kwargs:
                if new_arg in kwargs:
                    raise_duplicate_arg_error(old_arg, new_arg)
                arg_value = kwargs.pop(old_arg)
                kwargs[new_arg] = arg_value
                converted.append((new_arg, arg_value))

        if converted:
            signature = '`Dense(' + str(args[1])
            for name, value in converted:
                signature += ', ' + name + '='
                if isinstance(value, six.string_types):
                    signature += ('"' + value + '"')
                else:
                    signature += str(value)
            signature += ')`'
            warnings.warn('Update your `Dense` layer call '
                          'to the Keras 2 API: ' + signature)

        return func(*args, **kwargs)
    return wrapper


def legacy_dropout_support(func):
    """Function wrapper to convert the `Dropout` constructor from Keras 1 to 2.

    # Arguments
        func: `__init__` method of `Dropout`.

    # Returns
        A constructor conversion wrapper.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 2:
            # The first entry in `args` is `self`.
            raise TypeError('The `Dropout` layer can have at most '
                            'one positional argument (the `rate` argument).')

        # Convert `p` to `rate` if keyword arguement is used
        if 'p' in kwargs:
            if len(args) > 1:
                raise TypeError('Got both a positional argument '
                                'and keyword argument for argument '
                                '`rate` '
                                '(`p` in the legacy interface).')
            if 'rate' in kwargs:
                raise_duplicate_arg_error('p', 'rate')
            rate = kwargs.pop('p')
            args = (args[0], rate)
            signature = '`Dropout(rate=`' + str(args[1]) + ')`'
            warnings.warn('Update your `Dropout` layer call to Keras 2 API.' + signature)

        return func(*args, **kwargs)
    return wrapper
