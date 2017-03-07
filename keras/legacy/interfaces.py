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


def legacy_embedding_support(func):
    """Function wrapper to convert the `Embedding` constructor from Keras 1 to 2.

    # Arguments
        func: `__init__` method of `Embedding`.

    # Returns
        A constructor conversion wrapper.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 3:
            # The first entry in `args` is `self`.
            raise TypeError('The `Embedding` layer can have at most '
                            'two positional argument (`input_dim` and `output_dim`).')

        # input_dim and output_dim
        input_dim = kwargs.pop('input_dim', None)
        if input_dim is None:
            input_dim = args[1]

        output_dim = kwargs.pop('output_dim', None)
        if output_dim is None:
            output_dim = args[2]

        if 'dropout' in kwargs:
            del kwargs['dropout']
            warnings.warn('The `dropout` argument is no longer exists. '
                          'Please use keras.layers.SpatialDropout1D layer '
                          'right after the Embedding layer to get the same behavior.')

        args = (args[0], input_dim, output_dim)

        converted = []

        # Remaining kwargs.
        conversions = [
            ('init', 'embeddings_initializer'),
            ('W_regularizer', 'embeddings_regularizer'),
            ('W_constraint', 'embeddings_constraint'),
        ]

        for old_arg, new_arg in conversions:
            if old_arg in kwargs:
                if new_arg in kwargs:
                    raise_duplicate_arg_error(old_arg, new_arg)
                arg_value = kwargs.pop(old_arg)
                kwargs[new_arg] = arg_value
                converted.append((new_arg, arg_value))

        if converted:
            signature = '`Embedding(' + str(input_dim) + ', ' + str(output_dim)

            for name, value in converted:
                signature += ', ' + name + '='
                if isinstance(value, six.string_types):
                    signature += ('"' + value + '"')
                else:
                    signature += str(value)
            signature += ')`'
            warnings.warn('Update your `Embedding` layer call '
                          'to the Keras 2 API: ' + signature)

        return func(*args, **kwargs)
    return wrapper
