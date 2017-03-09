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


def convert_legacy_kwargs(layer_name,
                          args,
                          kwargs,
                          conversions,
                          converted=None):
    converted = converted or []
    for old_arg, new_arg in conversions:
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise_duplicate_arg_error(old_arg, new_arg)
            arg_value = kwargs.pop(old_arg)
            kwargs[new_arg] = arg_value
            converted.append((new_arg, arg_value))

    if converted:
        signature = '`' + layer_name + '('
        for value in args:
            if isinstance(value, six.string_types):
                signature += '"' + value + '"'
            else:
                signature += str(value)
            signature += ', '
        for i, (name, value) in enumerate(converted):
            signature += name + '='
            if isinstance(value, six.string_types):
                signature += '"' + value + '"'
            else:
                signature += str(value)
            if i < len(converted) - 1:
                signature += ', '
        signature += ')`'
        warnings.warn('Update your `' + layer_name + '` layer call '
                      'to the Keras 2 API: ' + signature)
    return kwargs


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

        # Remaining kwargs.
        conversions = [
            ('init', 'kernel_initializer'),
            ('W_regularizer', 'kernel_regularizer'),
            ('b_regularizer', 'bias_regularizer'),
            ('W_constraint', 'kernel_constraint'),
            ('b_constraint', 'bias_constraint'),
            ('bias', 'use_bias'),
        ]
        kwargs = convert_legacy_kwargs('Dense',
                                       args[1:],
                                       kwargs,
                                       conversions)
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
            signature = '`Dropout(' + str(args[1])
            for kwarg in kwargs:
                signature += ', ' + kwarg + '='
                if isinstance(kwargs[kwarg], six.string_types):
                    signature += ('"' + kwargs[kwarg] + '"')
                else:
                    signature += str(kwargs[kwarg])
            signature += ')`'
            warnings.warn('Update your `Dropout` layer call '
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

        kwargs = convert_legacy_kwargs('Embedding', args[1:], kwargs, conversions, converted)
        return func(*args, **kwargs)
    return wrapper


def legacy_pooling1d_support(func):
    """Function wrapper to convert `MaxPooling1D` or `AvgPooling1D` constructor from Keras 1 to 2.

    # Arguments
        func: `__init__` method of `MaxPooling1D` or `AvgPooling1D`.

    # Returns
        A constructor conversion wrapper.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 2:
            # The first entry in `args` is `self`.
            raise TypeError(args[0].__name__ + ' layer can have at most '
                            'one positional argument (the `pool_size` argument).')

        # make sure that only keyword argument 'pool_size'(or pool_length' in the legacy interface)
        # can be also used as positional argument, which is keyword argument originally.
        if 'pool_length' in kwargs:
            if len(args) > 1:
                raise TypeError('Got both a positional argument '
                                'and keyword argument for argument '
                                '`pool_size` '
                                '(`pool_length` in the legacy interface).')

        elif 'pool_size' in kwargs:
            if len(args) > 1:
                raise TypeError('Got both a positional argument '
                                'and keyword argument for argument '
                                '`pool_size`. ')

        # Remaining kwargs.
        conversions = [
            ('pool_length', 'pool_size'),
            ('border_mode', 'padding'),
        ]
        kwargs = convert_legacy_kwargs(args[0].__name__,
                                       args[1:],
                                       kwargs,
                                       conversions)
        return func(*args, **kwargs)
    return wrapper
