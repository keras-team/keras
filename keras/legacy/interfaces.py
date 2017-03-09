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


def legacy_convert(layer_name, args_convert, kwargs_convert):
    """
    A function that generates legacy API converters for functions 
    with single positional arguement

    # Arguments
    layer_name : Name of the function to be converted

    args_convert : A list/tuple containing names of legacy and new positional 
                   arguements (legacy_arg_name, new_arg_name)`  
    
    kwargs_convert : A list of tuples, each containing names of legacy
                     and new keywords arguement names 
                     [(legacy_arg_name1, new_arg_name1), (legacy_arg_name2, new_arg_name2) ... ]
    """
    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > 2:
                raise TypeError("The `" + layer_name + '` can have at'
                                'most one positional arguement (the `' + args_convert[1] +
                                '` argument)')

            if args_convert[0] in kwargs:
                if len(args) > 1:
                    raise TypeError('Got both a positional arguement '
                                    'and keyword argument for arguement `' +
                                    args_convert[1] +
                                    '` (`' + args_convert[0] + '` in the'
                                    'legacy interface).')
                if args_convert[1] in kwargs:
                    raise_duplicate_arg_error(args_convert[0], args_convert[1])
                
                arg = kwargs.pop(args_convert[0])
                args = (args[0], arg)

            kwargs = convert_legacy_kwargs(layer_name,
                                           args[1:],
                                           kwargs,
                                           kwargs_convert)
            return func(*args, **kwargs)
        return wrapper
    return legacy_support

legacy_dense_support = legacy_convert('Dense',
                                      ('output_dim', 'units'),
                                      [('init', 'kernel_initializer'),
                                       ('W_regularizer', 'kernel_regularizer'),
                                       ('b_regularizer', 'bias_regularizer'),
                                       ('W_constraint', 'kernel_constraint'),
                                       ('b_constraint', 'bias_constraint'),
                                       ('bias', 'use_bias')])

legacy_dropout_support = legacy_convert('Dropout',
                                        ('p', 'rate'),
                                        [('init', 'kernel_initializer'),
                                         ('W_regularizer', 'kernel_regularizer'),
                                         ('b_regularizer', 'bias_regularizer'),
                                         ('W_constraint', 'kernel_constraint'),
                                         ('b_constraint', 'bias_constraint'),
                                         ('bias', 'use_bias')])

legacy_maxpooling1d_support = legacy_convert('MaxPooling1D',
                                             ('pool_length', 'pool_size'),
                                             [('border_mode', 'padding')])
