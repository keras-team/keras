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


def legacy_convert(layer_name, conversions):
    """
    A function that generates legacy API converters for functions

    # Arguments
    layer_name : Name of the function to be converted
    conversions : Dict where values are tuples of (legacy_keywords, new_keyword)
                  and keys are corresponding positions
    """
    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(1, len(args)):
                '''Checks if both positional arguement and
                   correspoding keyword argument is passed
                '''
                if i in conversions.keys():
                    if conversions[i][0] in kwargs or conversions[i][1] in kwargs:
                        raise TypeError('Got both a positional arguement '
                                        'and keyword argument for arguement `' +
                                        conversions[i][1] +
                                        '` (`' + conversions[i][0] + '` in the'
                                        'legacy interface).')

            kwargs = convert_legacy_kwargs(layer_name, args[1:], kwargs, conversions.values())
            return func(*args, **kwargs)
        return wrapper
    return legacy_support

legacy_dense_support = legacy_convert('Dense',
                                      {1: ('output_dim', 'units'),
                                       2: ('init', 'kernel_initializer'),
                                       5: ('W_regularizer', 'kernel_regularizer'),
                                       6: ('b_regularizer', 'bias_regularizer'),
                                       8: ('W_constraint', 'kernel_constraint'),
                                       9: ('b_constraint', 'bias_constraint'),
                                       10: ('bias', 'use_bias')})

legacy_dropout_support = legacy_convert('Dropout',
                                        {1: ('p', 'rate')})


legacy_simplernn_support = legacy_convert('SimpleRNN',
                                          {1: ('output_dim', 'units'),
                                           2: ('init', 'kernel_initializer'),
                                           3: ('inner_init', 'recurrent_initializer'),
                                           5: ('W_regularizer', 'kernel_regularizer'),
                                           6: ('U_regularizer', 'recurrent_regularizer'),
                                           7: ('b_regularizer', 'bias_regularizer'),
                                           9: ('dropout_W', 'dropout'),
                                           10: ('dropout_U', 'recurrent_dropout')})

legacy_gru_support = legacy_convert('GRU',
                                    {1: ('output_dim', 'units'),
                                     2: ('init', 'kernel_initializer'),
                                     3: ('inner_init', 'recurrent_initializer'),
                                     5: ('inner_activation', 'recurrent_activation'),
                                     6: ('W_regularizer', 'kernel_regularizer'),
                                     7: ('U_regularizer', 'recurrent_regularizer'),
                                     8: ('b_regularizer', 'bias_regularizer'),
                                     9: ('dropout_W', 'dropout'),
                                     10: ('dropout_U', 'recurrent_dropout')})

legacy_lstm_support = legacy_convert('LSTM',
                                     {1: ('output_dim', 'units'),
                                      2: ('init', 'kernel_initializer'),
                                      3: ('inner_init', 'recurrent_initializer'),
                                      4: ('forget_bias_init', 'bias_initializer'),
                                      6: ('inner_activation', 'recurrent_activation'),
                                      7: ('W_regularizer', 'kernel_regularizer'),
                                      8: ('U_regularizer', 'recurrent_regularizer'),
                                      9: ('b_regularizer', 'bias_regularizer'),
                                      10: ('dropout_W', 'dropout'),
                                      11: ('dropout_U', 'recurrent_dropout')})


legacy_maxpooling1d_support = legacy_convert('MaxPooling1D',
                                             {1: ('pool_length', 'pool_size'),
                                              2: ('stride', 'strides'),
                                              3: ('border_mode', 'padding')})

legacy_averagepooling1d_support = legacy_convert('AveragePooling1D',
                                                 {1: ('pool_length', 'pool_size'),
                                                  2: ('stride', 'strides'),
                                                  3: ('border_mode', 'padding')})
