"""Interface converters for Keras 1 support in Keras 2.
"""
import six
import warnings


def generate_legacy_interface(allowed_positional_args=None,
                              conversions=None,
                              preprocessor=None,
                              value_conversions=None):
    allowed_positional_args = allowed_positional_args or []
    conversions = conversions or []
    value_conversions = value_conversions or []

    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            layer_name = args[0].__class__.__name__
            if preprocessor:
                args, kwargs, converted = preprocessor(args, kwargs)
            else:
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
            for key in value_conversions:
                if key in kwargs:
                    old_value = kwargs[key]
                    if old_value in value_conversions[key]:
                        kwargs[key] = value_conversions[key][old_value]
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


def embedding_kwargs_preprocessor(args, kwargs):
    converted = []
    if 'dropout' in kwargs:
        kwargs.pop('dropout')
        warnings.warn('The `dropout` argument is no longer support in `Embedding`. '
                      'You can apply a `keras.layers.SpatialDropout1D` layer '
                      'right after the `Embedding` layer to get the same behavior.')
    return args, kwargs, converted

legacy_embedding_support = generate_legacy_interface(
    allowed_positional_args=['input_dim', 'output_dim'],
    conversions=[('init', 'embeddings_initializer'),
                 ('W_regularizer', 'embeddings_regularizer'),
                 ('W_constraint', 'embeddings_constraint')],
    preprocessor=embedding_kwargs_preprocessor)

legacy_pooling1d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('pool_length', 'pool_size'),
                 ('stride', 'strides'),
                 ('border_mode', 'padding')])

legacy_prelu_support = generate_legacy_interface(
    allowed_positional_args=['alpha_initializer'],
    conversions=[('init', 'alpha_initializer')])


legacy_gaussiannoise_support = generate_legacy_interface(
    allowed_positional_args=['stddev'],
    conversions=[('sigma', 'stddev')])


def lstm_args_preprocessor(args, kwargs):
    converted = []
    if 'forget_bias_init' in kwargs:
        if kwargs['forget_bias_init'] == 'one':
            kwargs.pop('forget_bias_init')
            kwargs['unit_forget_bias'] = True
            converted.append(('forget_bias_init', 'unit_forget_bias'))
        else:
            kwargs.pop('forget_bias_init')
            warnings.warn('The `forget_bias_init` argument '
                          'has been ignored. Use `unit_forget_bias=True` '
                          'instead to intialize with ones')
    return args, kwargs, converted

legacy_recurrent_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units'),
                 ('init', 'kernel_initializer'),
                 ('inner_init', 'recurrent_initializer'),
                 ('inner_activation', 'recurrent_activation'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('U_regularizer', 'recurrent_regularizer'),
                 ('dropout_W', 'dropout'),
                 ('dropout_U', 'recurrent_dropout')],
    preprocessor=lstm_args_preprocessor)

legacy_gaussiandropout_support = generate_legacy_interface(
    allowed_positional_args=['rate'],
    conversions=[('p', 'rate')])

legacy_pooling2d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('border_mode', 'padding'),
                 ('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})
