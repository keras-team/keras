from ..layers import Lambda
from ..engine.topology import merge, Merge
from .. import backend as K


def _slice_shape(shape, slices):
    '''Infer the shape of a tensor when it is sliced / indexed using the __getitem__ operator.
    
    # Arguments:
    shape: tuple. shape of the tensor
    slices: slice, int or tuple of slice and int used to slice or index the tensor.
    '''
    output_shape = []
    if type(slices) not in [list, tuple]:
        slices = [slices]
    while len(shape) > len(slices):
        slices += (slice(None),)
    for i in range(len(slices)):
        s = slices[i]
        if type(s) is not slice:
            continue
        else:
            start = s.start
            stop = s.stop
            step = s.step
            if not start:
                start = 0
            if not step:
                step = 1
            if shape[i] is None:
                if stop >= 0 and start >= 0:
                    output_shape.append((stop - start) / step)
                else:
                    output_shape.append(None)
            else:
                if start < 0:
                    start = shape[i] + start
                if not stop:
                    stop = shape[i]
                elif stop < 0:
                    stop = shape[i] + stop
                output_shape.append((stop - start) / step)
    for i in range(len(output_shape)):
        if output_shape[i] and output_shape[i] < 0:
            output_shape[i] = 0
    return tuple(output_shape)

def _compatible(shape1, shape2):
    '''check if shapes of 2 tensors are compatible for element wise operations.
    '''
    if len(shape1) != len(shape2):
        return False
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 is not None and dim2 is not None and dim1 != dim2:
            return False
    return True

def _override_operator(tensor_class, operator):
    ''' Override an operator of a class
    
    # Arguments:
    tensor_class: class for which the operator is to be overridden.
    operator: the operator to be overridden.
    '''
    _operator = '_keras' + operator[1:-2]  # we add '_keras' so that it does not conflict with any existing functions
    setattr(tensor_class, _operator, getattr(tensor_class, operator))
    unary_operators = ['__neg__', '__pos__', '__abs__', '__invert__']
    binary_operators = ['add', 'sub', 'mul', 'matmul', 'div', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or']
    binary_operators += map(lambda x: 'r' + x, binary_operators)
    binary_operators += ['getitem']
    binary_operators = map(lambda x: '__' + x + '__', binary_operators)
    if operator in unary_operators:
        def op(x):
            x_k = hasattr(x, '_keras_history')
            if not x_k:
                return getattr(x, _operator)()
            else:
                def func(x):
                    return getattr(x, _operator)()
                lambda_layer = Lambda(func, output_shape=lambda x: x)
                lambda_layer.build(None)
                lambda_layer.supports_masking = True
                return lambda_layer(x)
    elif operator in binary_operators:
        def op(x, y):
            x_k = hasattr(x, '_keras_history')
            y_k = hasattr(y, '_keras_history')
            if not x_k and not y_k:
                res = getattr(x, _operator)(y)
                override_operators(res.__class__)
                return res 
            elif x_k and not y_k:
                _merge = False
                _op_tensor = False
                previous_layer = x._keras_history[0]
                if hasattr(previous_layer, '_op_layer'):
                    _op_tensor = True
                    if previous_layer.__class__ == Merge:
                        _merge = True
                    def func(x):
                        x = previous_layer.call(x)
                        return getattr(x, _operator)(y)
                else:
                        def func(x):
                            return getattr(x, _operator)(y)
                if operator == '__getitem__':
                    output_shape = lambda _: _slice_shape(x._keras_shape, y)
                else:
                    if _merge:
                        output_shape = lambda x: x[0]
                    else:
                        output_shape = lambda x: x
                if _op_tensor:
                    x = previous_layer.input
                if _merge:
                    res = merge(x, mode=func, output_shape=output_shape)
                else:
                    lambda_layer = Lambda(func, output_shape=output_shape)
                    lambda_layer.build(None)
                    lambda_layer.supports_masking = True
                    res = lambda_layer(x)
            elif not x_k and y_k:
                _merge = False
                _op_tensor = False
                previous_layer = y._keras_history[0]
                if hasattr(previous_layer, '_op_layer'):
                    _op_tensor = True
                    if previous_layer.__class__ == Merge:
                        _merge = True
                    def func(y):
                        y = previous_layer.call(y)
                        return getattr(x, _operator)(y)
                else:
                        def func(y):
                            return getattr(x, _operator)(y)
                if _op_tensor:
                    y = previous_layer.input
                if _merge:
                    res = merge(y, mode=func, output_shape=lambda s: previous_layer.output_shape)
                else:
                    lambda_layer = Lambda(func, output_shape=lambda x: x)
                    lambda_layer.build(None)
                    lambda_layer.supports_masking = True
                    res = lambda_layer(y)
            else:
                shape1 = x._keras_shape
                shape2 = y._keras_shape
                assert _compatible(shape1, shape2), 'Incompatible shapes : ' + str(shape1) + ' and ' + str(shape2) + '.'
                _left_merge = False
                _right_merge = False
                previous_layer = x._keras_history[0]
                if hasattr(previous_layer, '_op_layer'):
                    x_func = previous_layer.call
                    if previous_layer.__class__ == Merge:
                        _left_merge = True
                else:
                    x_func = lambda x: x
                previous_layer = y._keras_history[0]
                if hasattr(previous_layer, '_op_layer'):
                    y_func = previous_layer.call
                    if previous_layer.__class__ == Merge:
                        _right_merge = True
                else:
                    y_func = lambda x: x
                def func(X):
                    if _left_merge:
                        x = X[:2]
                    else:
                        x = X[0]
                    if _right_merge:
                        y = X[:-2]
                    else:
                        y = X[1]
                    x = x_func(x)
                    y = y_func(y)
                    return getattr(x, _operator)(y)
                inputs = []
                if _left_merge:
                    inputs += x._keras_history[0].input
                else:
                    inputs += [x]
                if _right_merge:
                    inputs += y._keras_history[0].input
                else:
                    inputs += [y]  
                res = merge(inputs, mode=func, output_shape=lambda _: shape1)
            setattr(res._keras_history[0], '_op_layer', True)
            override_operators(res.__class__)  # In some cases the resultant tensor might belong to a different class than the operands.
            return res
    else:
        raise Exception('Invalid operator: ' + operator)
    setattr(tensor_class, operator, op)

def override_operators(tensor_class):
    '''Override operators of a class

    # Arguments:
    tensor_class: class whose operators has to be overridden.
    '''
    if hasattr(tensor_class, '_keras_operators_supported'):
        return
    else:
        setattr(tensor_class, '_keras_operators_supported', True)
    operators = ['add', 'sub', 'mul', 'div', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or']
    operators += map(lambda x: 'r' + x, operators)
    operators += ['neg', 'pos', 'abs', 'invert']
    operators += ['getitem']
    operators = map(lambda x: '__' + x + '__', operators)
    for op in operators:
        if hasattr(tensor_class, op):
            _override_operator(tensor_class, op)
