from ..layers import Lambda
from ..engine.topology import merge
from .. import backend as K


def _slice_shape(shape, slices):
    '''slice a shape
    '''
    output_shape = []
    if type(slices) in [int, slice]:
        slices = [slices]
    while len(shape) > len(slices):
        slices += (slice(None),)
    for i in range(len(slices)):
        s = slices[i]
        if type(s) == int:
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
    '''check if shapes of 2 tensors are compatible for addition, multiplication etc.
    '''
    if len(shape1) != len(shape2):
        return False
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 and dim2 and dim1 != dim2:
            return False
    return True

def _override_operator(tensor_class, operator):
    _operator = '_keras' + operator[1:-2]  # we add '_keras' so that it does not conflict with any existing functions
    setattr(tensor_class, _operator, getattr(tensor_class, operator))
    def op(x, y):
        x_k = hasattr(x, '_keras_history')
        y_k = hasattr(y, '_keras_history')
        if not x_k and not y_k :
            res = getattr(x, _operator)(y)
        elif x_k and not y_k:
            def func(x):
                return getattr(x, _operator)(y)
            if operator == '__getitem__':
                output_shape = lambda _ : _slice_shape(x._keras_shape, y)
            else:
                output_shape = lambda x : x
            lambda_layer = Lambda(func, output_shape=output_shape)
            lambda_layer.build(None)
            res = lambda_layer(x)
        elif not x_k and y_k:
            def func(y):
                return getattr(x, _operator)(y)
            lambda_layer = Lambda(func, output_shape=lambda x : x)
            lambda_layer.build(None)
            res = lambda_layer(y)
        else:
            shape1 = x._keras_shape
            shape2 = y._keras_shape
            assert _compatible(shape1, shape2), 'Incompatible shapes : ' + str(shape1) + ' and ' + str(shape2) + '.'
            def func(X):
                x = X[0]
                y = X[1]
                return getattr(x, _operator)(y)
            res = merge([x, y], mode=func, output_shape=lambda _ : shape1)
        override_operators(res.__class__)
        return res
    setattr(tensor_class, operator, op)

def override_operators(tensor_class):
    '''override operators of a tensor class so that topology information is preserved
    '''
    if hasattr(tensor_class, 'operators_overridden'):
        return
    else:
        setattr(tensor_class, 'operators_overridden', True)
    operators = ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or']
    operators += map(lambda x : 'r' + x, operators)
    operators += ['neg', 'pos', 'abs', 'invert']
    operators += ['getitem']
    operators = map(lambda x : '__' + x + '__', operators)
    for op in operators:
        if hasattr(tensor_class, op):
            _override_operator(tensor_class, op)
