def get_value(x):
    return x.data.numpy()


def batch_get_value(x):
    return list(map(get_value, x))


def set_value(x, value):
    tensor = Tensor(value).type(value.dtype)
    return Variable(tensor)


def batch_set_value(tuples):
    return list(map(lambda x, value: set_value(x, value), tuples))


def get_variable_shape(x):
    return tuple(map(int, x.size()))


def print_tensor(x, message=''):
    print(x)
