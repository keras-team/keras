from keras_core import backend


def relu(x):
    return backend.nn.relu(x)


def identity(x):
    return x


def get(identifier):
    if identifier is None:
        return identity
    if identifier == "relu":
        return relu
    return identifier


def serialize(activation):
    return activation.__name__
