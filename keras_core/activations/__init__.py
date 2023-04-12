def identity(x):
    return x


def get(identifier):
    if identifier is None:
        return identity
    return identifier
