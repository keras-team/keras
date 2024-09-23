def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    raise NotImplementedError(
        "`normal` is not supported with openvino backend"
    )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    raise NotImplementedError(
        "`uniform` is not supported with openvino backend"
    )


def categorical(logits, num_samples, dtype="int64", seed=None):
    raise NotImplementedError(
        "`categorical` is not supported with openvino backend"
    )


def randint(shape, minval, maxval, dtype="int32", seed=None):
    raise NotImplementedError(
        "`randint` is not supported with openvino backend"
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    raise NotImplementedError(
        "`truncated_normal` is not supported with openvino backend"
    )


def dropout(inputs, rate, noise_shape=None, seed=None):
    raise NotImplementedError(
        "`dropout` is not supported with openvino backend"
    )


def shuffle(x, axis=0, seed=None):
    raise NotImplementedError(
        "`shuffle` is not supported with openvino backend"
    )


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError(
        "`gamma` is not supported with openvino backend"
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "`binomial` is not supported with openvino backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError(
        "`beta` is not supported with openvino backend"
    )
