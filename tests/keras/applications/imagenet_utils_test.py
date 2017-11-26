import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.applications import imagenet_utils as utils
from keras.models import Model
from keras.layers import Input, Lambda


def test_preprocess_input():
    # Test image batch
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    assert utils.preprocess_input(x).shape == x.shape

    out1 = utils.preprocess_input(x, 'channels_last')
    out2 = utils.preprocess_input(np.transpose(x, (0, 3, 1, 2)),
                                  'channels_first')
    assert_allclose(out1, out2.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    assert utils.preprocess_input(x).shape == x.shape

    out1 = utils.preprocess_input(x, 'channels_last')
    out2 = utils.preprocess_input(np.transpose(x, (2, 0, 1)),
                                  'channels_first')
    assert_allclose(out1, out2.transpose(1, 2, 0))


def test_preprocess_input_symbolic():
    # Test image batch
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    inputs = Input(shape=x.shape[1:])
    outputs = Lambda(utils.preprocess_input, output_shape=x.shape[1:])(inputs)
    model = Model(inputs, outputs)
    assert model.predict(x).shape == x.shape

    outputs1 = Lambda(lambda x: utils.preprocess_input(x, 'channels_last'),
                      output_shape=x.shape[1:])(inputs)
    model1 = Model(inputs, outputs1)
    out1 = model1.predict(x)
    x2 = np.transpose(x, (0, 3, 1, 2))
    inputs2 = Input(shape=x2.shape[1:])
    outputs2 = Lambda(lambda x: utils.preprocess_input(x, 'channels_first'),
                      output_shape=x2.shape[1:])(inputs2)
    model2 = Model(inputs2, outputs2)
    out2 = model2.predict(x2)
    assert_allclose(out1, out2.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    inputs = Input(shape=x.shape)
    outputs = Lambda(utils.preprocess_input, output_shape=x.shape)(inputs)
    model = Model(inputs, outputs)
    assert model.predict(x[np.newaxis])[0].shape == x.shape

    outputs1 = Lambda(lambda x: utils.preprocess_input(x, 'channels_last'),
                      output_shape=x.shape)(inputs)
    model1 = Model(inputs, outputs1)
    out1 = model1.predict(x[np.newaxis])[0]
    x2 = np.transpose(x, (2, 0, 1))
    inputs2 = Input(shape=x2.shape)
    outputs2 = Lambda(lambda x: utils.preprocess_input(x, 'channels_first'),
                      output_shape=x2.shape)(inputs2)
    model2 = Model(inputs2, outputs2)
    out2 = model2.predict(x2[np.newaxis])[0]
    assert_allclose(out1, out2.transpose(1, 2, 0))


def test_decode_predictions():
    x = np.zeros((2, 1000))
    x[0, 372] = 1.0
    x[1, 549] = 1.0
    outs = utils.decode_predictions(x, top=1)
    scores = [out[0][2] for out in outs]
    assert scores[0] == scores[1]

    # the numbers of columns and ImageNet classes are not identical.
    with pytest.raises(ValueError):
        utils.decode_predictions(np.ones((2, 100)))


def test_obtain_input_shape():
    # input_shape and default_size are not identical.
    with pytest.raises(ValueError):
        utils._obtain_input_shape(
            input_shape=(224, 224, 3),
            default_size=299,
            min_size=139,
            data_format='channels_last',
            require_flatten=True,
            weights='imagenet')

    # Test invalid use cases
    for data_format in ['channels_last', 'channels_first']:

        # test warning
        shape = (139, 139)
        input_shape = shape + (99,) if data_format == 'channels_last' else (99,) + shape
        with pytest.warns(UserWarning):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False,
                weights='fake_weights')

        # input_shape is smaller than min_size.
        shape = (100, 100)
        input_shape = shape + (3,) if data_format == 'channels_last' else (3,) + shape
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False)

        # shape is 1D.
        shape = (100,)
        input_shape = shape + (3,) if data_format == 'channels_last' else (3,) + shape
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False)

        # the number of channels is 5 not 3.
        shape = (100, 100)
        input_shape = shape + (5,) if data_format == 'channels_last' else (5,) + shape
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False)

        # require_flatten=True with dynamic input shape.
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=None,
                default_size=None,
                min_size=139,
                data_format='channels_first',
                require_flatten=True)

    # test include top
    assert utils._obtain_input_shape(
        input_shape=(3, 200, 200),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=True) == (3, 200, 200)

    assert utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False) == (None, None, 3)

    assert utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=False) == (3, None, None)

    assert utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False) == (None, None, 3)

    assert utils._obtain_input_shape(
        input_shape=(150, 150, 3),
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False) == (150, 150, 3)

    assert utils._obtain_input_shape(
        input_shape=(3, None, None),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=False) == (3, None, None)


if __name__ == '__main__':
    pytest.main([__file__])
