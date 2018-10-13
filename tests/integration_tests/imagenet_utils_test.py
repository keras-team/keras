import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.applications import imagenet_utils as utils
from keras.models import Model
from keras.layers import Input, Lambda


def test_preprocess_input():
    # Test image batch with float and int image input
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    xint = x.astype('int32')
    assert utils.preprocess_input(x).shape == x.shape
    assert utils.preprocess_input(xint).shape == xint.shape

    out1 = utils.preprocess_input(x, 'channels_last')
    out1int = utils.preprocess_input(xint, 'channels_last')
    out2 = utils.preprocess_input(np.transpose(x, (0, 3, 1, 2)),
                                  'channels_first')
    out2int = utils.preprocess_input(np.transpose(xint, (0, 3, 1, 2)),
                                     'channels_first')
    assert_allclose(out1, out2.transpose(0, 2, 3, 1))
    assert_allclose(out1int, out2int.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    xint = x.astype('int32')
    assert utils.preprocess_input(x).shape == x.shape
    assert utils.preprocess_input(xint).shape == xint.shape

    out1 = utils.preprocess_input(x, 'channels_last')
    out1int = utils.preprocess_input(xint, 'channels_last')
    out2 = utils.preprocess_input(np.transpose(x, (2, 0, 1)),
                                  'channels_first')
    out2int = utils.preprocess_input(np.transpose(xint, (2, 0, 1)),
                                     'channels_first')
    assert_allclose(out1, out2.transpose(1, 2, 0))
    assert_allclose(out1int, out2int.transpose(1, 2, 0))

    # Test that writing over the input data works predictably
    for mode in ['torch', 'tf']:
        x = np.random.uniform(0, 255, (2, 10, 10, 3))
        xint = x.astype('int')
        x2 = utils.preprocess_input(x, mode=mode)
        xint2 = utils.preprocess_input(xint)
        assert_allclose(x, x2)
        assert xint.astype('float').max() != xint2.max()
    # Caffe mode works differently from the others
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    xint = x.astype('int')
    x2 = utils.preprocess_input(x, data_format='channels_last', mode='caffe')
    xint2 = utils.preprocess_input(xint)
    assert_allclose(x, x2[..., ::-1])
    assert xint.astype('float').max() != xint2.max()


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


if __name__ == '__main__':
    pytest.main([__file__])
