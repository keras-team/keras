import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.utils import layer_utils


def test_convert_weights():
    def get_model(shape, data_format):
        model = Sequential()
        model.add(Conv2D(filters=2,
                         kernel_size=(4, 3),
                         input_shape=shape,
                         data_format=data_format))
        model.add(Flatten())
        model.add(Dense(5))
        return model

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            shape = (3, 5, 5)
            target_shape = (5, 5, 3)
            prev_shape = (2, 3, 2)
            flip = lambda x: np.flip(np.flip(x, axis=2), axis=3)
            transpose = lambda x: np.transpose(x, (0, 2, 3, 1))
            target_data_format = 'channels_last'
        elif data_format == 'channels_last':
            shape = (5, 5, 3)
            target_shape = (3, 5, 5)
            prev_shape = (2, 2, 3)
            flip = lambda x: np.flip(np.flip(x, axis=1), axis=2)
            transpose = lambda x: np.transpose(x, (0, 3, 1, 2))
            target_data_format = 'channels_first'

        model1 = get_model(shape, data_format)
        model2 = get_model(target_shape, target_data_format)
        conv = K.function([model1.input], [model1.layers[0].output])

        x = np.random.random((1,) + shape)

        # Test equivalence of convert_all_kernels_in_model
        convout1 = conv([x])[0]
        layer_utils.convert_all_kernels_in_model(model1)
        convout2 = flip(conv([flip(x)])[0])

        assert_allclose(convout1, convout2, atol=1e-5)

        # Test equivalence of convert_dense_weights_data_format
        out1 = model1.predict(x)
        layer_utils.convert_dense_weights_data_format(
            model1.layers[2], prev_shape, target_data_format)
        for (src, dst) in zip(model1.layers, model2.layers):
            dst.set_weights(src.get_weights())
        out2 = model2.predict(transpose(x))

        assert_allclose(out1, out2, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
