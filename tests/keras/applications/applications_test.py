import pytest
from keras.utils.test_utils import keras_test
from keras.utils.test_utils import layer_test
from keras.utils.generic_utils import CustomObjectScope
from keras.models import Sequential
from keras import applications
from keras import backend as K


@keras_test
def test_resnet50():
    model = applications.ResNet50(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_resnet50_notop():
    model = applications.ResNet50(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
def test_resnet50_notop_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    output_shape = (None, 2048, 1, 1) if K.image_data_format() == 'channels_first' else (None, 1, 1, 2048)
    assert model.output_shape == output_shape


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_resnet50_pooling():
    model = applications.ResNet50(weights=None,
                                  include_top=False,
                                  pooling='avg')
    assert model.output_shape == (None, 2048)


@keras_test
def test_resnet50_pooling_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.ResNet50(weights=None,
                                  include_top=False,
                                  pooling='avg',
                                  input_shape=input_shape)
    assert model.output_shape == (None, 2048)


@keras_test
def test_vgg16():
    model = applications.VGG16(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_vgg16_notop():
    model = applications.VGG16(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 512)


@keras_test
def test_vgg16_notop_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    output_shape = (None, 512, 9, 9) if K.image_data_format() == 'channels_first' else (None, 9, 9, 512)
    assert model.output_shape == output_shape


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_vgg16_pooling():
    model = applications.VGG16(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 512)


@keras_test
def test_vgg16_pooling_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.VGG16(weights=None, include_top=False, pooling='avg', input_shape=input_shape)
    assert model.output_shape == (None, 512)


@keras_test
def test_vgg19():
    model = applications.VGG19(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_vgg19_notop():
    model = applications.VGG19(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 512)


@keras_test
def test_vgg19_notop_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.VGG19(weights=None, include_top=False, input_shape=input_shape)
    output_shape = (None, 512, 9, 9) if K.image_data_format() == 'channels_first' else (None, 9, 9, 512)
    assert model.output_shape == output_shape


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_vgg19_pooling():
    model = applications.VGG16(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 512)


@keras_test
def test_vgg19_pooling_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.VGG16(weights=None, include_top=False, pooling='avg', input_shape=input_shape)
    assert model.output_shape == (None, 512)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires tensorflow backend')
def test_xception():
    model = applications.Xception(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires tensorflow backend')
def test_xception_notop():
    model = applications.Xception(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires tensorflow backend')
def test_xception_pooling():
    model = applications.Xception(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 2048)


@keras_test
def test_inceptionv3():
    model = applications.InceptionV3(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_inceptionv3_notop():
    model = applications.InceptionV3(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support padding with non-concrete dimension")
def test_inceptionv3_pooling():
    model = applications.InceptionV3(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 2048)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason="MobileNets are supported only on TensorFlow")
def test_mobilenet():
    model = applications.MobileNet(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason="MobileNets are supported only on TensorFlow")
def test_mobilenet_no_top():
    model = applications.MobileNet(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 1024)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason="MobileNets are supported only on TensorFlow")
def test_mobilenet_pooling():
    model = applications.MobileNet(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 1024)


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TF backend')
@keras_test
def test_depthwise_conv_2d():
    _convolution_paddings = ['valid', 'same']
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    with CustomObjectScope({'relu6': applications.mobilenet.relu6,
                            'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
        for padding in _convolution_paddings:
            for strides in [(1, 1), (2, 2)]:
                for multiplier in [1, 2]:
                    if padding == 'same' and strides != (1, 1):
                        continue

                    layer_test(applications.mobilenet.DepthwiseConv2D,
                               kwargs={'kernel_size': (3, 3),
                                       'padding': padding,
                                       'strides': strides,
                                       'depth_multiplier': multiplier},
                               input_shape=(num_samples, num_row, num_col, stack_size))

        layer_test(applications.mobilenet.DepthwiseConv2D,
                   kwargs={'kernel_size': 3,
                           'padding': padding,
                           'data_format': 'channels_first',
                           'activation': None,
                           'depthwise_regularizer': 'l2',
                           'bias_regularizer': 'l2',
                           'activity_regularizer': 'l2',
                           'depthwise_constraint': 'unit_norm',
                           'strides': strides,
                           'depth_multiplier': multiplier},
                   input_shape=(num_samples, stack_size, num_row, num_col))

        # Test invalid use case
        with pytest.raises(ValueError):
            model = Sequential([applications.mobilenet.DepthwiseConv2D(kernel_size=3,
                                                                       padding=padding,
                                                                       batch_input_shape=(None, None, 5, None))])


if __name__ == '__main__':
    pytest.main([__file__])
