import pytest
from keras.utils.test_utils import keras_test
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


if __name__ == '__main__':
    pytest.main([__file__])
