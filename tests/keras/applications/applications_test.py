import pytest
from multiprocessing import Process, Queue
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
def test_resnet50_notop():
    model = applications.ResNet50(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
def test_resnet50_variable_input_channels():
    input_shape = (1, None, None) if K.image_data_format() == 'channels_first' else (None, None, 1)
    model = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 2048)

    input_shape = (4, None, None) if K.image_data_format() == 'channels_first' else (None, None, 4)
    model = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
def test_resnet50_pooling():
    model = applications.ResNet50(weights=None,
                                  include_top=False,
                                  pooling='avg')
    assert model.output_shape == (None, 2048)


@keras_test
def test_vgg16():
    model = applications.VGG16(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
def test_vgg16_notop():
    model = applications.VGG16(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 512)


@keras_test
def test_vgg16_variable_input_channels():
    input_shape = (1, None, None) if K.image_data_format() == 'channels_first' else (None, None, 1)
    model = applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 512)

    input_shape = (4, None, None) if K.image_data_format() == 'channels_first' else (None, None, 4)
    model = applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 512)


@keras_test
def test_vgg16_pooling():
    model = applications.VGG16(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 512)


@keras_test
def test_vgg19():
    model = applications.VGG19(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
def test_vgg19_notop():
    model = applications.VGG19(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 512)


@keras_test
def test_vgg19_variable_input_channels():
    input_shape = (1, None, None) if K.image_data_format() == 'channels_first' else (None, None, 1)
    model = applications.VGG19(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 512)

    input_shape = (4, None, None) if K.image_data_format() == 'channels_first' else (None, None, 4)
    model = applications.VGG19(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 512)


@keras_test
def test_vgg19_notop_specified_input_shape():
    input_shape = (3, 300, 300) if K.image_data_format() == 'channels_first' else (300, 300, 3)
    model = applications.VGG19(weights=None, include_top=False, input_shape=input_shape)
    output_shape = (None, 512, 9, 9) if K.image_data_format() == 'channels_first' else (None, 9, 9, 512)
    assert model.output_shape == output_shape


@keras_test
def test_vgg19_pooling():
    model = applications.VGG16(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 512)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
def test_xception():
    model = applications.Xception(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
def test_xception_notop():
    model = applications.Xception(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
def test_xception_pooling():
    model = applications.Xception(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 2048)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
def test_xception_variable_input_channels():
    input_shape = (1, None, None) if K.image_data_format() == 'channels_first' else (None, None, 1)
    model = applications.Xception(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 2048)

    input_shape = (4, None, None) if K.image_data_format() == 'channels_first' else (None, None, 4)
    model = applications.Xception(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
def test_inceptionv3():
    model = applications.InceptionV3(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
def test_inceptionv3_notop():
    model = applications.InceptionV3(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
def test_inceptionv3_pooling():
    model = applications.InceptionV3(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 2048)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='cntk does not support padding with non-concrete dimension')
def test_inceptionv3_variable_input_channels():
    input_shape = (1, None, None) if K.image_data_format() == 'channels_first' else (None, None, 1)
    model = applications.InceptionV3(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 2048)

    input_shape = (4, None, None) if K.image_data_format() == 'channels_first' else (None, None, 4)
    model = applications.InceptionV3(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 2048)


@keras_test
def test_inceptionresnetv2():
    # Create model in a subprocess so that the memory consumed by InceptionResNetV2 will be
    # released back to the system after this test (to deal with OOM error on CNTK backend)
    # TODO: remove the use of multiprocessing from these tests once a memory clearing mechanism
    # is implemented in the CNTK backend
    def target(queue):
        model = applications.InceptionResNetV2(weights=None)
        queue.put(model.output_shape)
    queue = Queue()
    p = Process(target=target, args=(queue,))
    p.start()
    p.join()

    # The error in a subprocess won't propagate to the main process, so we check if the model
    # is successfully created by checking if the output shape has been put into the queue
    assert not queue.empty(), 'Model creation failed.'
    model_output_shape = queue.get_nowait()
    assert model_output_shape == (None, 1000)


@keras_test
def test_inceptionresnetv2_notop():
    def target(queue):
        model = applications.InceptionResNetV2(weights=None, include_top=False)
        queue.put(model.output_shape)

    global_image_data_format = K.image_data_format()
    queue = Queue()

    K.set_image_data_format('channels_first')
    p = Process(target=target, args=(queue,))
    p.start()
    p.join()
    K.set_image_data_format(global_image_data_format)
    assert not queue.empty(), 'Model creation failed.'
    model_output_shape = queue.get_nowait()
    assert model_output_shape == (None, 1536, None, None)

    K.set_image_data_format('channels_last')
    p = Process(target=target, args=(queue,))
    p.start()
    p.join()
    K.set_image_data_format(global_image_data_format)
    assert not queue.empty(), 'Model creation failed.'
    model_output_shape = queue.get_nowait()
    assert model_output_shape == (None, None, None, 1536)


@keras_test
def test_inceptionresnetv2_pooling():
    def target(queue):
        model = applications.InceptionResNetV2(weights=None, include_top=False, pooling='avg')
        queue.put(model.output_shape)
    queue = Queue()
    p = Process(target=target, args=(queue,))
    p.start()
    p.join()
    assert not queue.empty(), 'Model creation failed.'
    model_output_shape = queue.get_nowait()
    assert model_output_shape == (None, 1536)


@keras_test
def test_inceptionresnetv2_variable_input_channels():
    def target(queue, input_shape):
        model = applications.InceptionResNetV2(weights=None, include_top=False, input_shape=input_shape)
        queue.put(model.output_shape)

    queue = Queue()
    p = Process(target=target, args=(queue, (None, None, 1)))
    p.start()
    p.join()
    assert not queue.empty(), 'Model creation failed.'
    model_output_shape = queue.get_nowait()
    assert model_output_shape == (None, None, None, 1536)

    p = Process(target=target, args=(queue, (None, None, 4)))
    p.start()
    p.join()
    assert not queue.empty(), 'Model creation failed.'
    model_output_shape = queue.get_nowait()
    assert model_output_shape == (None, None, None, 1536)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='MobileNets are supported only on TensorFlow')
def test_mobilenet():
    model = applications.MobileNet(weights=None)
    assert model.output_shape == (None, 1000)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='MobileNets are supported only on TensorFlow')
def test_mobilenet_no_top():
    model = applications.MobileNet(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, 1024)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='MobileNets are supported only on TensorFlow')
def test_mobilenet_pooling():
    model = applications.MobileNet(weights=None, include_top=False, pooling='avg')
    assert model.output_shape == (None, 1024)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='MobileNets are supported only on TensorFlow')
def test_mobilenet_variable_input_channels():
    input_shape = (1, None, None) if K.image_data_format() == 'channels_first' else (None, None, 1)
    model = applications.MobileNet(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 1024)

    input_shape = (4, None, None) if K.image_data_format() == 'channels_first' else (None, None, 4)
    model = applications.MobileNet(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, 1024)


@keras_test
@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='MobileNets are supported only on TensorFlow')
def test_mobilenet_image_size():
    valid_image_sizes = [128, 160, 192, 224]
    for size in valid_image_sizes:
        input_shape = (size, size, 3) if K.image_data_format() == 'channels_last' else (3, size, size)
        model = applications.MobileNet(input_shape=input_shape, weights='imagenet', include_top=True)
        assert model.input_shape == (None,) + input_shape

    invalid_image_shape = (112, 112, 3) if K.image_data_format() == 'channels_last' else (3, 112, 112)
    with pytest.raises(ValueError):
        model = applications.MobileNet(input_shape=invalid_image_shape, weights='imagenet', include_top=True)


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
