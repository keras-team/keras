import pytest
import random
import os
from multiprocessing import Process, Queue
from keras.utils.test_utils import keras_test
from keras.utils.test_utils import layer_test
from keras.models import Sequential
from keras import applications
from keras import backend as K


pytestmark = pytest.mark.skipif(
    os.environ.get('CORE_CHANGED', 'True') == 'False' and
    os.environ.get('APP_CHANGED', 'True') == 'False',
    reason='Runs only when the relevant files have been modified.')


DENSENET_LIST = [(applications.DenseNet121, 1024),
                 (applications.DenseNet169, 1664),
                 (applications.DenseNet201, 1920)]
NASNET_LIST = [(applications.NASNetMobile, 1056),
               (applications.NASNetLarge, 4032)]


def _get_output_shape(model_fn):
    if K.backend() == 'cntk':
        # Create model in a subprocess so that
        # the memory consumed by InceptionResNetV2 will be
        # released back to the system after this test
        # (to deal with OOM error on CNTK backend).
        # TODO: remove the use of multiprocessing from these tests
        # once a memory clearing mechanism
        # is implemented in the CNTK backend.
        def target(queue):
            model = model_fn()
            queue.put(model.output_shape)
        queue = Queue()
        p = Process(target=target, args=(queue,))
        p.start()
        p.join()
        # The error in a subprocess won't propagate
        # to the main process, so we check if the model
        # is successfully created by checking if the output shape
        # has been put into the queue
        assert not queue.empty(), 'Model creation failed.'
        return queue.get_nowait()
    else:
        model = model_fn()
        return model.output_shape


@keras_test
def _test_application_basic(app, last_dim=1000):
    output_shape = _get_output_shape(lambda: app(weights=None))
    assert output_shape == (None, last_dim)


@keras_test
def _test_application_notop(app, last_dim):
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False))
    assert output_shape == (None, None, None, last_dim)


@keras_test
def _test_application_variable_input_channels(app, last_dim):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, input_shape=input_shape))
    assert output_shape == (None, None, None, last_dim)

    if K.image_data_format() == 'channels_first':
        input_shape = (4, None, None)
    else:
        input_shape = (None, None, 4)
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, input_shape=input_shape))
    assert output_shape == (None, None, None, last_dim)


@keras_test
def _test_app_pooling(app, last_dim):
    output_shape = _get_output_shape(
        lambda: app(weights=None,
                    include_top=False,
                    pooling=random.choice(['avg', 'max'])))
    assert output_shape == (None, last_dim)


def test_resnet50():
    app = applications.ResNet50
    last_dim = 2048
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_vgg():
    app = random.choice([applications.VGG16, applications.VGG19])
    last_dim = 512
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='Requires TensorFlow backend')
def test_xception():
    app = applications.Xception
    last_dim = 2048
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_inceptionv3():
    app = applications.InceptionV3
    last_dim = 2048
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_app_pooling(app, last_dim)

    if K.backend() != 'cntk':
        # CNTK does not support dynamic padding.
        _test_application_variable_input_channels(app, last_dim)


def test_inceptionresnetv2():
    app = applications.InceptionResNetV2
    last_dim = 1536
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='MobileNets are supported only on TensorFlow')
def test_mobilenet():
    app = applications.MobileNet
    last_dim = 1024
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_densenet():
    app, last_dim = random.choice(DENSENET_LIST)
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


@pytest.mark.skipif((K.backend() != 'tensorflow'),
                    reason='NASNets are supported only on TensorFlow')
def test_nasnet():
    app, last_dim = random.choice(NASNET_LIST)
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


if __name__ == '__main__':
    pytest.main([__file__])
