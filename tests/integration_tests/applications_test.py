import pytest
import random
import os
from multiprocessing import Process, Queue
from keras import applications
from keras import backend as K


MODEL_LIST = [
    (applications.ResNet50, 2048),
    (applications.ResNet101, 2048),
    (applications.ResNet152, 2048),
    (applications.ResNet50V2, 2048),
    (applications.ResNet101V2, 2048),
    (applications.ResNet152V2, 2048),
    (applications.VGG16, 512),
    (applications.VGG19, 512),
    (applications.Xception, 2048),
    (applications.InceptionV3, 2048),
    (applications.InceptionResNetV2, 1536),
    (applications.MobileNet, 1024),
    (applications.MobileNetV2, 1280),
    (applications.DenseNet121, 1024),
    (applications.DenseNet169, 1664),
    (applications.DenseNet201, 1920),
    # Note that NASNetLarge is too heavy to test on Travis.
    (applications.NASNetMobile, 1056)
]


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


def _test_application_basic(app, last_dim=1000):
    output_shape = _get_output_shape(lambda: app(weights=None))
    assert output_shape == (None, last_dim)


def _test_application_notop(app, last_dim):
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False))
    assert len(output_shape) == 4
    assert output_shape[-1] == last_dim


def test_applications():
    for _ in range(3):
        app, last_dim = random.choice(MODEL_LIST)
        _test_application_basic(app)
        _test_application_notop(app, last_dim)


if __name__ == '__main__':
    pytest.main([__file__])
