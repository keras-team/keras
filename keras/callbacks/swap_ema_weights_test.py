import pytest

from keras import callbacks
from keras import layers
from keras import ops
from keras import optimizers
from keras import testing
from keras.models import Sequential
from keras.testing import test_utils
from keras.utils import numerical_utils


class SwapEMAWeightsTest(testing.TestCase):
    def setUp(self):
        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=10,
            test_samples=10,
            input_shape=(3,),
            num_classes=2,
            random_seed=2023,
        )
        y_train = numerical_utils.to_categorical(y_train)

        self.x_train = x_train
        self.y_train = y_train

    def _get_numpy_trainable_variables(self, var_list):
        result = []
        for v in var_list:
            result.append(ops.convert_to_numpy(v))
        return result

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights(self):
        optimizer = optimizers.SGD(use_ema=True, ema_momentum=0.9)
        model = Sequential(
            [layers.Dense(2, kernel_initializer="ones", use_bias=False)]
        )
        model.compile(loss="mse", optimizer=optimizer)
        swap_ema_weights = callbacks.SwapEMAWeights()
        swap_ema_weights.set_model(model)

        # train_on_batch to make ema work
        for _ in range(5):
            model.train_on_batch(self.x_train, self.y_train)
        model_variables = ops.convert_to_numpy(model.trainable_variables[0])

        # test stage
        swap_ema_weights.on_test_begin()
        ema_variables = ops.convert_to_numpy(model.trainable_variables[0])
        self.assertNotAllClose(ema_variables, model_variables)

        swap_ema_weights.on_test_end()
        restored_variables = ops.convert_to_numpy(model.trainable_variables[0])
        self.assertAllClose(restored_variables, model_variables)

        # predict stage
        swap_ema_weights.on_predict_begin()
        ema_variables = ops.convert_to_numpy(model.trainable_variables[0])
        self.assertNotAllClose(ema_variables, model_variables)

        swap_ema_weights.on_predict_end()
        restored_variables = ops.convert_to_numpy(model.trainable_variables[0])
        self.assertAllClose(restored_variables, model_variables)

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_on_epoch(self):
        optimizer = optimizers.SGD(use_ema=True, ema_momentum=0.9)
        model = Sequential(
            [layers.Dense(2, kernel_initializer="ones", use_bias=False)]
        )
        model.compile(loss="mse", optimizer=optimizer)
        swap_ema_weights = callbacks.SwapEMAWeights(swap_on_epoch=True)
        swap_ema_weights.set_model(model)

        # train_on_batch to make ema work
        for _ in range(5):
            model.train_on_batch(self.x_train, self.y_train)
        model_variables = ops.convert_to_numpy(model.trainable_variables[0])

        # epoch stage
        swap_ema_weights.on_epoch_begin()
        ema_variables = ops.convert_to_numpy(model.trainable_variables[0])
        self.assertNotAllClose(ema_variables, model_variables)

        swap_ema_weights.on_epoch_end(epoch=1)
        restored_variables = ops.convert_to_numpy(model.trainable_variables[0])
        self.assertAllClose(restored_variables, model_variables)
