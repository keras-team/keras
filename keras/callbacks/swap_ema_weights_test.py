import tempfile

import pytest

from keras import callbacks
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras import saving
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

    def _get_compiled_model(self, use_ema=True):
        model = Sequential(
            [layers.Dense(2, kernel_initializer="ones", use_bias=False)]
        )
        model.compile(
            optimizer=optimizers.SGD(use_ema=use_ema, ema_momentum=0.9),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        return model

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_with_invalid_optimizer(self):
        model = self._get_compiled_model(use_ema=False)
        with self.assertRaisesRegex(
            ValueError,
            "SwapEMAWeights must be used with `use_ema=True`",
        ):
            model.fit(
                self.x_train,
                self.y_train,
                epochs=2,
                callbacks=[callbacks.SwapEMAWeights()],
                validation_data=(self.x_train, self.y_train),
            )

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights(self):
        # not using SwapEMAWeights
        model = self._get_compiled_model()
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=2,
            validation_data=(self.x_train, self.y_train),
        )
        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        # final metric during fitting is different from the evaluation
        self.assertNotEqual(
            history.history["val_mean_squared_error"][-1],
            logs["mean_squared_error"],
        )

        # using SwapEMAWeights
        model = self._get_compiled_model()
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=2,
            callbacks=[callbacks.SwapEMAWeights()],
            validation_data=(self.x_train, self.y_train),
        )
        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        # final metric during fitting is same as the evaluation
        self.assertEqual(
            history.history["val_mean_squared_error"][-1],
            logs["mean_squared_error"],
        )

    @pytest.mark.requires_trainable_backend
    def test_swap_ema_weights_on_epoch(self):
        # using SwapEMAWeights together with ModelCheckpoint
        model = self._get_compiled_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model.fit(
                self.x_train,
                self.y_train,
                epochs=2,
                callbacks=[
                    callbacks.ModelCheckpoint(temp_dir + "/{epoch:1d}.keras"),
                    callbacks.SwapEMAWeights(swap_on_epoch=True),
                ],
                validation_data=(self.x_train, self.y_train),
            )
            model2 = saving.load_model(temp_dir + "/2.keras")

        logs = model.evaluate(self.x_train, self.y_train, return_dict=True)
        logs2 = model2.evaluate(self.x_train, self.y_train, return_dict=True)
        # saved checkpoint will be applied by EMA weights
        self.assertEqual(
            logs["mean_squared_error"],
            logs2["mean_squared_error"],
        )
