import pytest

from keras.src import callbacks
from keras.src import layers
from keras.src import optimizers
from keras.src import testing
from keras.src.models import Sequential
from keras.src.testing import test_utils
from keras.src.utils import io_utils
from keras.src.utils import numerical_utils


class ReduceLROnPlateauTest(testing.TestCase):
    def setUp(self):
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=10,
            test_samples=10,
            input_shape=(3,),
            num_classes=2,
        )
        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)

        model = Sequential([layers.Dense(5), layers.Dense(2)])

        model.compile(
            loss="mse",
            optimizer=optimizers.Adam(0.1),
        )

        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @pytest.mark.requires_trainable_backend
    def test_reduces_lr_with_model_fit(self):
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=1, factor=0.1, monitor="val_loss", min_delta=100
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            callbacks=[reduce_lr],
            epochs=2,
        )

        self.assertEqual(self.model.optimizer.learning_rate.value, 0.01)

    @pytest.mark.requires_trainable_backend
    def test_throws_when_optimizer_has_schedule(self):
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=1, factor=0.1, monitor="val_loss", min_delta=100
        )

        self.model.compile(
            loss="mse",
            optimizer=optimizers.Adam(
                optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=0.1, decay_steps=10
                )
            ),
        )

        with self.assertRaisesRegex(
            TypeError,
            "This optimizer was created with a `LearningRateSchedule`",
        ):
            self.model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_test, self.y_test),
                callbacks=[reduce_lr],
                epochs=2,
            )

    @pytest.mark.requires_trainable_backend
    def test_verbose_logging(self):
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=1, factor=0.1, monitor="val_loss", min_delta=100, verbose=1
        )
        io_utils.disable_interactive_logging()
        io_utils.set_logging_verbosity("INFO")

        with self.assertLogs() as logs:
            self.model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_test, self.y_test),
                callbacks=[reduce_lr],
                epochs=2,
            )
            expected_log = "ReduceLROnPlateau reducing learning rate to 0.01"
            self.assertTrue(any(expected_log in log for log in logs.output))

    @pytest.mark.requires_trainable_backend
    def test_honors_min_lr(self):
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=1,
            factor=0.1,
            monitor="val_loss",
            min_delta=10,
            min_lr=0.005,
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            callbacks=[reduce_lr],
            epochs=4,
        )

        self.assertEqual(self.model.optimizer.learning_rate.value, 0.005)

    @pytest.mark.requires_trainable_backend
    def test_cooldown(self):
        reduce_lr = callbacks.ReduceLROnPlateau(
            patience=1,
            factor=0.1,
            monitor="val_loss",
            min_delta=100,
            cooldown=2,
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            callbacks=[reduce_lr],
            epochs=4,
        )

        # With a cooldown of 2 epochs, we should only reduce the LR every other
        # epoch, so after 4 epochs we will have reduced 2 times.
        self.assertAllClose(self.model.optimizer.learning_rate.value, 0.001)
