import csv
import os
import re
import tempfile

import numpy as np
import pytest

from keras import callbacks
from keras import initializers
from keras import layers
from keras import testing
from keras.models import Sequential
from keras.utils import numerical_utils

TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
INPUT_DIM = 3
BATCH_SIZE = 4


class CSVLoggerTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_CSVLogger(self):
        OUTPUT_DIM = 1
        np.random.seed(1337)
        temp_dir = tempfile.TemporaryDirectory()
        filepath = os.path.join(temp_dir.name, "log.tsv")

        sep = "\t"
        x_train = np.random.random((TRAIN_SAMPLES, INPUT_DIM))
        y_train = np.random.random((TRAIN_SAMPLES, OUTPUT_DIM))
        x_test = np.random.random((TEST_SAMPLES, INPUT_DIM))
        y_test = np.random.random((TEST_SAMPLES, OUTPUT_DIM))

        def make_model():
            np.random.seed(1337)
            model = Sequential(
                [
                    layers.Dense(2, activation="relu"),
                    layers.Dense(OUTPUT_DIM),
                ]
            )
            model.compile(
                loss="mse",
                optimizer="sgd",
                metrics=["mse"],
            )
            return model

        # case 1, create new file with defined separator
        model = make_model()
        cbks = [callbacks.CSVLogger(filepath, separator=sep)]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )

        assert os.path.exists(filepath)
        with open(filepath) as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read())
        assert dialect.delimiter == sep
        del model
        del cbks

        # case 2, append data to existing file, skip header
        model = make_model()
        cbks = [callbacks.CSVLogger(filepath, separator=sep, append=True)]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )

        # case 3, reuse of CSVLogger object
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=2,
            verbose=0,
        )

        with open(filepath) as csvfile:
            list_lines = csvfile.readlines()
            for line in list_lines:
                assert line.count(sep) == 4
            assert len(list_lines) == 5
            output = " ".join(list_lines)
            assert len(re.findall("epoch", output)) == 1

        os.remove(filepath)

        # case 3, Verify Val. loss also registered when Validation Freq > 1
        model = make_model()
        cbks = [callbacks.CSVLogger(filepath, separator=sep)]
        hist = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            validation_freq=3,
            callbacks=cbks,
            epochs=5,
            verbose=0,
        )
        assert os.path.exists(filepath)
        # Verify that validation loss is registered at val. freq
        with open(filepath) as csvfile:
            rows = csv.DictReader(csvfile, delimiter=sep)
            for idx, row in enumerate(rows, 1):
                self.assertIn("val_loss", row)
                if idx == 3:
                    self.assertEqual(
                        row["val_loss"], str(hist.history["val_loss"][0])
                    )
                else:
                    self.assertEqual(row["val_loss"], "NA")

    @pytest.mark.requires_trainable_backend
    def test_stop_training_csv(self):
        # Test that using the CSVLogger callback with the TerminateOnNaN
        # callback does not result in invalid CSVs.
        tmpdir = tempfile.TemporaryDirectory()
        csv_logfile = os.path.join(tmpdir.name, "csv_logger.csv")
        NUM_CLASSES = 2
        np.random.seed(1337)
        x_train = np.random.random((TRAIN_SAMPLES, INPUT_DIM))
        y_train = np.random.choice(np.arange(NUM_CLASSES), size=TRAIN_SAMPLES)
        x_test = np.random.random((TEST_SAMPLES, INPUT_DIM))
        y_test = np.random.choice(np.arange(NUM_CLASSES), size=TEST_SAMPLES)

        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)
        model = Sequential()
        initializer = initializers.Constant(value=1e5)
        for _ in range(5):
            model.add(
                layers.Dense(
                    2,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        model.add(layers.Dense(NUM_CLASSES))
        model.compile(loss="mean_squared_error", optimizer="sgd")

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[
                callbacks.TerminateOnNaN(),
                callbacks.CSVLogger(csv_logfile),
            ],
            epochs=20,
        )
        loss = history.history["loss"]
        self.assertEqual(len(loss), 1)
        self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))

        values = []
        with open(csv_logfile) as f:
            # On Windows, due to \r\n line ends, we may end up reading empty
            # lines after each line. Skip empty lines.
            values = [x for x in csv.reader(f) if x]
        self.assertIn("nan", values[-1], "NaN not logged in CSV Logger.")
