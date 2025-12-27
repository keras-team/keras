from keras.src import callbacks
from keras.src import testing


class ProgbarLoggerTest(testing.TestCase):
    def test_progbar_logger_full_flow(self):
        logger = callbacks.ProgbarLogger()
        params = {
            "verbose": 1,
            "epochs": 2,
            "steps": 10,
        }
        logger.set_params(params)

        logger.on_train_begin()
        logger.on_epoch_begin(0)
        self.assertFalse(logger.pinned)

        for i in range(10):
            logger.on_train_batch_end(i, logs={"loss": 0.5})
        logger.on_epoch_end(0)

    def test_progbar_logger_pinned(self):
        """
        Verify that the pinned parameter correctly
        initializes the Progbar.

        """
        logger = callbacks.ProgbarLogger(pinned=True)

        params = {
            "verbose": 1,
            "epochs": 1,
            "steps": 5,
        }
        logger.set_params(params)

        logger.on_train_begin()
        logger.on_epoch_begin(0)

        self.assertTrue(logger.progbar.pinned)

        for i in range(5):
            logger.on_train_batch_end(i, logs={"loss": 0.1})

        logger.on_epoch_end(0)

    def test_progbar_logger_verbose_2(self):
        logger = callbacks.ProgbarLogger(pinned=True)
        params = {
            "verbose": 2,
            "epochs": 1,
            "steps": 5,
        }
        logger.set_params(params)
        logger.on_epoch_begin(0)
        logger.on_epoch_end(0, logs={"loss": 0.1})
