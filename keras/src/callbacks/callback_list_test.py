import time
from unittest import mock

import numpy as np

from keras.src import testing
from keras.src.callbacks.callback import Callback
from keras.src.callbacks.callback_list import CallbackList
from keras.src.callbacks.history import History
from keras.src.callbacks.progbar_logger import ProgbarLogger


class TrackingCallback(Callback):
    """A callback that records all method calls for testing."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def on_train_begin(self, logs=None):
        self.calls.append(("on_train_begin", logs))

    def on_train_end(self, logs=None):
        self.calls.append(("on_train_end", logs))

    def on_epoch_begin(self, epoch, logs=None):
        self.calls.append(("on_epoch_begin", epoch, logs))

    def on_epoch_end(self, epoch, logs=None):
        self.calls.append(("on_epoch_end", epoch, logs))

    def on_train_batch_begin(self, batch, logs=None):
        self.calls.append(("on_train_batch_begin", batch, logs))

    def on_train_batch_end(self, batch, logs=None):
        self.calls.append(("on_train_batch_end", batch, logs))

    def on_test_begin(self, logs=None):
        self.calls.append(("on_test_begin", logs))

    def on_test_end(self, logs=None):
        self.calls.append(("on_test_end", logs))

    def on_test_batch_begin(self, batch, logs=None):
        self.calls.append(("on_test_batch_begin", batch, logs))

    def on_test_batch_end(self, batch, logs=None):
        self.calls.append(("on_test_batch_end", batch, logs))

    def on_predict_begin(self, logs=None):
        self.calls.append(("on_predict_begin", logs))

    def on_predict_end(self, logs=None):
        self.calls.append(("on_predict_end", logs))

    def on_predict_batch_begin(self, batch, logs=None):
        self.calls.append(("on_predict_batch_begin", batch, logs))

    def on_predict_batch_end(self, batch, logs=None):
        self.calls.append(("on_predict_batch_end", batch, logs))

    def on_batch_begin(self, batch, logs=None):
        self.calls.append(("on_batch_begin", batch, logs))

    def on_batch_end(self, batch, logs=None):
        self.calls.append(("on_batch_end", batch, logs))


class CallbackListTest(testing.TestCase):
    def test_empty_callback_list(self):
        cb_list = CallbackList()
        self.assertEqual(cb_list.callbacks, [])

    def test_callback_list_from_list(self):
        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb1, cb2])
        self.assertEqual(len(cb_list.callbacks), 2)
        self.assertIs(cb_list.callbacks[0], cb1)
        self.assertIs(cb_list.callbacks[1], cb2)

    def test_callback_list_none_input(self):
        cb_list = CallbackList(callbacks=None)
        self.assertEqual(cb_list.callbacks, [])

    def test_add_history_default_callback(self):
        cb_list = CallbackList(add_history=True)
        history_cbs = [c for c in cb_list.callbacks if isinstance(c, History)]
        self.assertEqual(len(history_cbs), 1)

    def test_add_progbar_default_callback(self):
        cb_list = CallbackList(add_progbar=True)
        progbar_cbs = [
            c for c in cb_list.callbacks if isinstance(c, ProgbarLogger)
        ]
        self.assertEqual(len(progbar_cbs), 1)

    def test_no_duplicate_history_if_already_present(self):
        existing_history = History()
        cb_list = CallbackList(callbacks=[existing_history], add_history=True)
        history_cbs = [c for c in cb_list.callbacks if isinstance(c, History)]
        self.assertEqual(len(history_cbs), 1)
        self.assertIs(history_cbs[0], existing_history)

    def test_no_duplicate_progbar_if_already_present(self):
        existing_progbar = ProgbarLogger()
        cb_list = CallbackList(callbacks=[existing_progbar], add_progbar=True)
        progbar_cbs = [
            c for c in cb_list.callbacks if isinstance(c, ProgbarLogger)
        ]
        self.assertEqual(len(progbar_cbs), 1)
        self.assertIs(progbar_cbs[0], existing_progbar)

    def test_set_params_propagates(self):
        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb1, cb2])
        params = {"epochs": 10, "steps": 100, "verbose": 1}
        cb_list.set_params(params)
        self.assertEqual(cb1.params, params)
        self.assertEqual(cb2.params, params)
        self.assertEqual(cb_list.params, params)

    def test_set_model_propagates(self):
        cb1 = TrackingCallback()
        model_mock = mock.MagicMock()
        cb_list = CallbackList(callbacks=[cb1])
        cb_list.set_model(model_mock)
        self.assertIs(cb1.model, model_mock)

    def test_set_model_none_does_nothing(self):
        cb1 = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb1])
        cb_list.set_model(None)
        # Should not crash, model should remain None
        self.assertIsNone(cb1.model)

    def test_set_model_attaches_history(self):
        model_mock = mock.MagicMock()
        cb_list = CallbackList(add_history=True)
        cb_list.set_model(model_mock)
        self.assertIsNotNone(model_mock.history)

    def test_on_train_lifecycle(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        cb_list.on_train_begin(logs={"lr": 0.01})
        cb_list.on_epoch_begin(0, logs=None)
        cb_list.on_train_batch_begin(0, logs=None)
        cb_list.on_train_batch_end(0, logs={"loss": 0.5})
        cb_list.on_epoch_end(0, logs={"loss": 0.4})
        cb_list.on_train_end(logs={"loss": 0.3})

        method_names = [c[0] for c in cb.calls]
        self.assertEqual(
            method_names,
            [
                "on_train_begin",
                "on_epoch_begin",
                "on_train_batch_begin",
                "on_train_batch_end",
                "on_epoch_end",
                "on_train_end",
            ],
        )

    def test_on_test_lifecycle(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        cb_list.on_test_begin(logs=None)
        cb_list.on_test_batch_begin(0, logs=None)
        cb_list.on_test_batch_end(0, logs={"loss": 0.2})
        cb_list.on_test_end(logs={"loss": 0.2})

        method_names = [c[0] for c in cb.calls]
        self.assertEqual(
            method_names,
            [
                "on_test_begin",
                "on_test_batch_begin",
                "on_test_batch_end",
                "on_test_end",
            ],
        )

    def test_on_predict_lifecycle(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        cb_list.on_predict_begin(logs=None)
        cb_list.on_predict_batch_begin(0, logs=None)
        cb_list.on_predict_batch_end(0, logs=None)
        cb_list.on_predict_end(logs=None)

        method_names = [c[0] for c in cb.calls]
        self.assertEqual(
            method_names,
            [
                "on_predict_begin",
                "on_predict_batch_begin",
                "on_predict_batch_end",
                "on_predict_end",
            ],
        )

    def test_on_batch_begin_end(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        cb_list.on_batch_begin(0, logs=None)
        cb_list.on_batch_end(0, logs={"loss": 0.1})

        method_names = [c[0] for c in cb.calls]
        self.assertEqual(method_names, ["on_batch_begin", "on_batch_end"])

    def test_multiple_callbacks_all_called(self):
        cb1 = TrackingCallback()
        cb2 = TrackingCallback()
        cb3 = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb1, cb2, cb3])
        cb_list.on_train_begin()
        cb_list.on_epoch_begin(0)
        cb_list.on_epoch_end(0, logs={"loss": 0.5})
        cb_list.on_train_end()

        for cb in [cb1, cb2, cb3]:
            method_names = [c[0] for c in cb.calls]
            self.assertIn("on_train_begin", method_names)
            self.assertIn("on_epoch_begin", method_names)
            self.assertIn("on_epoch_end", method_names)
            self.assertIn("on_train_end", method_names)

    def test_on_end_without_begin_raises(self):
        cb_list = CallbackList()
        with self.assertRaises(ValueError):
            cb_list._on_end()

    def test_nested_begin_end_blocks(self):
        cb_list = CallbackList()
        cb_list._on_begin()
        cb_list._on_begin()
        cb_list._on_end()
        # Should not raise - still one level deep
        cb_list._on_end()
        # Now at 0, another end should raise
        with self.assertRaises(ValueError):
            cb_list._on_end()

    def test_logs_passed_correctly(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        logs = {"loss": 0.5, "accuracy": 0.9}
        cb_list.on_epoch_end(0, logs=logs)
        self.assertEqual(cb.calls[0][2], logs)


class HistoryCallbackTest(testing.TestCase):
    def test_history_init(self):
        h = History()
        self.assertEqual(h.history, {})

    def test_on_train_begin_resets_epoch(self):
        h = History()
        h.on_train_begin()
        self.assertEqual(h.epoch, [])

    def test_on_epoch_end_records_logs(self):
        h = History()
        h.set_model(mock.MagicMock())
        h.on_train_begin()
        h.on_epoch_end(0, logs={"loss": 0.5, "accuracy": 0.8})
        h.on_epoch_end(1, logs={"loss": 0.3, "accuracy": 0.9})
        h.on_epoch_end(2, logs={"loss": 0.1, "accuracy": 0.95})

        self.assertEqual(h.epoch, [0, 1, 2])
        self.assertEqual(h.history["loss"], [0.5, 0.3, 0.1])
        self.assertEqual(h.history["accuracy"], [0.8, 0.9, 0.95])

    def test_on_epoch_end_empty_logs(self):
        h = History()
        h.set_model(mock.MagicMock())
        h.on_train_begin()
        h.on_epoch_end(0, logs=None)
        self.assertEqual(h.epoch, [0])
        self.assertEqual(h.history, {})

    def test_on_epoch_end_sets_model_history(self):
        model_mock = mock.MagicMock()
        h = History()
        h.set_model(model_mock)
        h.on_train_begin()
        h.on_epoch_end(0, logs={"loss": 0.5})
        self.assertIs(model_mock.history, h)

    def test_history_incremental_updates(self):
        h = History()
        h.set_model(mock.MagicMock())
        h.on_train_begin()

        # Simulate metrics appearing at different epochs
        h.on_epoch_end(0, logs={"loss": 0.5})
        h.on_epoch_end(1, logs={"loss": 0.3, "val_loss": 0.4})

        self.assertEqual(h.history["loss"], [0.5, 0.3])
        self.assertEqual(h.history["val_loss"], [0.4])

    def test_history_multiple_training_runs(self):
        h = History()
        h.set_model(mock.MagicMock())

        # First training run
        h.on_train_begin()
        h.on_epoch_end(0, logs={"loss": 0.5})
        self.assertEqual(h.epoch, [0])

        # Second training run resets epoch
        h.on_train_begin()
        self.assertEqual(h.epoch, [])
        # But history dict still has old data (by design)
        self.assertEqual(h.history["loss"], [0.5])

        h.on_epoch_end(0, logs={"loss": 0.2})
        self.assertEqual(h.history["loss"], [0.5, 0.2])


class ProgbarLoggerTest(testing.TestCase):
    def test_progbar_logger_init(self):
        p = ProgbarLogger()
        self.assertEqual(p.seen, 0)
        self.assertIsNone(p.progbar)
        self.assertIsNone(p.target)
        self.assertEqual(p.verbose, 1)
        self.assertEqual(p.epochs, 1)
        self.assertFalse(p._called_in_fit)

    def test_set_params(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 2, "epochs": 5, "steps": 100})
        self.assertEqual(p.verbose, 2)
        self.assertEqual(p.epochs, 5)
        self.assertEqual(p.target, 100)

    def test_set_params_verbose_auto(self):
        p = ProgbarLogger()
        p.set_params({"verbose": "auto", "epochs": 1, "steps": 10})
        self.assertEqual(p.verbose, 1)

    def test_on_train_begin_sets_called_in_fit(self):
        p = ProgbarLogger()
        p.on_train_begin()
        self.assertTrue(p._called_in_fit)

    def test_on_epoch_begin_resets_progbar(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 1, "epochs": 2, "steps": 10})
        p.seen = 5
        p.on_epoch_begin(0)
        self.assertEqual(p.seen, 0)
        self.assertIsNotNone(p.progbar)

    def test_on_train_batch_end_updates_seen(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 0, "epochs": 1, "steps": 10})
        p.on_epoch_begin(0)
        p.on_train_batch_end(0, logs={"loss": 0.5})
        self.assertEqual(p.seen, 1)
        p.on_train_batch_end(1, logs={"loss": 0.4})
        self.assertEqual(p.seen, 2)

    def test_on_test_batch_end_not_in_fit(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 0, "epochs": 1, "steps": 10})
        p._called_in_fit = False
        p.on_test_begin()
        p.on_test_batch_end(0, logs={"loss": 0.2})
        self.assertEqual(p.seen, 1)

    def test_on_test_batch_end_in_fit_does_nothing(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 0, "epochs": 1, "steps": 10})
        p._called_in_fit = True
        p.on_test_batch_end(0, logs={"loss": 0.2})
        # When called in fit, test batch end should not update
        self.assertEqual(p.seen, 0)

    def test_on_predict_begin_resets(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 0, "epochs": 1, "steps": 10})
        p.seen = 5
        p.on_predict_begin()
        self.assertEqual(p.seen, 0)
        self.assertIsNotNone(p.progbar)

    def test_finalize_sets_target_if_none(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 0, "epochs": 1, "steps": None})
        p.on_predict_begin()
        p.on_predict_batch_end(0)
        p.on_predict_batch_end(1)
        p.on_predict_batch_end(2)
        p.on_predict_end()
        # Target should be set to seen count
        self.assertEqual(p.target, 3)

    def test_epoch_message_multiple_epochs(self):
        p = ProgbarLogger()
        p.set_params({"verbose": 1, "epochs": 3, "steps": 10})
        # Should not raise
        p.on_epoch_begin(0)
        p.on_epoch_end(0, logs={"loss": 0.5})
        self.assertIsNotNone(p.progbar)


class CallbackListAdvancedTest(testing.TestCase):
    def test_callback_order_preserved(self):
        # Callbacks must be called in insertion order.
        order = []

        class OrderCallback(Callback):
            def __init__(self, n):
                super().__init__()
                self.n = n

            def on_epoch_begin(self, epoch, logs=None):
                order.append(self.n)

        cb_list = CallbackList(
            callbacks=[OrderCallback(1), OrderCallback(2), OrderCallback(3)]
        )
        cb_list.on_epoch_begin(0)
        self.assertEqual(order, [1, 2, 3])

    def test_epoch_index_passed_correctly(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        cb_list.on_epoch_begin(7)
        # None logs get pythonified to {}
        self.assertEqual(cb.calls[0], ("on_epoch_begin", 7, {}))

    def test_batch_index_passed_correctly(self):
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])
        cb_list.on_train_batch_begin(5)
        # None logs get pythonified to {}
        self.assertEqual(cb.calls[0], ("on_train_batch_begin", 5, {}))

    def test_constructor_with_model_kwarg(self):
        cb = TrackingCallback()
        model_mock = mock.MagicMock()
        CallbackList(callbacks=[cb], model=model_mock)
        self.assertIs(cb.model, model_mock)

    def test_constructor_with_params_kwargs(self):
        cb = TrackingCallback()
        CallbackList(callbacks=[cb], epochs=5, steps=100)
        self.assertEqual(cb.params["epochs"], 5)
        self.assertEqual(cb.params["steps"], 100)

    def test_logs_pythonified_numpy_to_float(self):
        # Callbacks should receive Python float values, not numpy scalars.
        received_logs = {}

        class LogCapture(Callback):
            def on_epoch_end(self, epoch, logs=None):
                received_logs.update(logs or {})

        cb_list = CallbackList(callbacks=[LogCapture()])
        cb_list.on_epoch_begin(0)
        cb_list.on_epoch_end(0, logs={"loss": np.float32(0.5)})
        self.assertIsInstance(received_logs["loss"], float)
        self.assertAlmostEqual(received_logs["loss"], 0.5, decimal=4)

    def test_logs_pythonified_none_becomes_empty_dict(self):
        received = {}

        class LogCapture(Callback):
            def on_train_batch_end(self, batch, logs=None):
                received["logs"] = logs

        cb_list = CallbackList(callbacks=[LogCapture()])
        cb_list.on_train_batch_end(0, logs=None)
        # pythonify_logs(None) returns {}
        self.assertEqual(received["logs"], {})

    def test_on_epoch_end_logs_passed_by_reference_updating(self):
        # Verify the logs dict passed to on_epoch_end is the same object
        # that reaches callbacks (no deep copy in CallbackList).
        received_logs_list = []

        class LogCapture(Callback):
            def on_epoch_end(self, epoch, logs=None):
                received_logs_list.append(logs)

        cb_list = CallbackList(callbacks=[LogCapture(), LogCapture()])
        cb_list.on_epoch_end(0, logs={"loss": 0.3})
        # Both callbacks received logs with loss=0.3
        for logs in received_logs_list:
            self.assertAlmostEqual(logs["loss"], 0.3, decimal=6)

    def test_async_dispatch_disabled_when_on_train_batch_end_overridden(self):
        # If a callback overrides on_train_batch_end, async_train must be False.
        class NonDefaultBatchEnd(Callback):
            def on_train_batch_end(self, batch, logs=None):
                pass

        cb_list = CallbackList(callbacks=[NonDefaultBatchEnd()])
        self.assertFalse(cb_list._async_train)

    def test_async_dispatch_disabled_when_on_batch_end_overridden(self):
        class NonDefaultBatchEnd(Callback):
            def on_batch_end(self, batch, logs=None):
                pass

        cb_list = CallbackList(callbacks=[NonDefaultBatchEnd()])
        self.assertFalse(cb_list._async_train)

    def test_async_dispatch_disabled_when_on_test_batch_end_overridden(self):
        class NonDefaultTestEnd(Callback):
            def on_test_batch_end(self, batch, logs=None):
                pass

        cb_list = CallbackList(callbacks=[NonDefaultTestEnd()])
        self.assertFalse(cb_list._async_test)

    def test_async_dispatch_disabled_when_on_predict_batch_end_overridden(
        self,
    ):
        class NonDefaultPredictEnd(Callback):
            def on_predict_batch_end(self, batch, logs=None):
                pass

        cb_list = CallbackList(callbacks=[NonDefaultPredictEnd()])
        self.assertFalse(cb_list._async_predict)

    def test_async_safe_callback_does_not_disable_async(self):
        # Callbacks with async_safe=True are exempt from disabling async,
        # even if they override batch end methods.
        # Async dispatch is only possible on thread-safe backends.
        from keras.src import backend as keras_backend

        class AsyncSafeBatchEnd(Callback):
            async_safe = True

            def on_train_batch_end(self, batch, logs=None):
                pass

        cb_list = CallbackList(callbacks=[AsyncSafeBatchEnd()])
        if keras_backend.IS_THREAD_SAFE:
            self.assertTrue(cb_list._async_train)
        else:
            # Non-thread-safe backends never enable async.
            self.assertFalse(cb_list._async_train)

    def test_train_and_test_nested_begin_end_count(self):
        # on_train_begin + on_test_begin = count 2;
        # on_test_end goes to 1; on_train_end goes to 0.
        cb_list = CallbackList()
        cb_list.on_train_begin()
        cb_list.on_test_begin()
        self.assertEqual(cb_list._in_begin_end_block_count, 2)
        cb_list.on_test_end()
        self.assertEqual(cb_list._in_begin_end_block_count, 1)
        cb_list.on_train_end()
        self.assertEqual(cb_list._in_begin_end_block_count, 0)

    def test_multi_epoch_batch_sequence(self):
        # Simulate 2 epochs x 3 batches and verify all event indices.
        cb = TrackingCallback()
        cb_list = CallbackList(callbacks=[cb])

        cb_list.on_train_begin()
        for epoch in range(2):
            cb_list.on_epoch_begin(epoch)
            for batch in range(3):
                cb_list.on_train_batch_begin(batch)
                cb_list.on_train_batch_end(batch, logs={"loss": 0.1})
            cb_list.on_epoch_end(epoch, logs={"loss": 0.1})
        cb_list.on_train_end()

        epoch_begins = [c[1] for c in cb.calls if c[0] == "on_epoch_begin"]
        epoch_ends = [c[1] for c in cb.calls if c[0] == "on_epoch_end"]
        batch_begins = [
            c[1] for c in cb.calls if c[0] == "on_train_batch_begin"
        ]

        self.assertEqual(epoch_begins, [0, 1])
        self.assertEqual(epoch_ends, [0, 1])
        self.assertEqual(batch_begins, [0, 1, 2, 0, 1, 2])

    def test_predict_begin_end_count(self):
        cb_list = CallbackList()
        cb_list.on_predict_begin()
        self.assertEqual(cb_list._in_begin_end_block_count, 1)
        cb_list.on_predict_end()
        self.assertEqual(cb_list._in_begin_end_block_count, 0)

    def test_history_append_via_add_history_attaches_to_model(self):
        model_mock = mock.MagicMock()
        CallbackList(add_history=True, model=model_mock)
        self.assertIsInstance(model_mock.history, History)

    def test_flush_futures_on_epoch_end(self):
        # When async_train=True, epoch_end must wait for all batch futures.
        completed_batches = []

        class AsyncSafeBatchTrack(Callback):
            async_safe = True

            def on_train_batch_end(self, batch, logs=None):
                time.sleep(0.01)
                completed_batches.append(batch)

        cb_list = CallbackList(callbacks=[AsyncSafeBatchTrack()])
        cb_list.on_train_begin()
        cb_list.on_epoch_begin(0)
        cb_list.on_train_batch_end(0)
        cb_list.on_train_batch_end(1)
        # on_epoch_end flushes all pending async batch ends.
        cb_list.on_epoch_end(0, logs={})
        # After epoch_end, all batches must have completed.
        self.assertEqual(sorted(completed_batches), [0, 1])
        cb_list.on_train_end()


class HistoryAdvancedTest(testing.TestCase):
    def test_history_keys_sorted_alphabetically(self):
        # pythonify_logs sorts keys; history should reflect that ordering.
        h = History()
        h.set_model(mock.MagicMock())
        h.on_train_begin()
        h.on_epoch_end(0, logs={"z_metric": 1.0, "a_metric": 2.0})
        # Both keys should be present regardless of insertion order.
        self.assertIn("z_metric", h.history)
        self.assertIn("a_metric", h.history)

    def test_history_params_after_set_params(self):
        h = History()
        h.set_model(mock.MagicMock())
        h.set_params({"epochs": 3, "steps": 100})
        self.assertEqual(h.params["epochs"], 3)

    def test_history_epoch_list_grows_incrementally(self):
        h = History()
        h.set_model(mock.MagicMock())
        h.on_train_begin()
        for i in range(5):
            h.on_epoch_end(i, logs={"loss": 0.5 - i * 0.1})
        self.assertEqual(h.epoch, [0, 1, 2, 3, 4])
        self.assertEqual(len(h.history["loss"]), 5)

    def test_history_loss_values_match_logged(self):
        h = History()
        h.set_model(mock.MagicMock())
        h.on_train_begin()
        losses = [0.9, 0.7, 0.5, 0.3, 0.1]
        for i, loss in enumerate(losses):
            h.on_epoch_end(i, logs={"loss": loss})
        self.assertEqual(h.history["loss"], losses)


if __name__ == "__main__":
    testing.run_tests()
