import numpy as np
import openvino as ov
import openvino.runtime.opset14 as ov_opset

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import tree
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import get_device
from keras.src.backend.openvino.core import is_tensor
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils


class OpenVINOTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.test_function = None
        self.predict_function = None

    def test_step(self, data):
        (x, y, sample_weight,) = data_adapter_utils.unpack_x_y_sample_weight(data)
        y_pred = self(x)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        self._loss_tracker.update_state(loss, sample_weight=tree.flatten(x)[0].shape[0])
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        y_pred = self(x)
        return y_pred

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def one_test_step(data):
            data = data[0]
            return self.test_step(data)

        def multi_test_steps(data):
            for single_step_data in data:
                logs = one_test_step([single_step_data])
            return logs

        if self.steps_per_execution > 1:
            test_step = multi_test_steps
        else:
            test_step = one_test_step

        self.test_function = test_step

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        ov_inputs = []
        for _input in self._inputs:
            ov_type = OPENVINO_DTYPES[_input.dtype]
            ov_shape = _input.shape
            ov_shape = list(ov_shape)
            for i in range(len(ov_shape)):
                if ov_shape[i] is None:
                    ov_shape[i] = -1
            param = ov_opset.parameter(shape=ov_shape, dtype=ov_type)
            ov_inputs.append(param)
        # build OpenVINO graph ov.Model
        ov_outputs = self._run_through_graph(ov_inputs, operation_fn=lambda op: op)
        ov_outputs = tree.flatten(ov_outputs)
        ov_model = ov.Model(results=ov_outputs, parameters=ov_inputs)
        self.predict_function = ov.compile_model(ov_model, get_device())
        return self.predict_function

    def _symbolic_build(self, data_batch):
        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
                self._compile_metrics is not None
                and not self._compile_metrics.built
        )
        if model_unbuilt or compile_metrics_unbuilt:
            # Create symbolic tensors matching an input batch.

            def to_symbolic_input(v):
                if is_tensor(v):
                    return KerasTensor(v.shape, standardize_dtype(v.dtype))
                return v

            data_batch = tree.map_structure(to_symbolic_input, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            # Build all model state with `backend.compute_output_spec`.
            try:
                y_pred = backend.compute_output_spec(self, x)
            except:
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it."
                )
            if compile_metrics_unbuilt:
                # Build all metric state with `backend.compute_output_spec`.
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
        self._post_build()

    def fit(
            self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
    ):
        raise NotImplementedError(
            "fit not supported with openvino backend."
        )

    @traceback_utils.filter_traceback
    def predict(
            self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = EpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        def unpack_singleton(x):
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return x[0]
            return x

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for step, data in epoch_iterator.enumerate_epoch():
            callbacks.on_predict_batch_begin(step)
            flat_inputs = tree.flatten(data)
            batch_outputs = self.predict_function(flat_inputs)
            batch_outputs = unpack_singleton(tree.pack_sequence_as(self._outputs_struct, batch_outputs.to_tuple()))
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    @traceback_utils.filter_traceback
    def evaluate(
            self,
            x=None,
            y=None,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            return_dict=False,
            **kwargs,
    ):
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = EpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        if not all(layer.built for layer in self._flatten_layers()):
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch():
                data_batch = data[0]
                self._symbolic_build(data_batch)
                break

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()
        for step, data in epoch_iterator.enumerate_epoch():
            callbacks.on_test_batch_begin(step)
            logs = self.test_function(data)
            logs = self._pythonify_logs(logs)
            callbacks.on_test_batch_end(step, logs)
            if self.stop_evaluating:
                break
        logs = self._get_metrics_result_or_logs(logs)
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def train_on_batch(
            self,
            x,
            y=None,
            sample_weight=None,
            class_weight=None,
            return_dict=False,
    ):
        raise NotImplementedError(
            "train_on_batch not supported with openvino backend."
        )

    def test_on_batch(
            self,
            x,
            y=None,
            sample_weight=None,
            return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        data = (x, y, sample_weight)

        # Maybe build model
        self._symbolic_build(data)
        self.make_test_function()

        logs = self.test_function([data])
        logs = tree.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs
