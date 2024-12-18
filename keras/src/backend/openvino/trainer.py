import numpy as np
import openvino as ov
import openvino.runtime.opset14 as ov_opset

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import tree
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_device
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils


class OpenVINOTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.test_function = None
        self.predict_function = None
        self.ov_compiled_model = None
        self.ov_device = None
        self.struct_params = None
        self.struct_outputs = None

    def _unpack_singleton(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]
        return x

    def test_step(self, data):
        raise NotImplementedError(
            "`test_step` is not supported with openvino backend"
        )

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        ov_compiled_model = self._get_compiled_model(x)
        flatten_x = tree.flatten(x)
        y_pred = ov_compiled_model(flatten_x)
        # recover structure of the model output
        y_pred = self._unpack_singleton(
            tree.pack_sequence_as(self.struct_outputs, y_pred.to_tuple())
        )
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

    def _parameterize_data(self, data):
        if isinstance(data, (list, tuple)):
            parametrize_data = []
            for elem in data:
                param_elem = self._parameterize_data(elem)
                parametrize_data.append(param_elem)
        elif isinstance(data, dict):
            parametrize_data = dict()
            for elem_name, elem in data.items():
                param_elem = self._parameterize_data(elem)
                parametrize_data[elem_name] = param_elem
        elif isinstance(data, np.ndarray) or np.isscalar(data):
            ov_type = OPENVINO_DTYPES[str(data.dtype)]
            ov_shape = list(data.shape)
            param = ov_opset.parameter(shape=ov_shape, dtype=ov_type)
            parametrize_data = OpenVINOKerasTensor(param.output(0))
        elif isinstance(data, int):
            param = ov_opset.parameter(shape=[], dtype=ov.Type.i32)
            parametrize_data = OpenVINOKerasTensor(param.output(0))
        elif isinstance(data, float):
            param = ov_opset.parameter(shape=[], dtype=ov.Type.f32)
            parametrize_data = OpenVINOKerasTensor(param.output(0))
        else:
            raise "Unknown type of input data {}".format(type(data))
        return parametrize_data

    def _get_compiled_model(self, data):
        if (
            self.ov_compiled_model is not None
            and get_device() == self.ov_device
        ):
            return self.ov_compiled_model

        # remove the previous cached compiled model if exists
        del self.ov_compiled_model

        # prepare parameterized input
        self.struct_params = self._parameterize_data(data)
        # construct OpenVINO graph during calling Keras Model
        self.struct_outputs = self(self.struct_params)

        parameters = []
        for p in tree.flatten(self.struct_params):
            parameters.append(p.output.get_node())
        results = []
        for r in tree.flatten(self.struct_outputs):
            results.append(ov_opset.result(r.output))

        # prepare compiled model from scratch
        ov_model = ov.Model(results=results, parameters=parameters)
        self.ov_compiled_model = ov.compile_model(ov_model, get_device())
        self.ov_device = get_device()
        return self.ov_compiled_model

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def one_predict_step(data):
            data = data[0]
            return self.predict_step(data)

        def multi_predict_steps(data):
            outputs = one_predict_step(data[:1])

            for single_step_data in data[1:]:
                step_outputs = one_predict_step([single_step_data])
                outputs = tree.map_structure(
                    lambda t1, t2: np.concatenate([t1, t2]),
                    outputs,
                    step_outputs,
                )
            return outputs

        if self.steps_per_execution > 1:
            predict_step = multi_predict_steps
        else:
            predict_step = one_predict_step

        self.predict_function = predict_step

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
            "`fit` is not supported with openvino backend"
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

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for step, data in epoch_iterator.enumerate_epoch():
            callbacks.on_predict_batch_begin(step)
            batch_outputs = self.predict_function(data)
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
        raise NotImplementedError(
            "`evaluate` is not supported with openvino backend"
        )

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        raise NotImplementedError(
            "`train_on_batch` is not supported with openvino backend"
        )

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        raise NotImplementedError(
            "`test_on_batch` is not supported with openvino backend"
        )

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs
