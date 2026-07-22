import warnings

import numpy as np
import torch
from packaging.version import parse
from torch.nn.parallel import DistributedDataParallel

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.distribution_lib import _to_backend_mesh
from keras.src.distribution.distribution_lib import distribution
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
from keras.src.utils.python_utils import pythonify_logs


class TorchTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def _should_torch_compile(self):
        # require torch>=2.1.0 to enable dynamo since it
        # includes many improvements/fixes to torch.compile()
        # TODO eventually we want to get rid of this when
        # torch is upgraded to >=2.1 (from 2.0.1) in g3
        if self.jit_compile and parse(torch.__version__) < parse("2.1.0"):
            warnings.warn(
                "Please upgrade to torch>=2.1.0 for `jit_compile=True` "
                "to take effect. Using `jit_compile=False`"
            )
            self.jit_compile = False

        return self.jit_compile

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Compute predictions
        model = getattr(self, "ddp_model", self)
        if self._call_has_training_arg:
            y_pred = model(x, training=True)
        else:
            y_pred = model(x)

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True
        )
        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        if self.optimizer is not None:
            loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            # Call torch.Tensor.backward() on the loss to compute gradients
            # for the weights.
            loss.backward()

            trainable_weights = self.trainable_weights[:]
            gradients = [v.value.grad for v in trainable_weights]

            # Update weights
            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        (
            x,
            y,
            sample_weight,
        ) = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        if torch.distributed.is_initialized() and not hasattr(
            self, "ddp_model"
        ):
            device = get_device()
            if str(device).startswith("cuda"):
                if ":" in str(device):
                    device_ids = [int(str(device).split(":")[-1])]
                else:
                    device_ids = [torch.cuda.current_device()]
            else:
                device_ids = None

            active_distribution = distribution()
            process_group = None
            if active_distribution is not None:
                backend_mesh = _to_backend_mesh(active_distribution.device_mesh)
                # get_group expects the axis name
                process_group = backend_mesh.get_group(
                    active_distribution.batch_dim_name
                )

            # Set find_unused_parameters=False by default to avoid hangs
            # and overhead. It can be made configurable if needed.
            object.__setattr__(
                self,
                "ddp_model",
                DistributedDataParallel(
                    self,
                    device_ids=device_ids,
                    process_group=process_group,
                    find_unused_parameters=False,
                ),
            )

        train_step = self.train_step
        if self._should_torch_compile():
            train_step = torch.compile(train_step)

        def train_function(data):
            """Runs training steps on a list of batches of data."""
            logs = {}
            for step_data in data:
                logs = train_step(step_data)
            return logs

        self.train_function = train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        test_step = self.test_step
        if self._should_torch_compile():
            test_step = torch.compile(test_step)

        def test_function(data):
            """Runs test steps on a list of batches of data."""
            logs = {}
            with torch.no_grad():
                for step_data in data:
                    logs = test_step(step_data)
            return logs

        self.test_function = test_function

    def _sync_metrics(self):
        if torch.distributed.is_initialized():
            import torch.distributed as dist

            from keras.src.distribution.distribution_lib import distribution

            active_distribution = distribution()
            process_group = None
            if active_distribution is not None:
                from keras.src.backend.torch.distribution_lib import (
                    _to_backend_mesh,
                )

                backend_mesh = _to_backend_mesh(active_distribution.device_mesh)
                process_group = backend_mesh.get_group(
                    active_distribution.batch_dim_name
                )

            with torch.no_grad():
                for metric in self.metrics:
                    for v in metric.variables:
                        val = getattr(v, "_value", None)
                        if val is not None:
                            tensor = (
                                val.to_local()
                                if hasattr(val, "to_local")
                                else val
                            )
                            backend_name = (
                                dist.get_backend(process_group)
                                if process_group is not None
                                else dist.get_backend()
                            )
                            if (
                                backend_name == "nccl"
                                and tensor.device.type == "cpu"
                            ):
                                cuda_device = torch.device(
                                    f"cuda:{torch.cuda.current_device()}"
                                )
                                cuda_tensor = tensor.to(cuda_device)
                                dist.all_reduce(
                                    cuda_tensor,
                                    op=dist.ReduceOp.SUM,
                                    group=process_group,
                                )
                                tensor.copy_(cuda_tensor.to("cpu"))
                            else:
                                dist.all_reduce(
                                    tensor,
                                    op=dist.ReduceOp.SUM,
                                    group=process_group,
                                )

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        predict_step = self.predict_step
        if self._should_torch_compile():
            predict_step = torch.compile(predict_step)

        def predict_function(data):
            """Runs predict steps on a list of batches of data."""
            outputs = []
            with torch.no_grad():
                for step_data in data:
                    outputs.append(predict_step(step_data))

            def concat_outputs(outputs):
                if not outputs:
                    return []
                if len(outputs) == 1:
                    return outputs[0]
                return tree.map_structure(
                    lambda *args: torch.cat(args, dim=0),
                    *outputs,
                )

            return concat_outputs(outputs)

        self.predict_function = predict_function

    @traceback_utils.filter_traceback
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
        self._assert_compile_called("fit")
        # Possibly cap epochs for debugging runs.
        max_epochs = config.max_epochs()
        if max_epochs and max_epochs < epochs:
            warnings.warn("Limiting epochs to %d" % max_epochs)
            epochs = max_epochs

        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            # TODO: Support torch tensors for validation data.
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TorchEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        training_logs = {}
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            # Switch the torch Module to training mode. Inform torch layers to
            # do training behavior in case the user did not use `self.training`
            # when implementing a custom layer with torch layers.
            self.train()

            logs = {}
            for begin_step, end_step, data in epoch_iterator:
                # Callbacks
                callbacks.on_train_batch_begin(begin_step)

                logs = self.train_function(data)

                # Callbacks
                callbacks.on_train_batch_end(end_step, logs)
                if self.stop_training:
                    break

            # Synchronize metrics across ranks
            self._sync_metrics()

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Switch the torch Module back to testing mode.
            self.eval()

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create TorchEpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TorchEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    f"val_{name}": val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

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
            epoch_iterator = TorchEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_test_batch_begin(begin_step)
            logs = self.test_function(data)
            callbacks.on_test_batch_end(end_step, logs)
            if self.stop_evaluating:
                break

        # Synchronize metrics across ranks
        self._sync_metrics()

        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = TorchEpochIterator(
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

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_predict_batch_begin(begin_step)
            batch_outputs = self.predict_function(data)
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        outputs = tree.map_structure(backend.convert_to_numpy, outputs)
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        data = (x, y, sample_weight)

        # Maybe build model
        self._symbolic_build(data_batch=data)
        self.make_train_function()
        self.reset_metrics()

        logs = self.train_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

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
        self._symbolic_build(data_batch=data)
        self.make_test_function()
        self.reset_metrics()

        logs = self.test_function([data])
        logs = pythonify_logs(logs)
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


class TorchEpochIterator(EpochIterator):
    def _get_iterator(self):
        return self.data_adapter.get_torch_dataloader()
