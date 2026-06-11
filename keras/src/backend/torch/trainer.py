import contextlib
import warnings

import numpy as np
import torch
from packaging.version import parse

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend.torch import distribution_lib as torch_dist_lib
from keras.src.distribution import distribution_lib as dist_lib
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
        self._in_ddp_context = False
        self._current_dist = None

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

    def _setup_ddp(self):
        dist = dist_lib.distribution()
        if dist is not self._current_dist:
            self.train_function = None
            self.test_function = None
            self.predict_function = None
            self._current_dist = dist

        if dist is not None and isinstance(dist, dist_lib.DataParallel):
            if not hasattr(self, "_ddp_model"):
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{torch.cuda.current_device()}")
                    device_ids = [torch.cuda.current_device()]
                else:
                    device = torch.device("cpu")
                    device_ids = None

                # Check for mixed placement
                device_types = set()
                for v in self.variables:
                    device_types.add(v.value.device.type)
                if len(device_types) > 1:
                    warnings.warn(
                        "Mixed device placement detected before DDP setup. "
                        f"Devices found: {device_types}. The model will be "
                        f"moved to {device}."
                    )

                # Move model to the target device before DDP wrapping.
                # DDP requires all parameters to be on the same device.
                self.to(device)

                # Explicitly move Variables that might not be tracked by
                # TorchLayer or have DTensor issues.
                for v in self.variables:
                    if v.value.device != device:
                        v.assign(v.value.to(device))

                # Move optimizer state.
                if self.optimizer is not None:
                    for v in self.optimizer.variables:
                        if v.value.device != device:
                            v.assign(v.value.to(device))

                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    self,
                    device_ids=device_ids,
                )
                object.__setattr__(self, "_ddp_model", ddp_model)
            self._in_ddp_context = True
        else:
            if hasattr(self, "_ddp_model"):
                object.__delattr__(self, "_ddp_model")
            self._in_ddp_context = False

    def _distribute_data(self, data, replicate=False):
        from keras.src.distribution import distribution_lib

        dist = distribution_lib.distribution()
        if dist is not None:

            def _distribute_if_tensor(t):
                if (
                    backend.is_tensor(t) or isinstance(t, np.ndarray)
                ) and hasattr(t, "shape"):
                    layout = dist.get_data_layout(t.shape)
                    if replicate and isinstance(
                        dist, distribution_lib.ModelParallel
                    ):
                        from keras.src.distribution import TensorLayout

                        layout = TensorLayout(
                            [None] * len(t.shape), dist.device_mesh
                        )
                    return torch_dist_lib.distribute_data_input(
                        t, layout, dist.batch_dim_name
                    )
                return t

            return tree.map_structure(
                _distribute_if_tensor, data, none_is_leaf=False
            )
        return tree.map_structure(
            backend.convert_to_tensor, data, none_is_leaf=False
        )

    def _unpack_and_distribute_data(self, data):
        from keras.src.distribution import distribution_lib as dist_lib

        dist = dist_lib.distribution()
        data = self._distribute_data(data)
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        return dist_lib, dist, x, y, sample_weight

    def _forward(self, dist, x, training=False):
        from keras.src.distribution import distribution_lib as dist_lib

        if dist is not None and isinstance(dist, dist_lib.DataParallel):
            return self._ddp_model(x, training=training)
        if self._call_has_training_arg:
            return self(x, training=training)
        return self(x)

    def _sync_ddp_buffers(self, dist):
        from keras.src.distribution import distribution_lib as dist_lib

        if dist is not None and isinstance(dist, dist_lib.DataParallel):
            module = getattr(self._ddp_model, "module", None)
            if module is not None:
                aggregation_to_op = {
                    "sum": torch.distributed.ReduceOp.SUM,
                    "mean": torch.distributed.ReduceOp.SUM,
                    "max": torch.distributed.ReduceOp.MAX,
                    "min": torch.distributed.ReduceOp.MIN,
                }
                for name, buf in module.named_buffers(recurse=False):
                    if buf is None:
                        continue
                    if not buf.requires_grad:
                        if torch.distributed.is_initialized():
                            aggregation = module._buffer_aggregations.get(name)
                            if aggregation in (None, "none"):
                                # Unknown aggregation — broadcast from rank 0
                                # (safe default).
                                torch.distributed.broadcast(buf, src=0)
                                continue
                            if aggregation not in aggregation_to_op:
                                continue
                            reduce_op = aggregation_to_op[aggregation]
                            torch.distributed.all_reduce(buf, op=reduce_op)
                            if aggregation == "mean":
                                buf.div_(torch.distributed.get_world_size())

    def _get_ddp_sync_context(self, dist, dist_lib):
        if dist is not None and isinstance(dist, dist_lib.DataParallel):
            is_update_step = True
            if (
                self.optimizer is not None
                and self.optimizer.gradient_accumulation_steps
            ):
                is_update_step = (
                    self.optimizer._iterations.value + 1
                ) % self.optimizer.gradient_accumulation_steps == 0

            if not is_update_step:
                return self._ddp_model.no_sync()
        return contextlib.nullcontext()

    def _set_train_mode(self, training=True):
        if training:
            if self._in_ddp_context:
                self._ddp_model.train()
            else:
                self.train()
        else:
            if self._in_ddp_context:
                self._ddp_model.eval()
            else:
                self.eval()

    def train_step(self, data):
        (
            dist_lib,
            dist,
            x,
            y,
            sample_weight,
        ) = self._unpack_and_distribute_data(data)

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()
        y_pred = self._forward(dist, x, training=True)
        self._sync_ddp_buffers(dist)

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
            context = self._get_ddp_sync_context(dist, dist_lib)

            with context:
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
            dist_lib,
            dist,
            x,
            y,
            sample_weight,
        ) = self._unpack_and_distribute_data(data)

        y_pred = self._forward(dist, x, training=False)
        self._sync_ddp_buffers(dist)

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
        (
            dist_lib,
            dist,
            x,
            _,
            _,
        ) = self._unpack_and_distribute_data(data)

        y_pred = self._forward(dist, x, training=False)
        self._sync_ddp_buffers(dist)
        return y_pred

    def _sync_metrics(self):
        dist_obj = dist_lib.distribution()
        if dist_obj is not None and torch.distributed.is_initialized():
            aggregation_to_op = {
                "sum": torch.distributed.ReduceOp.SUM,
                "mean": torch.distributed.ReduceOp.SUM,
                "max": torch.distributed.ReduceOp.MAX,
                "min": torch.distributed.ReduceOp.MIN,
            }
            from torch.distributed.tensor import DTensor

            for metric in self.metrics:
                for variable in metric.variables:
                    if variable.aggregation in (None, "none"):
                        continue
                    is_dtensor = DTensor is not None and isinstance(
                        variable.value, DTensor
                    )
                    v = variable.value
                    if is_dtensor:
                        v = v.to_local()
                    if variable.aggregation == "only_first_replica":
                        torch.distributed.broadcast(v, src=0)
                    else:
                        reduce_op = aggregation_to_op.get(
                            variable.aggregation, torch.distributed.ReduceOp.SUM
                        )
                        torch.distributed.all_reduce(v, op=reduce_op)
                        if variable.aggregation == "mean":
                            v = v / torch.distributed.get_world_size()
                    variable.assign(v)

    def make_train_function(self, force=False):
        self._setup_ddp()
        if self.train_function is not None and not force:
            return self.train_function

        if self.steps_per_execution > 1:
            raise ValueError(
                "`steps_per_execution` must be 1 with the PyTorch backend. "
                f"Received: steps_per_execution={self.steps_per_execution}"
            )

        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            data = data[0]
            return self.train_step(data)

        if self._should_torch_compile():
            self.train_function = torch.compile(one_step_on_data)
        else:
            self.train_function = one_step_on_data

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        if self.steps_per_execution > 1:
            raise ValueError(
                "`steps_per_execution` must be 1 with the PyTorch backend. "
                f"Received: steps_per_execution={self.steps_per_execution}"
            )

        self._setup_ddp()

        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            data = data[0]
            with torch.no_grad():
                return self.test_step(data)

        if self._should_torch_compile():
            self.test_function = torch.compile(one_step_on_data)
        else:
            self.test_function = one_step_on_data

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        if self.steps_per_execution > 1:
            raise ValueError(
                "`steps_per_execution` must be 1 with the PyTorch backend. "
                f"Received: steps_per_execution={self.steps_per_execution}"
            )

        self._setup_ddp()

        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            data = data[0]
            with torch.no_grad():
                return self.predict_step(data)

        if self._should_torch_compile():
            self.predict_function = torch.compile(one_step_on_data)
        else:
            self.predict_function = one_step_on_data

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
            self._set_train_mode(True)

            logs = {}
            for begin_step, end_step, data in epoch_iterator:
                # Callbacks
                callbacks.on_train_batch_begin(begin_step)

                logs = self.train_function(data)

                # Callbacks
                callbacks.on_train_batch_end(end_step, logs)
                if self.stop_training:
                    break

            # Override with model metrics instead of last step logs if needed.
            self._sync_metrics()
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Switch the torch Module back to testing mode.
            self._set_train_mode(False)

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
        self._set_train_mode(False)

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
        self._set_train_mode(False)

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

        self._set_train_mode(True)

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

        self._set_train_mode(False)

        logs = self.test_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()

        self._set_train_mode(False)

        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs


class TorchEpochIterator(EpochIterator):
    def _get_iterator(self):
        return self.data_adapter.get_torch_dataloader()
