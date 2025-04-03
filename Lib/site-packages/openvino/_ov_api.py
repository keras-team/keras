# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Any, Iterable, Union, Optional, Dict, Tuple, Type, List
from pathlib import Path


from openvino._pyopenvino import Model as ModelBase
from openvino._pyopenvino import Core as CoreBase
from openvino._pyopenvino import CompiledModel as CompiledModelBase
from openvino._pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino._pyopenvino import Op as OpBase
from openvino._pyopenvino import Node, Output, Tensor

from openvino.utils.data_helpers import (
    OVDict,
    _InferRequestWrapper,
    _data_dispatch,
    tensor_from_file,
)


class Op(OpBase):
    def __init__(self, py_obj: "Op", inputs: Optional[Union[List[Union[Node, Output]], Tuple[Union[Node, Output, List[Union[Node, Output]]]]]] = None) -> None:
        super().__init__(py_obj)
        self._update_type_info()
        if isinstance(inputs, tuple):
            inputs = None if len(inputs) == 0 else list(inputs)
            if inputs is not None and len(inputs) == 1 and isinstance(inputs[0], list):
                inputs = inputs[0]
        if inputs is not None:
            self.set_arguments(inputs)
            self.constructor_validate_and_infer_types()


class ModelMeta(type):
    def __dir__(cls) -> list:
        return list(set(cls.__dict__.keys()) | set(dir(ModelBase)))


class Model(object, metaclass=ModelMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and not kwargs:
            if isinstance(args[0], ModelBase):
                self.__model = ModelBase(args[0])
            elif isinstance(args[0], Node):
                self.__model = ModelBase(*args)
            else:
                self.__model = ModelBase(*args)
        if args and kwargs:
            self.__model = ModelBase(*args, **kwargs)
        if kwargs and not args:
            self.__model = ModelBase(**kwargs)

    def __getattr__(self, name: str) -> Any:
        if self.__model is None:
            raise AttributeError(f"'Model' object has no attribute '{name}' or attribute is no longer accessible.")
        return getattr(self.__model, name)

    def clone(self) -> "Model":
        return Model(self.__model.clone())

    def __copy__(self) -> "Model":
        raise TypeError("Cannot copy 'openvino.Model'. Please, use deepcopy instead.")

    def __deepcopy__(self, memo: Dict) -> "Model":
        """Returns a deepcopy of Model.

        :return: A copy of Model.
        :rtype: openvino.Model
        """
        return Model(self.__model.clone())

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, exc_type: Type[BaseException], exc_value: BaseException, traceback: TracebackType) -> None:
        del self.__model
        self.__model = None

    def __repr__(self) -> str:
        return self.__model.__repr__()

    def __dir__(self) -> list:
        wrapper_methods = ["__copy__", "__deepcopy__", "__dict__", "__enter__", "__exit__", "__getattr__", "__weakref__"]
        return dir(self.__model) + wrapper_methods


class InferRequest(_InferRequestWrapper):
    """InferRequest class represents infer request which can be run in asynchronous or synchronous manners."""

    def infer(
        self,
        inputs: Any = None,
        share_inputs: bool = False,
        share_outputs: bool = False,
        *,
        decode_strings: bool = True,
    ) -> OVDict:
        """Infers specified input(s) in synchronous mode.

        Blocks all methods of InferRequest while request is running.
        Calling any method will lead to throwing exceptions.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
        (2) `openvino.Tensor`

        Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Any, optional
        :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.

                              If set to `False` inputs the data dispatcher will safely copy data
                              to existing Tensors (including up- or down-casting according to data type,
                              resizing of the input Tensor). Keeps Tensor inputs "as-is".

                              If set to `True` the data dispatcher tries to provide "zero-copy"
                              Tensors for every input in form of:
                              * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                              Data that is going to be copied:
                              * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                              * inputs which data types are mismatched from Infer Request's inputs
                              * inputs that should be in `BF16` data type
                              * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                              * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                              Keeps Tensor inputs "as-is".

                              Note: Use with extra care, shared data can be modified during runtime!
                              Note: Using `share_inputs` may result in extra memory overhead.

                              Default value: False
        :type share_inputs: bool, optional
        :param share_outputs: Enables `share_outputs` mode. Controls memory usage on inference's outputs.

                              If set to `False` outputs will safely copy data to numpy arrays.

                              If set to `True` the data will be returned in form of views of output Tensors.
                              This mode still returns the data in format of numpy arrays but lifetime of the data
                              is connected to OpenVINO objects.

                              Note: Use with extra care, shared data can be modified or lost during runtime!
                              Note: String/textual data will always be copied!

                              Default value: False
        :type share_outputs: bool, optional
        :param decode_strings: Controls decoding outputs of textual based data.

                               If set to `True` string outputs will be returned as numpy arrays of `U` kind.

                               If set to `False` string outputs will be returned as numpy arrays of `S` kind.

                               Default value: True
        :type decode_strings: bool, optional, keyword-only

        :return: Dictionary of results from output tensors with port/int/str keys.
        :rtype: OVDict
        """
        return OVDict(super().infer(_data_dispatch(
            self,
            inputs,
            is_shared=share_inputs,
        ), share_outputs=share_outputs, decode_strings=decode_strings))

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        share_inputs: bool = False,
    ) -> None:
        """Starts inference of specified input(s) in asynchronous mode.

        Returns immediately. Inference starts also immediately.
        Calling any method on the `InferRequest` object while the request is running
        will lead to throwing exceptions.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
        (2) `openvino.Tensor`

        Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Any, optional
        :param userdata: Any data that will be passed inside the callback.
        :type userdata: Any
        :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.

                              If set to `False` inputs the data dispatcher will safely copy data
                              to existing Tensors (including up- or down-casting according to data type,
                              resizing of the input Tensor). Keeps Tensor inputs "as-is".

                              If set to `True` the data dispatcher tries to provide "zero-copy"
                              Tensors for every input in form of:
                              * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                              Data that is going to be copied:
                              * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                              * inputs which data types are mismatched from Infer Request's inputs
                              * inputs that should be in `BF16` data type
                              * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                              * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                              Keeps Tensor inputs "as-is".

                              Note: Use with extra care, shared data can be modified during runtime!
                              Note: Using `share_inputs` may result in extra memory overhead.

                              Default value: False
        :type share_inputs: bool, optional
        """
        super().start_async(
            _data_dispatch(
                self,
                inputs,
                is_shared=share_inputs,
            ),
            userdata,
        )

    def get_compiled_model(self) -> "CompiledModel":
        """Gets the compiled model this InferRequest is using.

        :return: a CompiledModel object
        :rtype: openvino.CompiledModel
        """
        return CompiledModel(super().get_compiled_model())

    @property
    def results(self) -> OVDict:
        """Gets all outputs tensors of this InferRequest.

        :return: Dictionary of results from output tensors with ports as keys.
        :rtype: Dict[openvino.ConstOutput, numpy.array]
        """
        return OVDict(super().results)


class CompiledModel(CompiledModelBase):
    """CompiledModel class.

    CompiledModel represents Model that is compiled for a specific device by applying
    multiple optimization transformations, then mapping to compute kernels.
    """

    def __init__(self, other: CompiledModelBase, weights: Optional[bytes] = None) -> None:
        # Private memeber to store already created InferRequest
        self._infer_request: Optional[InferRequest] = None
        self._weights = weights
        super().__init__(other)

    def get_runtime_model(self) -> Model:
        return Model(super().get_runtime_model())

    def create_infer_request(self) -> InferRequest:
        """Creates an inference request object used to infer the compiled model.

        The created request has allocated input and output tensors.

        :return: New InferRequest object.
        :rtype: openvino.InferRequest
        """
        return InferRequest(super().create_infer_request())

    def query_state(self) -> None:
        """Gets state control interface for the underlaying infer request.

        :return: List of VariableState objects.
        :rtype: List[openvino.VariableState]
        """
        if self._infer_request is None:
            self._infer_request = self.create_infer_request()

        return self._infer_request.query_state()

    def reset_state(self) -> None:
        """Resets all internal variable states of the underlaying infer request.

        Resets all internal variable states to a value specified as default for
        the corresponding `ReadValue` node.
        """
        if self._infer_request is None:
            self._infer_request = self.create_infer_request()

        return self._infer_request.reset_state()

    def infer_new_request(self, inputs: Any = None) -> OVDict:
        """Infers specified input(s) in synchronous mode.

        Blocks all methods of CompiledModel while request is running.

        Method creates new temporary InferRequest and run inference on it.
        It is advised to use a dedicated InferRequest class for performance,
        optimizing workflows, and creating advanced pipelines.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
        (2) `openvino.Tensor`

        Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Any, optional
        :return: Dictionary of results from output tensors with port/int/str keys.
        :rtype: OVDict
        """
        # It returns wrapped python InferReqeust and then call upon
        # overloaded functions of InferRequest class
        return self.create_infer_request().infer(inputs)

    def __call__(
        self,
        inputs: Any = None,
        share_inputs: bool = True,
        share_outputs: bool = False,
        *,
        decode_strings: bool = True,
    ) -> OVDict:
        """Callable infer wrapper for CompiledModel.

        Infers specified input(s) in synchronous mode.

        Blocks all methods of CompiledModel while request is running.

        Method creates new temporary InferRequest and run inference on it.
        It is advised to use a dedicated InferRequest class for performance,
        optimizing workflows, and creating advanced pipelines.

        This method stores created `InferRequest` inside `CompiledModel` object,
        which can be later reused in consecutive calls.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
        (2) `openvino.Tensor`

        Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors.
        :type inputs: Any, optional
        :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.

                              If set to `False` inputs the data dispatcher will safely copy data
                              to existing Tensors (including up- or down-casting according to data type,
                              resizing of the input Tensor). Keeps Tensor inputs "as-is".

                              If set to `True` the data dispatcher tries to provide "zero-copy"
                              Tensors for every input in form of:
                              * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                              Data that is going to be copied:
                              * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                              * inputs which data types are mismatched from Infer Request's inputs
                              * inputs that should be in `BF16` data type
                              * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                              * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                              Keeps Tensor inputs "as-is".

                              Note: Use with extra care, shared data can be modified during runtime!
                              Note: Using `share_inputs` may result in extra memory overhead.

                              Default value: True
        :type share_inputs: bool, optional
        :param share_outputs: Enables `share_outputs` mode. Controls memory usage on inference's outputs.

                              If set to `False` outputs will safely copy data to numpy arrays.

                              If set to `True` the data will be returned in form of views of output Tensors.
                              This mode still returns the data in format of numpy arrays but lifetime of the data
                              is connected to OpenVINO objects.

                              Note: Use with extra care, shared data can be modified or lost during runtime!
                              Note: String/textual data will always be copied!

                              Default value: False
        :type share_outputs: bool, optional
        :param decode_strings: Controls decoding outputs of textual based data.

                               If set to `True` string outputs will be returned as numpy arrays of `U` kind.

                               If set to `False` string outputs will be returned as numpy arrays of `S` kind.

                               Default value: True
        :type decode_strings: bool, optional, keyword-only

        :return: Dictionary of results from output tensors with port/int/str as keys.
        :rtype: OVDict
        """
        if self._infer_request is None:
            self._infer_request = self.create_infer_request()

        return self._infer_request.infer(
            inputs,
            share_inputs=share_inputs,
            share_outputs=share_outputs,
            decode_strings=decode_strings,
        )


class AsyncInferQueue(AsyncInferQueueBase):
    """AsyncInferQueue with a pool of asynchronous requests.

    AsyncInferQueue represents a helper that creates a pool of asynchronous
    InferRequests and provides synchronization functions to control flow of
    a simple pipeline.
    """

    def __iter__(self) -> Iterable[InferRequest]:
        """Allows to iterate over AsyncInferQueue.

        Resulting objects are guaranteed to work with read-only methods like getting tensors.
        Any mutating methods (e.g. start_async, set_callback) of a single request
        will put the parent AsyncInferQueue object in an invalid state.

        :return: a generator that yields InferRequests.
        :rtype: Iterable[openvino.InferRequest]
        """
        return (InferRequest(x) for x in super().__iter__())

    def __getitem__(self, i: int) -> InferRequest:
        """Gets InferRequest from the pool with given i id.

        Resulting object is guaranteed to work with read-only methods like getting tensors.
        Any mutating methods (e.g. start_async, set_callback) of a request
        will put the parent AsyncInferQueue object in an invalid state.

        :param i:  InferRequest id.
        :type i: int
        :return: InferRequests from the pool with given id.
        :rtype: openvino.InferRequest
        """
        return InferRequest(super().__getitem__(i))

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        share_inputs: bool = False,
    ) -> None:
        """Run asynchronous inference using the next available InferRequest from the pool.

        The allowed types of keys in the `inputs` dictionary are:

        (1) `int`
        (2) `str`
        (3) `openvino.ConstOutput`

        The allowed types of values in the `inputs` are:

        (1) `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
        (2) `openvino.Tensor`

        Can be called with only one `openvino.Tensor` or `numpy.ndarray`,
        it will work only with one-input models. When model has more inputs,
        function throws error.

        :param inputs: Data to be set on input tensors of the next available InferRequest.
        :type inputs: Any, optional
        :param userdata: Any data that will be passed to a callback.
        :type userdata: Any, optional
        :param share_inputs: Enables `share_inputs` mode. Controls memory usage on inference's inputs.

                              If set to `False` inputs the data dispatcher will safely copy data
                              to existing Tensors (including up- or down-casting according to data type,
                              resizing of the input Tensor). Keeps Tensor inputs "as-is".

                              If set to `True` the data dispatcher tries to provide "zero-copy"
                              Tensors for every input in form of:
                              * `numpy.ndarray` and all the types that are castable to it, e.g. `torch.Tensor`
                              Data that is going to be copied:
                              * `numpy.ndarray` which are not C contiguous and/or not writable (WRITEABLE flag is set to False)
                              * inputs which data types are mismatched from Infer Request's inputs
                              * inputs that should be in `BF16` data type
                              * scalar inputs (i.e. `np.float_`/`str`/`bytes`/`int`/`float`)
                              * lists of simple data types (i.e. `str`/`bytes`/`int`/`float`)
                              Keeps Tensor inputs "as-is".

                              Note: Use with extra care, shared data can be modified during runtime!
                              Note: Using `share_inputs` may result in extra memory overhead.

                              Default value: False
        :type share_inputs: bool, optional
        """
        super().start_async(
            _data_dispatch(
                self[self.get_idle_request_id()],
                inputs,
                is_shared=share_inputs,
            ),
            userdata,
        )


class Core(CoreBase):
    """Core class represents OpenVINO runtime Core entity.

    User applications can create several Core class instances, but in this
    case, the underlying plugins are created multiple times and not shared
    between several Core instances. The recommended way is to have a single
    Core instance per application.
    """
    def read_model(
        self,
        model: Union[str, bytes, object],
        weights: Union[object, str, bytes, Tensor] = None,
        config: Optional[dict] = None
    ) -> Model:
        config = {} if config is None else config
        if isinstance(model, Model):
            model = model._Model__model

        if isinstance(weights, Tensor):
            return Model(super().read_model(model, weights))
        elif isinstance(model, bytes):
            return Model(super().read_model(model, bytes() if weights is None else weights))
        elif weights is None:
            return Model(super().read_model(model, config=config))
        else:
            return Model(super().read_model(model, weights, config))

    def compile_model(
        self,
        model: Union[Model, str, Path],
        device_name: Optional[str] = None,
        config: Optional[dict] = None,
        *,
        weights: Optional[bytes] = None,
    ) -> CompiledModel:
        """Creates a compiled model.

        Creates a compiled model from a source Model object or
        reads model and creates a compiled model from IR / ONNX / PDPD / TF and TFLite file or
        creates a compiled model from a IR xml and weights in memory.
        This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
        especially for cases when caching is enabled and cached model is available.
        If device_name is not specified, the default OpenVINO device will be selected by AUTO plugin.
        Users can create as many compiled models as they need, and use them simultaneously
        (up to the limitation of the hardware resources).

        :param model: Model acquired from read_model function or a path to a model in IR / ONNX / PDPD /
                      TF and TFLite format.
        :type model: Union[openvino.Model, str, pathlib.Path]
        :param device_name: Optional. Name of the device to load the model to. If not specified,
                            the default OpenVINO device will be selected by AUTO plugin.
        :type device_name: str
        :param config: Optional dict of pairs:
                       (property name, property value) relevant only for this load operation.
        :type config: dict, optional
        :param weights: Optional. Weights of model in memory to be loaded to the model.
        :type weights: bytes, optional, keyword-only
        :return: A compiled model.
        :rtype: openvino.CompiledModel
        """
        if isinstance(model, Model):
            model = model._Model__model
        if weights is None:
            if device_name is None:
                return CompiledModel(
                    super().compile_model(model, {} if config is None else config),
                )
            return CompiledModel(
                super().compile_model(model, device_name, {} if config is None else config),
            )
        else:
            if device_name is None:
                return CompiledModel(
                    super().compile_model(model, weights, {} if config is None else config),
                    weights=weights,
                )
            return CompiledModel(
                super().compile_model(model, weights, device_name, {} if config is None else config),
                weights=weights,
            )

    def query_model(
            self,
            model: Model,
            device_name: str,
            config: Optional[dict] = None,
    ) -> dict:
        return super().query_model(model._Model__model,
                                   device_name,
                                   {} if config is None else config, )

    def import_model(
        self,
        model_stream: bytes,
        device_name: str,
        config: Optional[dict] = None,
    ) -> CompiledModel:
        """Imports a compiled model from a previously exported one.

        :param model_stream: Input stream, containing a model previously exported, using export_model method.
        :type model_stream: bytes
        :param device_name: Name of device to which compiled model is imported.
                            Note: if device_name is not used to compile the original model,
                            an exception is thrown.
        :type device_name: str
        :param config: Optional dict of pairs:
                       (property name, property value) relevant only for this load operation.
        :type config: dict, optional
        :return: A compiled model.
        :rtype: openvino.CompiledModel

        :Example:

        .. code-block:: python

            user_stream = compiled.export_model()

            with open('./my_model', 'wb') as f:
                f.write(user_stream)

            # ...

            new_compiled = core.import_model(user_stream, "CPU")

        .. code-block:: python

            user_stream = io.BytesIO()
            compiled.export_model(user_stream)

            with open('./my_model', 'wb') as f:
                f.write(user_stream.getvalue()) # or read() if seek(0) was applied before

            # ...

            new_compiled = core.import_model(user_stream, "CPU")
        """
        return CompiledModel(
            super().import_model(
                model_stream,
                device_name,
                {} if config is None else config,
            ),
        )


def compile_model(
    model: Union[Model, str, Path],
    device_name: Optional[str] = "AUTO",
    config: Optional[dict] = None,
) -> CompiledModel:
    """Compact method to compile model with AUTO plugin.

    :param model: Model acquired from read_model function or a path to a model in IR / ONNX / PDPD /
                    TF and TFLite format.
    :type model: Union[openvino.Model, str, pathlib.Path]
    :param device_name: Optional. Name of the device to load the model to. If not specified,
                        the default OpenVINO device will be selected by AUTO plugin.
    :type device_name: str
    :param config: Optional dict of pairs:
                    (property name, property value) relevant only for this load operation.
    :type config: dict, optional
    :return: A compiled model.
    :rtype: openvino.CompiledModel

    """
    core = Core()
    if isinstance(model, Model):
        model = model._Model__model
    return core.compile_model(model, device_name, {} if config is None else config)
