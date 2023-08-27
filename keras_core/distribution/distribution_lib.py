"""Unified high level distribution APIs across backends.

!!!DO NOT USE!!! Currently under development and APIs are not final.

Currently only the JAX backend has been implemented, and the Tensorflow backend
will be implemented in future (via tf.dtensor API).
"""

import collections
import contextlib
import re
import warnings

import numpy as np

from keras_core.backend import distribution_lib
from keras_core.backend.common import global_state

DEFAULT_BATCH_DIM_NAME = "batch"
GLOBAL_ATTRIBUTE_NAME = "distribution"


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Default to `gpu` or
            `tpu` if available when device_type is not provided. Otherwise
            will return the `cpu` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    return distribution_lib.list_devices(device_type)


class DeviceMesh:
    """The cluster of computation devices for distributed computation.

    This is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in [jax.sharding.Mesh](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh)
    and [tf.dtensor.Mesh](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).
    """

    def __init__(
        self,
        shape,
        axis_names,
        devices=None,
    ):
        """Initialize the DeviceMesh for the given topology.

        Args:
            shape: tuple of list of integers. The shape of the overall
                DeviceMesh, e.g. `(8,)` for a data parallel only distribution,
                or `(4, 2)` for a model+data parallel distribution.
            axis_names: List of string. The logical name of the each axis for
                the DeviceMesh. The length of the `axis_names` should match to
                the rank of the `shape`. The `axis_names` will be used to
                match/create the `TensorLayout` when distribute the data and
                weights.
            devices: Optional list of devices. Default to all the available
                devices locally from `list_devices()`.
        """
        if not shape or not axis_names:
            raise ValueError(
                "Shape and axis_names cannot be empty. Got "
                f"shape: {shape}, axis_names: {axis_names}"
            )

        if len(shape) != len(axis_names):
            raise ValueError(
                "Shape and axis_names should have same size,"
                f"got shape: {shape} and axis_names: {axis_names}"
            )
        if devices is None:
            devices = list_devices()
        devices = np.array(devices)
        if np.prod(shape) != np.prod(devices.shape):
            raise ValueError(
                "Shape does not match the number of devices. "
                f"Got shape: {shape}, and shape of the "
                f"devices: {devices.shape}"
            )

        self._shape = shape
        self._axis_names = axis_names
        self._devices = np.reshape(devices, shape)

    @property
    def shape(self):
        return self._shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def devices(self):
        return self._devices


class TensorLayout:
    """The layout of a Tensor.

    This is aligned with `jax.sharding.NamedSharding` and `tf.dtensor.Layout`,
    which allocate the tensor to its logic axis based on the `DeviceMesh`. With
    `DeviceMesh` and `TensorLayout`, the actual mapping between a Tensor to the
    physical devices can be determined.

    See more details in [jax.sharding.NamedSharding](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding)
    and [tf.dtensor.Layout](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout).
    """

    def __init__(self, axes, device_mesh=None):
        """Initialize the TensorLayout with axis names.

        Args:
            axes: list of strings that should map to the `axis_names` in
                `DeviceMesh`. For any dimentions that doesn't need any sharding,
                A `None` can be used a placeholder.
            device_mesh: Optional `DeviceMesh` that will be used to create
                the layout. The actual mapping of tensor to physical device
                is not known until the mesh is specified.
        """
        self._axes = axes
        self._device_mesh = device_mesh
        self._validate_axes()

    @property
    def axes(self):
        return self._axes

    @property
    def device_mesh(self):
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, device_mesh):
        if self._device_mesh is not None:
            raise ValueError(
                "Cannot override device mesh value. Existing "
                f"value is {self._device_mesh}"
            )
        self._device_mesh = device_mesh
        self._validate_axes()

    def _validate_axes(self):
        if self._device_mesh:
            valid_axis_names = set(self._device_mesh.axis_names)
            axis_names = set(self._axes) - set([None])
            if axis_names - valid_axis_names:
                raise ValueError(
                    "Invalid axis names for Layout. Valid axis "
                    f"names: {valid_axis_names}, Got {axis_names}"
                )


class Distribution:
    """Base class for the distribution.

    The `Distribution` has following key functionalities.

    1. Distribute the model variables to the `DeviceMesh`.
    2. Distribute the input data to the `DeviceMesh`.

    It can create a context scope so that the framework to properly detect the
    `Distribution` and distribute the variable/data accordingly.
    """

    def __init__(self, device_mesh):
        self._device_mesh = device_mesh

    def get_data_layout(self, data_shape):
        """Retrieve the `TensorLayout` for the input data.

        Args:
            data_shape: shape for the input data in list or tuple format.

        Returns:
            The `TensorLayout` for the data, which can be used by
            `backend.distribute_value()` to redistribute a input data.
        """
        raise NotImplementedError()

    def get_variable_layout(self, variable):
        """Retrieve the `TensorLayout` for the variable.

        Args:
            variable: A `KerasVariable` to retrieve the `TensorLayout`.

        return:
            The `TensorLayout` for the variable, which can be used by
            `backend.distribute_value()` to redistribute a variable.
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def scope(self):
        """Context manager to make the `Distribution` current."""
        original_scope = distribution()
        set_distribution(self)
        try:
            yield
        finally:
            set_distribution(original_scope)

    @property
    def device_mesh(self):
        return self._device_mesh


class DataParallel(Distribution):
    def __init__(self, device_mesh=None, devices=None):
        """Create the data parallel distribution.

        You can choose to create this instance by either specifing
        the `device_mesh` or `devices` parameters (but not both).

        The device_mesh is expected to be a `DeviceMesh` instance, and is
        expected to be 1D only. In case that the mesh has multiple axes, then
        the first axis will be treated as the data parallel dimension
        (and a warning will be raised).

        When a list of `devices` are provided, they will be used to construct a
        1D mesh.

        When both `mesh` and `devices` are absent, then `list_devices()`
        will be used to detect any available devices and create a 1D mesh from
        them.
        """
        if device_mesh:
            self._initialize_with_device_mesh(device_mesh)
        elif devices:
            self._initialize_mesh_from_devices(devices)
        else:
            self._initialize_mesh_from_list_devices()

        self._batch_dim_name = self.device_mesh.axis_names[0]

    def _initialize_with_device_mesh(self, device_mesh):
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                "Expect `mesh` to be an instance of `DeviceMesh`. "
                f"Received: mesh={device_mesh} (of type {type(device_mesh)})"
            )
        super().__init__(device_mesh)
        if self.device_mesh.devices.ndim != 1:
            warnings.warn(
                "Expect the input mesh to be 1D, but received "
                "mesh.devices.ndim=%d. "
                "The first axis will be used for data-parallel sharding.",
                device_mesh.devices.ndim,
            )

    def _initialize_mesh_from_devices(self, devices):
        devices = np.array(devices)
        device_mesh = DeviceMesh(
            shape=devices.shape,
            axis_names=[DEFAULT_BATCH_DIM_NAME],
            devices=devices,
        )
        super().__init__(device_mesh)

    def _initialize_mesh_from_list_devices(self):
        devices = np.array(list_devices())
        device_mesh = DeviceMesh(
            shape=devices.shape,
            axis_names=[DEFAULT_BATCH_DIM_NAME],
            devices=devices,
        )
        super().__init__(device_mesh)

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name  # Shard on the first dim
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)


class ModelParallel(Distribution):
    """Distribution that shard model weights.

    Compare to DataParallel which replicates the weights across all the devices,
    ModelParallel allows user to shard weights in addition to the input data.

    To construct a ModelParallel distribution, user need to provide device mesh
    and layout mapping.

    1. `DeviceMesh`contains physcial device information, and the axis names in
        the mesh will be used to map the weight and data layout.
    2. `LayoutMap` contains the mapping for the variable path to its
        corresponding `TensorLayout`.

    Example:
    ```python
    devices = list_devices()    # Assume there are 8 devices.

    # Create a mesh with 2 devices on data parallel and 4 devices on weight
    # parallel.
    device_mesh = DeviceMesh(shape=(2, 4), axis_names=('batch', 'model'),
                             devices=devices)
    # Create a layout map that shard the dense layer and conv2d layer weights
    # on the last dimension. Based on the device_mesh, this means the weights
    # will be split across 4 devices. Any other weights that doesn't match for
    # any key in layout map will get be fully replicated.
    layout_map = LayoutMap(device_mesh)
    layout_map['.*dense.*kernel'] = TensorLayout([None, 'model'])
    layout_map['.*dense.*bias'] = TensorLayout(['model'])
    layout_map['.*conv2d.*kernel'] = TensorLayout([None, None, None, 'model'])
    layout_map['.*conv2d.*bias'] = TensorLayout(['model'])

    distribution = ModelParallel(device_mesh=device_mesh,
                                 layout_map=layout_map,
                                 batch_dim_name='batch')
    # Set the global distribution, or via `with distribution.scope():`
    set_distribution(distribution)

    model = model_creation()
    model.compile()
    model.fit(data)
    ```

    User can quickly update the device mesh shape to change the sharding factor
    of the weights. E.g.
    ```
    # With only the shape change for the device mesh, the weights will be
    # sharded across 8 devices instead of 4, which further reduce the memory
    # footprint of weights on each of the device.
    device_mesh = DeviceMesh(shape=(1, 8), axis_names=('batch', 'model'),
                             devices=devices)
    ```

    To figure out a proper layout mapping rule for all the model weights, you
    can first list out all the model weights path, which will be used as the key
    to map the weights to `TensorLayout`.

    e.g.
    ```
    model = create_model()
    for w in model.weights:
        print(w.path)
    ```
    """

    def __init__(self, device_mesh, layout_map, batch_dim_name=None):
        """Initialize the model parallel distribution.

        Args:
            device_mesh: `DeviceMesh` instance for physical device and its
                logical mapping.
            layout_map: `LayoutMap` instance which map the variable path to the
                corresponding `TensorLayout`. The axis names of the
                `TensorLayout`s should match to the axis names in the
                device_mesh, or exception will be raised.
            batch_dim_name: optional string, the axis name in the device_mesh
                that will be used to distribute data. The first axis from the
                device_mesh will be used if user didn't specify any.
        """
        super().__init__(device_mesh)
        self._layout_map = layout_map
        self._batch_dim_name = batch_dim_name or self.device_mesh.axis_names[0]

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name  # Shard on the first dim
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        variable_layout = self._layout_map[variable.path]
        if variable_layout is not None:
            return variable_layout
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)


class LayoutMap(collections.abc.MutableMapping):
    """A dict-like object that maps string to `TensorLayout` instances.

    `LayoutMap` uses a string as key and a `TensorLayout` as value. There is a
    behavior difference between a normal Python dict and this class. The string
    key will be treated as a regex when retrieving the value. See the docstring
    of `get` for more details.

    See below for a usage example. You can define the naming schema
    of the `TensorLayout`, and then retrieve the corresponding
    `TensorLayout` instance.

    In the normal case, the key to query is usually the `variable.path`, which
    is the idenifier of the variable.

    ```python
    layout_map = LayoutMap(device_mesh=None)
    layout_map['.*dense.*kernel'] = layout_2d
    layout_map['.*dense.*bias'] = layout_1d
    layout_map['.*conv2d.*kernel'] = layout_4d
    layout_map['.*conv2d.*bias'] = layout_1d

    layout_1 = layout_map['dense_1.kernel']             # layout_1 == layout_2d
    layout_2 = layout_map['dense_1.bias']               # layout_2 == layout_1d
    layout_3 = layout_map['dense_2.kernel']             # layout_3 == layout_2d
    layout_4 = layout_map['dense_2.bias']               # layout_4 == layout_1d
    layout_5 = layout_map['my_model/conv2d_123/kernel'] # layout_5 == layout_4d
    layout_6 = layout_map['my_model/conv2d_123/bias']   # layout_6 == layout_1d
    layout_7 = layout_map['my_model/conv3d_1/kernel']   # layout_7 == None
    layout_8 = layout_map['my_model/conv3d_1/bias']     # layout_8 == None
    ```

    Args:
        device_mesh: An optional `DeviceMesh` that can be used to populate the
            `TensorLayout.device_mesh` if the `TensorLayout.device_mesh` is not
            set.
    """

    def __init__(self, device_mesh=None):
        self._layout_map = collections.OrderedDict()
        self._device_mesh = device_mesh

    def __getitem__(self, key):
        """Retrieve the corresponding layout by the string key.

        When there isn't an exact match, all the existing keys in the layout map
        will be treated as a regex and map against the input key again. The
        first match will be returned, based on the key insertion order. Return
        None if there isn't any match found.

        Args:
            key: the string key as the query for the layout.

        Returns:
            Corresponding layout based on the query.
        """
        if key in self._layout_map:
            return self._layout_map[key]

        for k in self._layout_map:
            if re.match(k, key):
                return self._layout_map[k]
        return None

    def __setitem__(self, key, layout):
        if key in self._layout_map:
            raise ValueError(
                f"{key} already exist in the LayoutMap with "
                f"value {self._layout_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        if not isinstance(layout, TensorLayout):
            raise ValueError(
                f"{layout} should be a TensorLayout type, got {type(layout)}"
            )
        self._maybe_populate_device_mesh(layout)
        self._layout_map[key] = layout

    def __delitem__(self, key):
        # let the dict to handle the key missing error
        return self._layout_map.pop(key)

    def __len__(self):
        return len(self._layout_map)

    def __iter__(self):
        return iter(self._layout_map)

    @property
    def device_mesh(self):
        return self._device_mesh

    def _maybe_populate_device_mesh(self, layout):
        if layout.device_mesh is None and self.device_mesh is not None:
            layout.device_mesh = self.device_mesh


def distribution():
    """Retrieve the current distribution from global context."""
    return global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)


def set_distribution(value):
    """Set the distribution as the global distribution setting.

    Args:
        value: a `Distribution` instance.
    """
    global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, value)
