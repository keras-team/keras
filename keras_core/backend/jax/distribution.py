"""!!!DO NOT USE!!!

Distribution related class for JAX backend.

This is just a prototype and we might want to unify it
with other backends in the future.
"""

import contextlib

import jax
import numpy as np
from absl import logging

from keras_core.backend.common import global_state

DEFAULT_BATCH_DIM_NAME = "batch"
GLOBAL_ATTRIBUTE_NAME = "distribution"


def get_global_distribution():
    """Retrieve the current distribution from global context."""
    return global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)


def set_global_distribution(distribution):
    """Set the distribution as the global distribution setting."""
    # TODO(qlzh727): Do type checking for input once we have a base class.
    distribution.as_global_distribution()


class DataParallelDistribution:
    def __init__(self, mesh=None, devices=None):
        """Create the data parallel distribution.

        You can choose to create this instance by either specifing
        the `Mesh` or `devices` parameters (but not both).

        The mesh is expected to be a `jax.sharding.Mesh` instance, and is
        expected to be 1D only. In case that the mesh has multiple axes, then
        the first axis will be treated as the data parallel dimension
        (and a warning will be raised).

        When a list of `devices` are provided, they will be used to construct a
        1D mesh.

        When both `mesh` and `devices` are absent, then `jax.devices()`
        will be used to detect any available devices
        and create a 1D mesh from them.
        """
        super().__init__()
        if mesh:
            self._initialize_with_mesh(mesh)
        elif devices:
            self._initialize_mesh_from_devices(devices)
        else:
            self._initialize_mesh_from_jax_devices()

        self._configure_sharding_spec()
        self._batch_dim_name = self.mesh.axis_names[0]

    @contextlib.contextmanager
    def scope(self):
        original_scope = global_state.get_global_attribute(
            GLOBAL_ATTRIBUTE_NAME
        )
        global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, self)
        try:
            yield
        finally:
            global_state.set_global_attribute(
                GLOBAL_ATTRIBUTE_NAME, original_scope
            )

    def as_global_distribution(self):
        global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, self)

    def distribute_data(self, data):
        return jax.device_put(data, self._data_sharding)

    def distribute_variable(self, variable):
        return jax.device_put(variable, self._variable_sharding)

    def _initialize_with_mesh(self, mesh):
        if not isinstance(mesh, jax.sharding.Mesh):
            raise ValueError(
                "Expect `mesh` to be an instance of `jax.sharding.Mesh`. "
                f"Received: mesh={mesh} (of type {type(mesh)})"
            )
        self._user_provide_devices = None
        self.mesh = mesh
        if self.mesh.devices.ndim != 1:
            logging.warning(
                "Expect the input mesh to be 1D, but received "
                f"mesh.devices.ndim={mesh.devices.ndim}. "
                "The first axis will be used for data-parallel sharding."
            )

    def _initialize_mesh_from_devices(self, devices):
        self._user_provide_devices = devices
        self.mesh = jax.sharding.Mesh(np.array(devices), DEFAULT_BATCH_DIM_NAME)

    def _initialize_mesh_from_jax_devices(self):
        self._user_provide_devices = None
        self.mesh = jax.sharding.Mesh(
            np.array(jax.devices()), DEFAULT_BATCH_DIM_NAME
        )

    def _configure_sharding_spec(self):
        variable_shard_spec = [
            None
        ] * self.mesh.devices.ndim  # Fully replicated
        data_shard_spec = variable_shard_spec.copy()
        data_shard_spec[0] = self.mesh.axis_names[0]  # Shard on the first dim

        self._data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*data_shard_spec)
        )
        self._variable_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*variable_shard_spec)
        )
