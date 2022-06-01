# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for testing multi-worker distribution strategies with Keras."""

import threading
import unittest

import tensorflow.compat.v2 as tf

import keras
from keras.optimizers.optimizer_v2 import gradient_descent

# isort: off
from tensorflow.python.distribute.cluster_resolver import (
    SimpleClusterResolver,
)
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.server_lib import (
    ClusterSpec,
)

_portpicker_import_error = None
try:
    import portpicker
except (
    ImportError,
    ModuleNotFoundError,
) as _error:
    _portpicker_import_error = _error
    portpicker = None

ASSIGNED_PORTS = set()
lock = threading.Lock()


def mnist_synthetic_dataset(batch_size, steps_per_epoch):
    """Generate synthetic MNIST dataset for testing."""
    # train dataset
    x_train = tf.ones(
        [batch_size * steps_per_epoch, 28, 28, 1], dtype=tf.float32
    )
    y_train = tf.ones([batch_size * steps_per_epoch, 1], dtype=tf.int32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    # train_ds = train_ds.shuffle(100)
    train_ds = train_ds.batch(64, drop_remainder=True)

    # eval dataset
    x_test = tf.random.uniform([10000, 28, 28, 1], dtype=tf.float32)
    y_test = tf.random.uniform([10000, 1], minval=0, maxval=9, dtype=tf.int32)
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.batch(64, drop_remainder=True)

    return train_ds, eval_ds


def get_mnist_model(input_shape):
    """Define a deterministically-initialized CNN model for MNIST testing."""
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer=keras.initializers.TruncatedNormal(seed=99),
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x) + keras.layers.Flatten()(x)
    x = keras.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer=keras.initializers.TruncatedNormal(seed=99),
    )(x)
    model = keras.Model(inputs=inputs, outputs=x)

    # TODO(yuefengz): optimizer with slot variables doesn't work because of
    # optimizer's bug.
    # TODO(yuefengz): we should not allow non-v2 optimizer.
    model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=gradient_descent.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def make_parameter_server_cluster(num_workers, num_ps):
    cluster_def = create_in_process_cluster(
        num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc"
    )
    return SimpleClusterResolver(ClusterSpec(cluster_def), rpc_layer="grpc")


def pick_unused_port():
    """Returns an unused and unassigned local port."""
    if _portpicker_import_error:
        raise _portpicker_import_error

    global ASSIGNED_PORTS
    with lock:
        while True:
            try:
                port = portpicker.pick_unused_port()
            except portpicker.NoFreePortFoundError:
                raise unittest.SkipTest(
                    "Flakes in portpicker library do not represent "
                    "TensorFlow errors."
                )
            if port > 10000 and port not in ASSIGNED_PORTS:
                ASSIGNED_PORTS.add(port)
                logging.info("Using local port %r", port)
                return port


def _create_cluster(
    num_workers,
    num_ps,
    has_chief=False,
    has_eval=False,
    protocol="grpc",
    worker_config=None,
    ps_config=None,
    eval_config=None,
    worker_name="worker",
    ps_name="ps",
    chief_name="chief",
):
    """Creates and starts local servers and returns the cluster_spec dict."""
    if _portpicker_import_error:
        raise _portpicker_import_error
    worker_ports = [pick_unused_port() for _ in range(num_workers)]
    ps_ports = [pick_unused_port() for _ in range(num_ps)]

    cluster_dict = {}
    if num_workers > 0:
        cluster_dict[worker_name] = [
            "localhost:%s" % port for port in worker_ports
        ]
    if num_ps > 0:
        cluster_dict[ps_name] = ["localhost:%s" % port for port in ps_ports]
    if has_eval:
        cluster_dict["evaluator"] = ["localhost:%s" % pick_unused_port()]
    if has_chief:
        cluster_dict[chief_name] = ["localhost:%s" % pick_unused_port()]

    cs = tf.train.ClusterSpec(cluster_dict)

    for i in range(num_workers):
        tf.distribute.Server(
            cs,
            job_name=worker_name,
            protocol=protocol,
            task_index=i,
            config=worker_config,
            start=True,
        )

    for i in range(num_ps):
        tf.distribute.Server(
            cs,
            job_name=ps_name,
            protocol=protocol,
            task_index=i,
            config=ps_config,
            start=True,
        )

    if has_chief:
        tf.distribute.Server(
            cs,
            job_name=chief_name,
            protocol=protocol,
            task_index=0,
            config=worker_config,
            start=True,
        )

    if has_eval:
        tf.distribute.Server(
            cs,
            job_name="evaluator",
            protocol=protocol,
            task_index=0,
            config=eval_config,
            start=True,
        )

    return cluster_dict


def create_in_process_cluster(
    num_workers, num_ps, has_chief=False, has_eval=False, rpc_layer="grpc"
):
    """Create an in-process cluster that consists of only standard server."""
    # Leave some memory for cuda runtime.
    gpu_mem_frac = 0.7 / (num_workers + int(has_chief) + int(has_eval))
    worker_config = tf.compat.v1.ConfigProto()
    worker_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_frac

    # The cluster may hang if workers don't have enough inter_op threads. See
    # b/172296720 for more details.
    if worker_config.inter_op_parallelism_threads < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1

    # Enable collective ops which has no impact on non-collective ops.
    if has_chief:
        worker_config.experimental.collective_group_leader = (
            "/job:chief/replica:0/task:0"
        )
    else:
        worker_config.experimental.collective_group_leader = (
            "/job:worker/replica:0/task:0"
        )

    ps_config = tf.compat.v1.ConfigProto()
    ps_config.device_count["GPU"] = 0

    eval_config = tf.compat.v1.ConfigProto()
    eval_config.experimental.collective_group_leader = ""

    # Create in-process servers. Once an in-process tensorflow server is
    # created, there is no way to terminate it. So we create one cluster per
    # test process.  We could've started the server in another process, we could
    # then kill that process to terminate the server. The reasons why we don"t
    # want multiple processes are
    # 1) it is more difficult to manage these processes;
    # 2) there is something global in CUDA such that if we initialize CUDA in
    # the parent process, the child process cannot initialize it again and thus
    # cannot use GPUs (https://stackoverflow.com/questions/22950047).
    cluster = None
    try:
        cluster = _create_cluster(
            num_workers,
            num_ps=num_ps,
            has_chief=has_chief,
            has_eval=has_eval,
            worker_config=worker_config,
            ps_config=ps_config,
            eval_config=eval_config,
            protocol=rpc_layer,
        )
    except tf.errors.UnknownError as e:
        if "Could not start gRPC server" in e.message:
            raise unittest.SkipTest("Cannot start std servers.")
        else:
            raise
    return cluster
