# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Saving utilities to support Python's Pickle protocol."""
import io
import os
import tarfile
import uuid

import numpy
import tensorflow.compat.v2 as tf

from keras.saving import save as save_module


def deserialize_model_from_bytecode(serialized_model):
    """Reconstruct a Model from the output of `serialize_model_as_bytecode`.

    Args:
        serialized_model: (np.array) return value from
          `serialize_model_as_bytecode`.

    Returns:
        keras.Model: Keras Model instance.
    """
    temp_dir = f"ram://{uuid.uuid4()}"
    b = io.BytesIO(serialized_model)
    with tarfile.open(fileobj=b, mode="r") as archive:
        for name in archive.getnames():
            dest_path = tf.io.gfile.join(temp_dir, name)
            member = archive.getmember(name)
            tf.io.gfile.makedirs(os.path.dirname(dest_path))
            if member.isfile():
                with tf.io.gfile.GFile(dest_path, "wb") as f:
                    f.write(archive.extractfile(name).read())
    model = save_module.load_model(temp_dir)
    tf.io.gfile.rmtree(temp_dir)
    return model


def serialize_model_as_bytecode(model):
    """Convert a Keras Model into a bytecode representation for pickling.

    Args:
        model: (tf.keras.Model) Keras Model instance.

    Returns:
        tuple: tuple of arguments that can be sent to
            `deserialize_from_bytecode`.
    """
    temp_dir = f"ram://{uuid.uuid4()}"
    model.save(temp_dir)
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as archive:
        for root, dirs, filenames in tf.io.gfile.walk(temp_dir):
            for dirname in dirs:
                dest_path = tf.io.gfile.join(root, dirname)
                t = tarfile.TarInfo(dest_path)
                t.type = tarfile.DIRTYPE
                archive.addfile(t)
            for filename in filenames:
                dest_path = tf.io.gfile.join(root, filename)
                with tf.io.gfile.GFile(dest_path, "rb") as f:
                    info = tarfile.TarInfo(
                        name=os.path.relpath(dest_path, temp_dir)
                    )
                    info.size = f.size()
                    archive.addfile(tarinfo=info, fileobj=f)
    tf.io.gfile.rmtree(temp_dir)
    b.seek(0)
    return (numpy.asarray(memoryview(b.read())),)
