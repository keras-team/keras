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
import os
import tempfile

import tensorflow.compat.v2 as tf

from keras.saving.experimental import saving_lib


def deserialize_model_from_bytecode(serialized_model):
    """Reconstruct a Model from the output of `serialize_model_as_bytecode`.

    Args:
        serialized_model: (bytes) return value from
          `serialize_model_as_bytecode`.

    Returns:
        Keras Model instance.
    """
    # Note: we don't use a RAM path for this because zipfile cannot write
    # to such paths.
    temp_dir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(temp_dir, "model.keras")
        with open(filepath, "wb") as f:
            f.write(serialized_model)
        # When loading, direct import will work for most custom objects
        # though it will require get_config() to be implemented.
        # Some custom objects (e.g. an activation in a Dense layer,
        # serialized as a string by Dense.get_config()) will require
        # a custom_object_scope.
        model = saving_lib.load_model(filepath)
    except Exception as e:
        raise e
    else:
        return model
    finally:
        tf.io.gfile.rmtree(temp_dir)


def serialize_model_as_bytecode(model):
    """Convert a Keras Model into a bytecode representation for pickling.

    Args:
        model: Keras Model instance.

    Returns:
        Tuple that can be read by `deserialize_from_bytecode`.
    """
    # Note: we don't use a RAM path for this because zipfile cannot write
    # to such paths.
    temp_dir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(temp_dir, "model.keras")
        saving_lib.save_model(model, filepath)
        with open(filepath, "rb") as f:
            data = f.read()
    except Exception as e:
        raise e
    else:
        return data
    finally:
        tf.io.gfile.rmtree(temp_dir)
