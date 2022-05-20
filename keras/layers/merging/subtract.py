# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Layer that subtracts two inputs."""


from keras.layers.merging.base_merge import _Merge
from keras.utils import tf_utils

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Subtract")
class Subtract(_Merge):
    """Layer that subtracts two inputs.

    It takes as input a list of tensors of size 2,
    both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
    also of the same shape.

    Examples:

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # Equivalent to subtracted = keras.layers.subtract([x1, x2])
        subtracted = keras.layers.Subtract()([x1, x2])

        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(
                "A `Subtract` layer should be called on exactly 2 inputs. "
                f"Received: input_shape={input_shape}"
            )

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError(
                "A `Subtract` layer should be called on exactly 2 inputs. "
                f"Received: inputs={inputs}"
            )
        return inputs[0] - inputs[1]


@keras_export("keras.layers.subtract")
def subtract(inputs, **kwargs):
    """Functional interface to the `Subtract` layer.

    Args:
        inputs: A list of input tensors (exactly 2).
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the difference of the inputs.

    Examples:

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        subtracted = keras.layers.subtract([x1, x2])

        out = keras.layers.Dense(4)(subtracted)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """
    return Subtract(**kwargs)(inputs)
