# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import math


def is_serializable_value(value):
    """Returns whether a protobuf Value will be serializable by MessageToJson.

    The json_format documentation states that "attempting to serialize NaN or
    Infinity results in error."

    https://protobuf.dev/reference/protobuf/google.protobuf/#value

    Args:
      value: A value of type protobuf.Value.

    Returns:
      True if the Value should be serializable without error by MessageToJson.
      False, otherwise.
    """
    if not value.HasField("number_value"):
        return True

    number_value = value.number_value
    return not math.isnan(number_value) and not math.isinf(number_value)
