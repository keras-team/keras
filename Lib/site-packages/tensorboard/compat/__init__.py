# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Compatibility interfaces for TensorBoard.

This module provides logic for importing variations on the TensorFlow
APIs, as lazily loaded imports to help avoid circular dependency issues
and defer the search and loading of the module until necessary.
"""


import tensorboard.lazy as _lazy


@_lazy.lazy_load("tensorboard.compat.tf")
def tf():
    """Provide the root module of a TF-like API for use within TensorBoard.

    By default this is equivalent to `import tensorflow as tf`, but it can be used
    in combination with //tensorboard/compat:tensorflow (to fall back to a stub TF
    API implementation if the real one is not available) or with
    //tensorboard/compat:no_tensorflow (to force unconditional use of the stub).

    Returns:
      The root module of a TF-like API, if available.

    Raises:
      ImportError: if a TF-like API is not available.
    """
    try:
        from tensorboard.compat import notf  # noqa: F401
    except ImportError:
        try:
            import tensorflow

            return tensorflow
        except ImportError:
            pass
    from tensorboard.compat import tensorflow_stub

    return tensorflow_stub


@_lazy.lazy_load("tensorboard.compat.tf2")
def tf2():
    """Provide the root module of a TF-2.0 API for use within TensorBoard.

    Returns:
      The root module of a TF-2.0 API, if available.

    Raises:
      ImportError: if a TF-2.0 API is not available.
    """
    # Resolve the lazy `tf` compat API from earlier in this file and try to find
    # tf.compat.v2. Don't check tf.__version__ since this is not always reliable
    # if TF was built with tf_api_version!=2.
    if hasattr(tf, "compat") and hasattr(tf.compat, "v2"):
        return tf.compat.v2
    raise ImportError("cannot import tensorflow 2.0 API")
