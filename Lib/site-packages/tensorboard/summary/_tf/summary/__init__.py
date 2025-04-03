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

# NOTE: This module exists to provide the `tf.summary` module in the TensorFlow
# API using symbols defined in TensorBoard. This works via a mechanism within
# TensorFlow's API construction logic called "component_api_helper" that imports
# an arbitrary module and inserts it into the TF APIs as a "component API". That
# logic is specifically hardcoded to look for this exact tensorboard module.
#
# This note is in a comment, not the module docstring, because the module
# docstring below is what users will see as the tf.summary docstring and in the
# generated API documentation, and this is just an implementation detail.

"""Operations for writing summary data, for use in analysis and visualization.

The `tf.summary` module provides APIs for writing summary data. This data can be
visualized in TensorBoard, the visualization toolkit that comes with TensorFlow.
See the [TensorBoard website](https://www.tensorflow.org/tensorboard) for more
detailed tutorials about how to use these APIs, or some quick examples below.

Example usage with eager execution, the default in TF 2.0:

```python
writer = tf.summary.create_file_writer("/tmp/mylogs")
with writer.as_default():
  for step in range(100):
    # other model code would go here
    tf.summary.scalar("my_metric", 0.5, step=step)
    writer.flush()
```

Example usage with `tf.function` graph execution:

```python
writer = tf.summary.create_file_writer("/tmp/mylogs")

@tf.function
def my_func(step):
  # other model code would go here
  with writer.as_default():
    tf.summary.scalar("my_metric", 0.5, step=step)

for step in range(100):
  my_func(step)
  writer.flush()
```

Example usage with legacy TF 1.x graph execution:

```python
with tf.compat.v1.Graph().as_default():
  step = tf.Variable(0, dtype=tf.int64)
  step_update = step.assign_add(1)
  writer = tf.summary.create_file_writer("/tmp/mylogs")
  with writer.as_default():
    tf.summary.scalar("my_metric", 0.5, step=step)
  all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
  writer_flush = writer.flush()

  sess = tf.compat.v1.Session()
  sess.run([writer.init(), step.initializer])
  for i in range(100):
    sess.run(all_summary_ops)
    sess.run(step_update)
    sess.run(writer_flush)
```
"""


# Keep this import outside the function below for internal sync reasons.
import tensorflow as tf


def reexport_tf_summary():
    """Re-export all symbols from the original tf.summary.

    This function finds the original tf.summary V2 API and re-exports all the
    symbols from it within this module as well, so that when this module is
    patched into the TF API namespace as the new tf.summary, the effect is an
    overlay that just adds TensorBoard-provided symbols to the module.

    Finding the original tf.summary V2 API module reliably is a challenge, since
    this code runs *during* the overall TF API import process and depending on
    the order of imports (which is subject to change), different parts of the API
    may or may not be defined at the point in time we attempt to access them. This
    code also may be inserted into two places in the API (tf and tf.compat.v2)
    and may be re-executed multiple times even for the same place in the API (due
    to the TF module import system not populating sys.modules properly), so it
    needs to be robust to many different scenarios.

    The one constraint we can count on is that everywhere this module is loaded
    (via the component_api_helper mechanism in TF), it's going to be the 'summary'
    submodule of a larger API package that already has a 'summary' attribute
    that contains the TF-only summary API symbols we need to re-export. This
    may either be the original TF-only summary module (the first time we load
    this module) or a pre-existing copy of this module (if we're re-loading this
    module again). We don't actually need to differentiate those two cases,
    because it's okay if we re-import our own TensorBoard-provided symbols; they
    will just be overwritten later on in this file.

    So given that guarantee, the approach we take is to first attempt to locate
    a TF V2 API package that already has a 'summary' attribute (most likely this
    is the parent package into which we're being imported, but not necessarily),
    and then do the dynamic version of "from tf_api_package.summary import *".

    Lastly, this logic is encapsulated in a function to avoid symbol leakage.
    """
    import sys

    # API packages to check for the original V2 summary API, in preference order
    # to avoid going "under the hood" to the _api packages unless necessary.
    # Skip the top-level `tensorflow` package since it's hard to confirm that it
    # is the actual v2 API (just checking tf.__version__ is not always enough).
    packages = [
        "tensorflow.compat.v2",
        "tensorflow_core._api.v2",
        "tensorflow_core._api.v2.compat.v2",
        "tensorflow_core._api.v1.compat.v2",
        # Old names for `tensorflow_core._api.*`.
        "tensorflow._api.v2",
        "tensorflow._api.v2.compat.v2",
        "tensorflow._api.v1.compat.v2",
    ]

    def dynamic_wildcard_import(module):
        """Implements the logic of "from module import *" for the given
        module."""
        symbols = getattr(module, "__all__", None)
        if symbols is None:
            symbols = [
                k for k in module.__dict__.keys() if not k.startswith("_")
            ]
        globals().update(
            {symbol: getattr(module, symbol) for symbol in symbols}
        )

    notfound = object()  # sentinel value
    for package_name in packages:
        package = sys.modules.get(package_name, notfound)
        if package is notfound:
            # Either it isn't in this installation at all (e.g. the _api.vX packages
            # are only in API version X), it isn't imported yet, or it was imported
            # but not inserted into sys.modules under its user-facing name (for the
            # non-'_api' packages), at which point we continue down the list to look
            # "under the hood" for it via its '_api' package name.
            continue
        module = getattr(package, "summary", None)
        if module is None:
            # This happens if the package hasn't been fully imported yet. For example,
            # the 'tensorflow' package won't yet have 'summary' attribute if we are
            # loading this code via the 'tensorflow.compat...' path and 'compat' is
            # imported before 'summary' in the 'tensorflow' __init__.py file.
            continue
        # Success, we hope. Import all the public symbols into this module.
        dynamic_wildcard_import(module)
        return


reexport_tf_summary()

from tensorboard.summary.v2 import audio  # noqa: F401
from tensorboard.summary.v2 import histogram  # noqa: F401
from tensorboard.summary.v2 import image  # noqa: F401
from tensorboard.summary.v2 import scalar  # noqa: F401
from tensorboard.summary.v2 import text  # noqa: F401

del tf, reexport_tf_summary
