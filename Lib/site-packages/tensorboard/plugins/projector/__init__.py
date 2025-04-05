# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Public API for the Embedding Projector.

@@ProjectorPluginAsset
@@ProjectorConfig
@@EmbeddingInfo
@@EmbeddingMetadata
@@SpriteMetadata
"""


import os

from google.protobuf import text_format as _text_format
from tensorboard.compat import tf
from tensorboard.plugins.projector import metadata as _metadata
from tensorboard.plugins.projector.projector_config_pb2 import (  # noqa: F401
    EmbeddingInfo,
)
from tensorboard.plugins.projector.projector_config_pb2 import (  # noqa: F401
    SpriteMetadata,
)
from tensorboard.plugins.projector.projector_config_pb2 import (  # noqa: F401
    ProjectorConfig,
)


def visualize_embeddings(logdir, config):
    """Stores a config file used by the embedding projector.

    Args:
      logdir: Directory into which to store the config file, as a `str`.
        For compatibility, can also be a `tf.compat.v1.summary.FileWriter`
        object open at the desired logdir.
      config: `tf.contrib.tensorboard.plugins.projector.ProjectorConfig`
        proto that holds the configuration for the projector such as paths to
        checkpoint files and metadata files for the embeddings. If
        `config.model_checkpoint_path` is none, it defaults to the
        `logdir` used by the summary_writer.

    Raises:
      ValueError: If the summary writer does not have a `logdir`.
    """
    # Convert from `tf.compat.v1.summary.FileWriter` if necessary.
    logdir = getattr(logdir, "get_logdir", lambda: logdir)()

    # Sanity checks.
    if logdir is None:
        raise ValueError("Expected logdir to be a path, but got None")

    # Saving the config file in the logdir.
    config_pbtxt = _text_format.MessageToString(config)
    path = os.path.join(logdir, _metadata.PROJECTOR_FILENAME)
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(config_pbtxt)
