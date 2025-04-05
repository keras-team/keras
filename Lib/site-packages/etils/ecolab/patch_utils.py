# Copyright 2024 The etils Authors.
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

"""Graph utils."""

import os
import sys
import warnings


def set_verbose() -> None:
  """Log stderr & `absl.logging` in Colab (filtered by default)."""
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from absl import logging
  from colabtools import googlelog
  # pytype: enable=import-error
  # pylint: enable=g-import-not-at-top

  logging.set_verbosity(logging.INFO)
  googlelog.set_global_capture(True)

  # See:
  # https://docs.python.org/3/library/warnings.html#overriding-the-default-filter
  if not sys.warnoptions:
    warnings.simplefilter('default')
    os.environ['PYTHONWARNINGS'] = 'default'  # Also affect subprocesses


def patch_graphviz() -> None:
  """Fix `graphviz` display on Colab.

  By default, graphviz object raises an error when displayed on Colab:

  ```
  ExecutableNotFound: failed to execute ['dot', '-Tsvg'], make sure the
  Graphviz executables are on your systems' PATH
  ```

  Calling this function fix the behavior.
  """
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from colabtools import proto
  from colabtools import publish
  from colabtools import stubby

  import graphviz
  # pytype: enable=import-error
  # pylint: enable=g-import-not-at-top

  request_proto_cls = proto.GetProtoClass('graphviz_server.RenderRequest')
  graph_proto_cls = proto.GetProtoClass('graphviz_server.Graph')

  def _ipython_display_(self):
    graph = graph_proto_cls()
    graph.dot = self.source
    response = stubby.Call(
        'blade:graphviz-server',
        'RenderServer.Render',
        request_proto_cls(graph=graph),
    )
    publish.html(response.rendered_graph.rendered_bytes)

  if getattr(graphviz, 'files', None):
    files = getattr(graphviz, 'files')
    files.File._ipython_display_ = (  # pylint: disable=protected-access
        _ipython_display_
    )
