# SPDX-License-Identifier: Apache-2.0

"""tf2onnx package."""

__all__ = ["utils", "graph_matcher", "graph", "graph_builder",
           "tfonnx", "shape_inference", "schemas", "tf_utils", "tf_loader", "convert"]

import onnx
from .version import git_version, version as __version__
from . import verbose_logging as logging
from tf2onnx import tfonnx, utils, graph, graph_builder, graph_matcher, shape_inference, schemas, convert  # pylint: disable=wrong-import-order
