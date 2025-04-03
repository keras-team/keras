# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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


from tensorboard.compat.proto.config_pb2 import *  # noqa
from tensorboard.compat.proto.event_pb2 import *  # noqa
from tensorboard.compat.proto.graph_pb2 import *  # noqa
from tensorboard.compat.proto.meta_graph_pb2 import *  # noqa
from tensorboard.compat.proto.summary_pb2 import *  # noqa
from .dtypes import as_dtype  # noqa
from .dtypes import DType  # noqa
from .dtypes import string  # noqa
from . import app  # noqa
from . import compat  # noqa
from . import dtypes  # noqa
from . import error_codes  # noqa
from . import errors  # noqa
from . import flags  # noqa
from . import io  # noqa
from . import pywrap_tensorflow  # noqa
from . import tensor_shape  # noqa

compat.v1.errors = errors

# Set a fake __version__ to help distinguish this as our own stub API.
__version__ = "stub"
