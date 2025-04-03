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

"""Python utils public API."""

# pylint: disable=g-importing-member

import sys

from etils.epy import _internal
from etils.epy import typing
from etils.epy.adhoc_utils.binary_import import binary_adhoc
from etils.epy.contextlib import ContextManager
from etils.epy.contextlib import ExitStack
from etils.epy.env_utils import is_notebook
from etils.epy.env_utils import is_test
from etils.epy.itertools import groupby
from etils.epy.itertools import splitby
from etils.epy.itertools import zip_dict
from etils.epy.lazy_api_imports_utils import lazy_api_imports
from etils.epy.lazy_imports_utils import lazy_imports
from etils.epy.py_utils import frozen
from etils.epy.py_utils import is_namedtuple
from etils.epy.py_utils import issubclass_ as issubclass  # pylint: disable=redefined-builtin
from etils.epy.py_utils import StrEnum
from etils.epy.py_utils import wraps_cls
from etils.epy.re_utils import reverse_fstring
from etils.epy.reraise_utils import maybe_reraise
from etils.epy.reraise_utils import reraise
from etils.epy.text_utils import dedent
from etils.epy.text_utils import diff_str
from etils.epy.text_utils import Lines
from etils.epy.text_utils import pprint
from etils.epy.text_utils import pretty_repr
from etils.epy.text_utils import pretty_repr_top_level

# Inside tests, can use `epy.testing`
if 'pytest' in sys.modules:  # < Ensure open source does not trigger import
  try:
    from etils.epy import testing  # pylint: disable=g-import-not-at-top
  except ImportError:
    pass

del sys
