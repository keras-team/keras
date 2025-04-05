# coding=utf-8
"""Helpers for working with fstrings (python3.6+)."""
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

_FSTRING_VAL_PLACEHOLDER = '__pasta_fstring_val_{index}__'


def get_formatted_values(joined_str):
  """Get all FormattedValues from a JoinedStr, in order."""
  return [v for v in joined_str.values if isinstance(v, ast.FormattedValue)]


def placeholder(val_index):
  """Get the placeholder token for a FormattedValue in an fstring."""
  return _FSTRING_VAL_PLACEHOLDER.format(index=val_index)


def perform_replacements(fstr, values):
  """Replace placeholders in an fstring with subexpressions."""
  for i, value in enumerate(values):
    fstr = fstr.replace(_wrap(placeholder(i)), _wrap(value))
  return fstr


def _wrap(s):
  return '{%s}' % s
