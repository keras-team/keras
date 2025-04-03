# coding=utf-8
"""Operations for storing and retrieving formatting info on ast nodes."""
# Copyright 2017 Google LLC
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

PASTA_DICT = '__pasta__'


def get(node, name, default=None):
  try:
    return _formatting_dict(node).get(name, default)
  except AttributeError:
    return default


def set(node, name, value):
  if not hasattr(node, PASTA_DICT):
    try:
      setattr(node, PASTA_DICT, {})
    except AttributeError:
      pass
  _formatting_dict(node)[name] = value


def append(node, name, value):
  set(node, name, get(node, name, '') + value)


def prepend(node, name, value):
  set(node, name, value + get(node, name, ''))


def _formatting_dict(node):
  return getattr(node, PASTA_DICT)
