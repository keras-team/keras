# Copyright 2024 The Treescope Authors.
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

"""Utilities for escaping HTML values."""
import re


def escape_html_attribute(attribute: str) -> str:
  """Escapes a string for rendering in a HTML attribute."""
  return attribute.replace("&", "&amp;").replace('"', "&quot;")


def without_repeated_whitespace(s: str) -> str:
  """Replaces all repeated whitespace characters with single spaces."""
  return " ".join(s.split())


def heuristic_strip_javascript_comments(js_source: str) -> str:
  """Heuristically removes javascript comments.

  Transforms a string by removing anything between "/*" and "*/", and anything
  between "//" and the next newline. This will remove JavaScript comments. Note
  that this operates directly on the unparsed string and is not guaranteed to
  preserve the semantics of the original program (for instance, it will still
  replace "comments" that appear inside string literals). However, it works
  well enough for simple JavaScript logic.

  Args:
    js_source: String to strip comments from.

  Returns:
    Version of the input string with things that look like JavaScript comments
    removed.
  """
  no_block_comments = re.sub(
      r"/\*((?!\*/).)*\*/", "", js_source, flags=re.MULTILINE | re.DOTALL
  )
  no_line_comments = re.sub(r"//[^\n]*\n", "\n", no_block_comments)
  return no_line_comments
