# Copyright 2024 The JAX Authors.
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

# Thread-safe utilities for catching and testing for warnings.
#
# The Python warnings module, at least as of Python 3.13, is not thread-safe.
# The catch_warnings() feature is inherently racy, see
# https://py-free-threading.github.io/porting/#the-warnings-module-is-not-thread-safe
#
# This module offers a thread-safe way to catch and record warnings. We install
# a custom showwarning hook with the Python warning module, and then rely on
# the CPython warnings module to call our show warning function. We then use it
# to create our own thread-safe warning filtering utilities.

import contextlib
import re
import threading
import warnings


class _WarningContext(threading.local):
  "Thread-local state that contains a list of warning handlers."

  def __init__(self):
    self.handlers = []


_context = _WarningContext()


# Callback that applies the handlers in reverse order. If no handler matches,
# we raise an error.
def _showwarning(message, category, filename, lineno, file=None, line=None):
  for handler in reversed(_context.handlers):
    if handler(message, category, filename, lineno, file, line):
      return
  raise category(message)


@contextlib.contextmanager
def raise_on_warnings():
  "Context manager that raises an exception if a warning is raised."
  if warnings.showwarning is not _showwarning:
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      yield
    return

  def handler(message, category, filename, lineno, file=None, line=None):
    raise category(message)

  _context.handlers.append(handler)
  try:
    yield
  finally:
    _context.handlers.pop()


@contextlib.contextmanager
def record_warnings():
  "Context manager that yields a list of warnings that are raised."
  if warnings.showwarning is not _showwarning:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      yield w
    return

  log = []

  def handler(message, category, filename, lineno, file=None, line=None):
    log.append(warnings.WarningMessage(message, category, filename, lineno, file, line))
    return True

  _context.handlers.append(handler)
  try:
    yield log
  finally:
    _context.handlers.pop()


@contextlib.contextmanager
def ignore_warning(*, message: str | None = None, category: type = Warning):
  "Context manager that ignores any matching warnings."
  if warnings.showwarning is not _showwarning:
    with warnings.catch_warnings():
      warnings.filterwarnings(
        "ignore", message="" if message is None else message, category=category)
      yield
    return

  if message:
    message_re = re.compile(message)
  else:
    message_re = None

  category_cls = category

  def handler(message, category, filename, lineno, file=None, line=None):
    text = str(message) if isinstance(message, Warning) else message
    if (message_re is None or message_re.match(text)) and issubclass(
        category, category_cls
    ):
      return True
    return False

  _context.handlers.append(handler)
  try:
    yield
  finally:
    _context.handlers.pop()


def install_threadsafe_warning_handlers():
  # Hook the showwarning method. The warnings module explicitly notes that
  # this is a function that users may replace.
  warnings.showwarning = _showwarning

  # Set the warnings module to always display warnings. We hook into it by
  # overriding the "showwarning" method, so it's important that all warnings
  # are "shown" by the usual mechanism.
  warnings.simplefilter("always")
