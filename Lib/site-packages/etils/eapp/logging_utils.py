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

"""Logging utils."""

import functools
import logging as py_logging
import sys

from etils import epy
from etils.epy import _internal

with _internal.check_missing_deps():
  # pylint: disable=g-import-not-at-top
  from absl import app
  from absl import flags
  from absl import logging as absl_logging
  # pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS


class TqdmStream:
  """File-object-like abstraction which wrap`tqdm.write`.

  By default using `logging.info` inside a `tqdm` scope creates visual
  artifacts. This simple wrapper uses `tqdm.write` instead.

  Usage:

  ```python
  logger = logging.getLogger()
  logger.addHandler(logging.StreamHandler(TqdmStream()))

  for _ in tqdm.tqdm(range(10)):
    logger.info('No visual artifacts')
  ```
  """

  def write(self, x: str) -> None:
    import tqdm  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    tqdm.tqdm.write(x, end='')

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass


def _better_logging() -> None:
  """Modify Python logging (internal)."""
  # If `absl.run` was not called (e.g. open source `pytest` tests)
  if not FLAGS.is_parsed():
    return
  # User explicitly set --logtostderr, use default behavior
  if FLAGS.logtostderr or FLAGS.alsologtostderr:
    return

  # Display logs by default
  absl_logging.use_python_logging(quiet=True)

  file_link = '{filename}:{lineno}'

  # Using cleaner, less verbose logger
  formatter = py_logging.Formatter(
      # Only display single letter level (`INFO`, `DEBUG`,... -> `I`, `D`,...)
      f'{{levelname:1.1}} {{asctime}} [{file_link}]: {{message}}',
      # Do not display date by default (take a lot of space and is almost
      # never important locally.
      # Also milliseconds feel overkill
      datefmt='%H:%M:%S',
      style='{',
  )

  python_handler = absl_logging.get_absl_handler().python_handler
  python_handler.setFormatter(formatter)

  if 'tqdm' in sys.modules:
    # Replace `sys.stderr` by the TQDM file
    # This avoid visual artifacts when `logging.info` is used inside
    # a `tqdm.tqdm` context.
    python_handler.setStream(TqdmStream())


def _terminal_link(uri: str, text: str) -> str:
  """Returns a clickable link on the terminal."""
  parameters = ''
  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  return f'\033]8;{parameters};{uri}\033\\{text}\033]8;;\033\\'


def _new_factory(old_factory, *args, **kwargs) -> py_logging.LogRecord:
  """Update the logs."""
  # TODO(epot): Add color ?
  record = old_factory(*args, **kwargs)
  return record


def better_logging():
  """Improve Python logging when running locally.

  * Display Python logs by default (even when user forgot `--logtostderr`),
    without being polluted by hundreds of C++ logs.
  * Cleaner minimal log format (e.g. `I 15:04:05 [main.py:24]:`)
  * Avoid visual artifacts between TQDM & `logging`
  * Clickable hyperlinks redirecting to code search (require terminal support)

  Usage:

  ```python
  if __name__ == '__main__':
    eapp.better_logging()
    app.run(main)
  ```

  Note this has only effect when user run locally and without `--logtostderr`.
  """
  app.call_after_init(_better_logging)
