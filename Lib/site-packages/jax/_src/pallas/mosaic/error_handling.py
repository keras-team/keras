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

"""Utilities for raising more informative exceptions from Pallas."""
from collections import namedtuple
import re
import types
from jax._src import compiler
from jax._src import traceback_util
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir

# This is a simple ir.Location parsing regex that assumes the string is properly
# formatted coming from Mosaic.
# It will assume everything from the first to last parentheses
# in the string is part of the frame, and does not account for unbalanced
# parentheses.
LOCATION_PATTERN = re.compile(
    r'(?P<location>loc\((?P<eqn_str>\".*?\")(?P<frames>.*)\))'
)
FRAME_PATTERN = re.compile(
    r'(?P<fun_name>\".*?\")\((?P<filename>\".*?\"):'
    r'(?P<lineno>[0-9]+):(?P<colno>[0-9]+)\)'
)
MLIR_ERR_PREFIX = (
    'Pallas encountered an internal verification error.'
    'Please file a bug at https://github.com/jax-ml/jax/issues. '
    'Error details: '
)

RawFrame = namedtuple('RawFrame', ['func_name', 'filename', 'lineno', 'colno'])


class MosaicError(Exception):
  """Error thrown by Pallas when re-raising a Mosaic internal error."""


class VerificationError(MosaicError):
  """Error thrown by Pallas when re-raising a verification error."""

  def __init__(self, message: str):
    super().__init__(MLIR_ERR_PREFIX + message)


def _handle_xla_runtime_error(
    base_err: xla_client.XlaRuntimeError,
) -> MosaicError | None:
  """Reformats XLARuntimeError to include a Python traceback."""
  if 'Mosaic' not in str(base_err):
    return None
  try:
    _, frames = parse_location_string(str(base_err))
  except ValueError:
    # If no location string is found, skip handling and raise the original
    # error.
    return None
  new_tb = traceback_from_raw_frames(frames)
  err_msg = base_err.args[0]
  err_msg = redact_locations(err_msg)
  new_error = MosaicError(err_msg)
  new_error.__traceback__ = traceback_util.filter_traceback(new_tb)
  return new_error


compiler.register_xla_runtime_error_handler(_handle_xla_runtime_error)


def mlir_error_to_verification_error(
    base_err: ir.MLIRError) -> VerificationError:
  """Reformats MLIRError to include a Python traceback."""
  diagnostic = base_err.error_diagnostics[0]  # pytype: disable=attribute-error
  def _get_diagnostic_message(diagnostic) -> str:
    current_msg = diagnostic.message
    for d in diagnostic.notes:
      current_msg += "\n " + _get_diagnostic_message(d)
    return current_msg

  _, frames = parse_location_string(str(diagnostic.location.attr))
  new_tb = traceback_from_raw_frames(frames)
  new_error = VerificationError(_get_diagnostic_message(diagnostic))
  new_error.__traceback__ = traceback_util.filter_traceback(new_tb)
  return new_error


def redact_locations(err_msg: str) -> str:
  """Removes location strings from an error message."""
  for mat in re.finditer(LOCATION_PATTERN, err_msg):
    start, end = mat.span('location')
    # Remove the entire line containing the location.
    line_start = err_msg.rfind('\n', 0, end)
    line_start = line_start if line_start >= 0 else start
    line_end = err_msg.find('\n', start)
    line_end = line_end if line_end >= 0 else end
    return err_msg[:line_start] + err_msg[line_end+1:]
  return err_msg


def parse_location_string(location_string: str) -> tuple[str, list[RawFrame]]:
  """Parses a serialized MLIR location.

  Locations strings have the format:
  `loc("location_name"(<callsite>))`

  Where <callsite> is a nested callsite string representing the entire
  call stack:
  `callsite("fn_name"("filename":lineno:colno) at callsite(...))`

  Args:
    location_string: A string serialization of an MLIR location.

  Returns:
    A tuple (name, frames) where name is the name of the location and frames
    is a list of RawFrame objects representing the Python call stack associated
    with the location.
  """
  frame_str = ''
  loc_name = None
  matches = list(re.finditer(LOCATION_PATTERN, location_string))
  if len(matches) > 1:
    raise ValueError(
        'More than one location found in string: ', location_string)
  for mat in matches:
    loc_name = mat.group('eqn_str')[1:-1]
    frame_str = mat.group('frames')[1:-1]
  if loc_name is None:
    raise ValueError(f'Could not find location in string {location_string}')
  frames: list[RawFrame] = []
  for mat in re.finditer(FRAME_PATTERN, frame_str):
    frames.append(
        RawFrame(
            mat.group('fun_name')[1:-1],
            mat.group('filename')[1:-1],
            int(mat.group('lineno')),
            int(mat.group('colno')),
        )
    )
  return loc_name, frames


def traceback_from_raw_frames(frames: list[RawFrame]) -> types.TracebackType:
  """Constructs a traceback from a list of RawFrame objects."""
  xla_frames = [
    xla_client.Frame(frame.filename, frame.func_name, -1, frame.lineno)
    for frame in frames
  ]
  return xla_client.Traceback.traceback_from_frames(xla_frames)
