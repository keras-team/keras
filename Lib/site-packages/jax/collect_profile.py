# Copyright 2022 The JAX Authors.
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

from __future__ import annotations

import argparse
import gzip
import os
import pathlib
import tempfile

# pytype: disable=import-error
from jax._src import profiler as jax_profiler
try:
  from tensorflow.python.profiler import profiler_v2 as profiler
  from tensorflow.python.profiler import profiler_client
except ImportError:
  raise ImportError("This script requires `tensorflow` to be installed.")
try:
  from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
except ImportError:
  raise ImportError(
      "This script requires `tensorboard_plugin_profile` to be installed.")
# pytype: enable=import-error


_DESCRIPTION = """
To profile running JAX programs, you first need to start the profiler server
in the program of interest. You can do this via
`jax.profiler.start_server(<port>)`. Once the program is running and the
profiler server has started, you can run `collect_profile` to trace the execution
for a provided duration. The trace file will be dumped into a directory
(determined by `--log_dir`) and by default, a Perfetto UI link will be generated
to view the resulting trace.
"""
parser = argparse.ArgumentParser(description=_DESCRIPTION)
parser.add_argument("--log_dir", default=None,
                    help=("Directory to store log files. "
                          "Uses a temporary directory if none provided."),
                    type=str)
parser.add_argument("port", help="Port to collect trace", type=int)
parser.add_argument("duration_in_ms",
                    help="Duration to collect trace in milliseconds", type=int)
parser.add_argument("--no_perfetto_link",
                    help="Disable creating a perfetto link",
                    action="store_true")
parser.add_argument("--host", default="127.0.0.1",
                    help="Host to collect trace. Defaults to 127.0.0.1",
                    type=str)
parser.add_argument("--host_tracer_level", default=2,
                    help="Profiler host tracer level", type=int)
parser.add_argument("--device_tracer_level", default=1,
                    help="Profiler device tracer level", type=int)
parser.add_argument("--python_tracer_level", default=1,
                    help="Profiler Python tracer level", type=int)

def collect_profile(port: int, duration_in_ms: int, host: str,
                    log_dir: os.PathLike | str | None, host_tracer_level: int,
                    device_tracer_level: int, python_tracer_level: int,
                    no_perfetto_link: bool):
  options = profiler.ProfilerOptions(
      host_tracer_level=host_tracer_level,
      device_tracer_level=device_tracer_level,
      python_tracer_level=python_tracer_level,
  )
  log_dir_ = pathlib.Path(log_dir if log_dir is not None else tempfile.mkdtemp())
  profiler_client.trace(
      f"{host}:{port}",
      str(log_dir_),
      duration_in_ms,
      options=options)
  print(f"Dumped profiling information in: {log_dir_}")
  # The profiler dumps `xplane.pb` to the logging directory. To upload it to
  # the Perfetto trace viewer, we need to convert it to a `trace.json` file.
  # We do this by first finding the `xplane.pb` file, then passing it into
  # tensorflow_profile_plugin's `xplane` conversion function.
  curr_path = log_dir_.resolve()
  root_trace_folder = curr_path / "plugins" / "profile"
  trace_folders = [root_trace_folder / trace_folder for trace_folder
                   in root_trace_folder.iterdir()]
  latest_folder = max(trace_folders, key=os.path.getmtime)
  xplane = next(latest_folder.glob("*.xplane.pb"))
  result, _ = convert.xspace_to_tool_data([xplane], "trace_viewer^", {})

  with gzip.open(str(latest_folder / "remote.trace.json.gz"), "wb") as fp:
    fp.write(result.encode("utf-8"))

  if not no_perfetto_link:
    path = jax_profiler._write_perfetto_trace_file(log_dir_)
    jax_profiler._host_perfetto_trace_file(path)

def main(args):
  collect_profile(args.port, args.duration_in_ms, args.host, args.log_dir,
                  args.host_tracer_level, args.device_tracer_level,
                  args.python_tracer_level, args.no_perfetto_link)

if __name__ == "__main__":
  main(parser.parse_args())
