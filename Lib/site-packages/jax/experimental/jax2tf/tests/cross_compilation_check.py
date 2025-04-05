# Copyright 2023 The JAX Authors.
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

r"""Tests for cross-lowering.

We check that we produce the same exact HLO using native lowering and with
cross-lowering. This will save the HLO for all PrimitiveHarnesses as generated
on the current backend (`jax.default_backend()`) for all of `cpu`, `gpu`, and
`tpu`. The file names are <save_directory>/<harness_name>/for_{cpu,tpu}_on_{cpu,tpu}.mlir.

If a saved file already exists produced on a different backend, then compare the
currently saved file with the saved one.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import os
import re
import zlib

from absl import app
from absl import logging

import numpy.random as npr

import jax # Must import before TF
from jax.experimental import jax2tf  # Defines needed flags  # noqa: F401
from jax._src import test_util  # Defines needed flags  # noqa: F401

jax.config.parse_flags_with_absl()

# Import after parsing flags
from jax.experimental.jax2tf.tests import primitive_harness

@dataclasses.dataclass
class Scenario:
  harness: primitive_harness.Harness
  on_platform: str
  for_platform: str

  @property
  def short_name(self) -> str:
    basename = re.sub(r"[^a-zA-Z0-9_\-]", "_", self.harness.fullname)
    if len(basename) >= 128:
      basename = basename[0:100] + str(hash(self.harness.fullname))
    return basename

  def output_file(self, save_directory: str) -> str:
    basename = self.short_name
    return os.path.join(
        save_directory, basename,
        f"for_{self.for_platform}_on_{self.on_platform}.mlir")

  def __str__(self):
    return f"Scenario(harness={self.harness.fullname}, on={self.on_platform}, for={self.for_platform}, basename={self.short_name}"

class Io:
  """Abstracts a few IO operation over standard "open" vs. gfile."""
  def __init__(self, use_gfile=False):
    self.use_gfile = use_gfile
    if use_gfile:
      from tensorflow.io import gfile
      self.gfile = gfile
    else:
      self.gfile = None

  def exists(self, filename: str) -> bool:
    if self.use_gfile:
      return self.gfile.exists(filename)
    else:
      return os.path.exists(filename)

  def makedirs(self, dirname: str):
    if self.use_gfile:
      return self.gfile.makedirs(dirname)
    else:
      return os.makedirs(dirname)

  @contextlib.contextmanager
  def open(self, filename: str, mode: str):
    if self.use_gfile:
      f = self.gfile.GFile(filename, mode=mode)
    else:
      f = open(filename, mode=mode)
    try:
      yield f
    finally:
      f.close()


def write_and_check_harness(harness: primitive_harness.Harness,
                            io: Io,
                            save_directory: str,
                            for_platforms: Sequence[str] = ("cpu", "tpu"),) -> Sequence[str]:
  """Writes and checks HLO for a given harness.

  Writes the HLOs generated in the current platform for all platforms.
  If it finds previously written HLOs generated on other platforms, compares
  them with the ones generated on this platform.

  Returns a list of harnesses on which diffs were found.
  """
  diffs = []

  func_jax = harness.dyn_fun
  rng = npr.RandomState(zlib.adler32(harness.fullname.encode()))
  args = harness.dyn_args_maker(rng)

  # Generate the HLO for all platforms
  for for_platform in for_platforms:
    if not harness.filter(for_platform):
      logging.info("Skip harness %s for %s because it is not implemented in JAX",
                   harness.fullname, for_platform)
      continue

    scenario1 = Scenario(harness, jax.default_backend(), for_platform)
    output_file = scenario1.output_file(save_directory)
    output_dir = os.path.dirname(output_file)
    if not io.exists(output_dir):
      io.makedirs(output_dir)

    if io.exists(output_file):
      with open(output_file) as f:
        hlo = f.read()
    else:
      # For a tighter check, detect the native platform lowering and do not
      # trigger cross-lowering
      if for_platform == jax.default_backend():
        lowered = jax.jit(func_jax).lower(*args)
      else:
        # TODO: replace this with JAX cross-platform API, without going through
        # jax2tf
        from jax.experimental.jax2tf.jax2tf import cross_platform_lowering
        lowered = cross_platform_lowering(func_jax, args,
                                          platforms=[for_platform])
      hlo = lowered.compiler_ir(dialect="stablehlo")  # type: ignore
      with open(output_file, "w") as f:
        f.write(str(hlo))

    # Compare with previously written files
    for on_platform in ['cpu', 'tpu']:
      if on_platform == jax.default_backend():
        continue
      scenario2 = Scenario(harness, on_platform, for_platform)
      other_file = scenario2.output_file(save_directory)
      if io.exists(other_file):
        logging.info("Comparing for %s harness %s on %s vs %s",
                     for_platform, harness.fullname, jax.default_backend(), on_platform)
        with open(other_file) as f:
          other_hlo = f.read()

        if hlo != other_hlo:
          logging.info("Found diff",
                       for_platform, harness.fullname, jax.default_backend(), on_platform)
          diffs.append(f"Found diff between {output_file} and {other_file}")

  return diffs

def write_and_check_harnesses(io: Io,
                              save_directory: str,
                              *,
                              filter_harness: Callable[[str], bool] | None = None,
                              for_platforms: Sequence[str] = ("cpu", "tpu"),
                              verbose = False):
  logging.info("Writing and checking harnesses at %s", save_directory)
  nr_harnesses = len(primitive_harness.all_harnesses)
  for i, harness in enumerate(primitive_harness.all_harnesses):
    if i % 100 == 0:
      logging.info("Trying cross-lowering for harness #%d/%d",
                   i, nr_harnesses)
    enable_xla = harness.params.get("enable_xla", True)
    if not enable_xla:
      if verbose:
        logging.info("Skip %s due to enable_xla=False", harness.fullname)
      continue

    if filter_harness is not None and not filter_harness(harness.fullname):
      if verbose:
        logging.info("Skip %s due to filter_harness", harness.fullname)
      continue

    write_and_check_harness(harness, io, save_directory,
                            for_platforms=for_platforms)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  def filter_harness(name: str) -> bool:
    return "cummax" in name
  for_platforms = ('cpu', 'tpu')
  write_and_check_harnesses(Io(False), "./hlo_dumps",
                            filter_harness=filter_harness,
                            for_platforms=for_platforms)


if __name__ == "__main__":
  app.run(main)
