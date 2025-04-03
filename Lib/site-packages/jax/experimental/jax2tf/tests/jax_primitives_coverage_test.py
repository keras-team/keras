# Copyright 2020 The JAX Authors.
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
"""Tests the primitive harness limitations.

Runs all the harnesses surfaces the errors, and detects cases when we have
too many or too few limitations.

"""

import collections
from collections.abc import Sequence
import datetime
import logging
import os
from typing import Any
import unittest

from absl.testing import absltest

from jax._src import config
from jax._src import test_util as jtu

import numpy as np

config.parse_flags_with_absl()

# Import after parsing flags
from jax._src.internal_test_util import test_harnesses


@jtu.with_config(jax_legacy_prng_key='allow',
                 jax_debug_key_reuse=False)
class JaxPrimitiveTest(jtu.JaxTestCase):

  # This test runs for all primitive harnesses. For each primitive "xxx" the
  # test will be called "test_jax_implemented_xxx_...". The test harnesses,
  # including which dtypes are expected to fail, are defined in the
  # file test_harnesses.py.
  # If you want to run this test for only one harness, add parameter
  # `one_containing="foo"` to parameterized below.
  @test_harnesses.parameterized(test_harnesses.all_harnesses,
                                #one_containing="",
                                include_jax_unimpl=True)
  @jtu.ignore_warning(category=UserWarning,
                      message="Using reduced precision for gradient.*")
  def test_jax_implemented(self, harness: test_harnesses.Harness):
    """Runs all harnesses just with JAX to verify the jax_unimplemented field.

    Runs also harnesses that have jax_unimplemented but ignores their errors.
    """
    jax_unimpl = [l for l in harness.jax_unimplemented
                  if l.filter(device=jtu.device_under_test(),
                              dtype=harness.dtype)]
    if any(lim.skip_run for lim in jax_unimpl):
      logging.info(
          "Skipping run with expected JAX limitations: %s in harness %s",
          [u.description for u in jax_unimpl], harness.fullname)
      return
    try:
      harness.dyn_fun(*harness.dyn_args_maker(self.rng()))
    except Exception as e:
      if jax_unimpl:
        logging.info(
          "Found expected JAX error %s with expected JAX limitations: "
          "%s in harness %s",
          e, [u.description for u in jax_unimpl], harness.fullname)
        return
      else:
        raise e

    if jax_unimpl:
      logging.warning("Found no JAX error but expected JAX limitations: %s in "
                      "harness: %s",
                      [u.description for u in jax_unimpl], harness.fullname)
      # We do not fail the test if we have too many limitations. If you want
      # to find extraneous limitations, uncomment this assert and run the test
      # on all platforms.
      # self.assertEmpty(("Found no JAX error but expected JAX limitations: "
      #                  f"{[u.description for u in jax_unimpl]} in harness: {harness.fullname}"))

  def test_generate_primitives_coverage_doc(self):
    harnesses = test_harnesses.all_harnesses
    print(f"Found {len(harnesses)} harnesses")

    harness_groups: dict[str, Sequence[test_harnesses.Harness]] = collections.defaultdict(list)

    def unique_hash(h: test_harnesses.Harness, l: test_harnesses.Limitation):
      return (h.group_name, l.description, l.devices,
              tuple(np.dtype(d).name for d in l.dtypes))

    unique_limitations: dict[Any, tuple[test_harnesses.Harness,
                                        test_harnesses.Limitation]] = {}

    for h in harnesses:
      harness_groups[h.group_name].append(h)
      for l in h.jax_unimplemented:
        if l.enabled:
          unique_limitations[hash(unique_hash(h, l))] = (h, l)

    primitive_coverage_table = ["""
| Primitive | Total test harnesses | dtypes supported on at least one device | dtypes NOT tested on any device |
| --- | --- | --- | --- |"""]
    all_dtypes = set(jtu.dtypes.all)

    for group_name in sorted(harness_groups.keys()):
      hlist = harness_groups[group_name]
      dtypes_tested = set()  # Tested on at least some device
      for h in hlist:
        dtypes_tested = dtypes_tested.union({h.dtype})

      primitive_coverage_table.append(
        f"| {group_name} | {len(hlist)} | "
        f"{test_harnesses.dtypes_to_str(dtypes_tested)} | "
        f"{test_harnesses.dtypes_to_str(all_dtypes - dtypes_tested)} |")

    print(f"Found {len(unique_limitations)} unique limitations")
    primitive_unimpl_table = ["""
| Affected primitive | Description of limitation | Affected dtypes | Affected devices |
| --- | --- | --- | --- |"""]
    for h, l in sorted(
        unique_limitations.values(), key=lambda pair: unique_hash(*pair)):
      devices = ", ".join(l.devices)
      primitive_unimpl_table.append(
        f"|{h.group_name}|{l.description}|"
        f"{test_harnesses.dtypes_to_str(l.dtypes, empty_means_all=True)}|{devices}|")

    if not os.environ.get("JAX_OUTPUT_LIMITATIONS_DOC"):
      raise unittest.SkipTest("Set JAX_OUTPUT_LIMITATIONS_DOC=1 to enable the generation of the documentation")
    # The CPU/GPU have more supported types than TPU.
    self.assertEqual("cpu", jtu.device_under_test(), "The documentation can be generated only on CPU")
    self.assertTrue(config.enable_x64.value, "The documentation must be generated with JAX_ENABLE_X64=1")

    with open(os.path.join(os.path.dirname(__file__),
                           '../g3doc/jax_primitives_coverage.md.template')) as f:
      template = f.read()
    output_file = os.path.join(os.path.dirname(__file__),
                               '../g3doc/jax_primitives_coverage.md')

    with open(output_file, "w") as f:
      f.write(template.replace("{{generation_date}}", str(datetime.date.today())) \
              .replace("{{nr_harnesses}}", str(len(harnesses))) \
              .replace("{{nr_primitives}}", str(len(harness_groups))) \
              .replace("{{primitive_unimpl_table}}", "\n".join(primitive_unimpl_table)) \
              .replace("{{primitive_coverage_table}}", "\n".join(primitive_coverage_table)))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
