# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Benchmarks for utilities working with arbitrarily nested structures."""

import collections
import timeit

import tree


TIME_UNITS = [
    (1, "s"),
    (10**-3, "ms"),
    (10**-6, "us"),
    (10**-9, "ns"),
]


def format_time(time):
  for d, unit in TIME_UNITS:
    if time > d:
      return "{:.2f}{}".format(time / d, unit)


def run_benchmark(benchmark_fn, num_iters):
  times = timeit.repeat(benchmark_fn, repeat=2, number=num_iters)
  return times[-1] / num_iters  # Discard the first half for "warmup".


def map_to_list(func, *args):
  return list(map(func, *args))


def benchmark_map(map_fn, structure):
  def benchmark_fn():
    return map_fn(lambda v: v, structure)
  return benchmark_fn


BENCHMARKS = collections.OrderedDict([
    ("tree_map_1", benchmark_map(tree.map_structure, [0])),
    ("tree_map_8", benchmark_map(tree.map_structure, [0] * 8)),
    ("tree_map_64", benchmark_map(tree.map_structure, [0] * 64)),
    ("builtin_map_1", benchmark_map(map_to_list, [0])),
    ("builtin_map_8", benchmark_map(map_to_list, [0] * 8)),
    ("builtin_map_64", benchmark_map(map_to_list, [0] * 64)),
])


def main():
  for name, benchmark_fn in BENCHMARKS.items():
    print(name, format_time(run_benchmark(benchmark_fn, num_iters=1000)))


if __name__ == "__main__":
  main()
