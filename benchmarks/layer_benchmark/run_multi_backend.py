"""Run a layer benchmark once per backend and compare the throughputs."""

import argparse
import json
import os
import subprocess
import sys

_RESULT_PREFIX = "BENCHMARK_RESULT:"


def run_backend(backend, module, flags):
    env = dict(os.environ, KERAS_BACKEND=backend, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-m", module, *flags, "--print_results_json"]
    results = []
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, text=True
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            if _RESULT_PREFIX in line:
                results.append(json.loads(line.split(_RESULT_PREFIX, 1)[1]))
    return proc.returncode == 0, results


def print_table(backends, results):
    rows = {}
    for backend in backends:
        for r in results[backend]:
            key = f"{r['layer']} ({r['mode']})"
            rows.setdefault(key, {})[backend] = r["throughput"]
    if not rows:
        print("No benchmark results collected.")
        return
    print("\nThroughput (samples/sec)\n")
    width = max(len(key) for key in rows) + 2
    print(f"{'Benchmark':<{width}}" + "".join(f"{b:>15}" for b in backends))
    for key, values in rows.items():
        cells = "".join(
            f"{values[b]:>15.2f}" if b in values else f"{'n/a':>15}"
            for b in backends
        )
        print(f"{key:<{width}}{cells}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other flags are passed on to the benchmark module.",
    )
    parser.add_argument(
        "--backends",
        default="jax,torch,tensorflow",
        help="Comma separated backends to run. Defaults to %(default)s.",
    )
    parser.add_argument(
        "module",
        help="Benchmark module to run, like "
        "benchmarks.layer_benchmark.conv_benchmark.",
    )
    args, flags = parser.parse_known_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not backends:
        parser.error("--backends must not be empty")

    results, failed = {}, []
    for backend in backends:
        print(f"=== {backend} ===", flush=True)
        ok, results[backend] = run_backend(backend, args.module, flags)
        if not ok:
            failed.append(backend)

    print_table(backends, results)
    if failed:
        sys.exit(f"Failed backends: {', '.join(failed)}")


if __name__ == "__main__":
    main()
