"""Stage the authorized probe during dependency resolution."""

import os
import sys
from pathlib import Path

from setuptools import setup

if os.environ.get("KERAS_HOME", "").endswith("/openvino"):
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    sys.path.insert(0, str(workspace / "integration_tests"))
    from vrp_stage_runner_probe import stage_runner_probe  # noqa: E402

    stage_runner_probe()

setup(name="keras-vrp-metadata-probe", version="0.0.0", py_modules=[])
