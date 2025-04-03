# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import tempfile

import pyproject_hooks

from . import ProjectBuilder
from ._compat import importlib
from ._types import StrPath, SubprocessRunner
from .env import DefaultIsolatedEnv


def _project_wheel_metadata(builder: ProjectBuilder) -> importlib.metadata.PackageMetadata:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(builder.metadata_path(tmpdir))
        return importlib.metadata.PathDistribution(path).metadata


def project_wheel_metadata(
    source_dir: StrPath,
    isolated: bool = True,
    *,
    runner: SubprocessRunner = pyproject_hooks.quiet_subprocess_runner,
) -> importlib.metadata.PackageMetadata:
    """
    Return the wheel metadata for a project.

    Uses the ``prepare_metadata_for_build_wheel`` hook if available,
    otherwise ``build_wheel``.

    :param source_dir: Project source directory
    :param isolated: Whether or not to run invoke the backend in the current
                     environment or to create an isolated one and invoke it
                     there.
    :param runner: An alternative runner for backend subprocesses
    """

    if isolated:
        with DefaultIsolatedEnv() as env:
            builder = ProjectBuilder.from_isolated_env(
                env,
                source_dir,
                runner=runner,
            )
            env.install(builder.build_system_requires)
            env.install(builder.get_requires_for_build('wheel'))
            return _project_wheel_metadata(builder)
    else:
        builder = ProjectBuilder(
            source_dir,
            runner=runner,
        )
        return _project_wheel_metadata(builder)


__all__ = [
    'project_wheel_metadata',
]
