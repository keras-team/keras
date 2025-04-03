# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import difflib
import os
import subprocess
import sys
import warnings
import zipfile

from collections.abc import Iterator
from typing import Any, Mapping, Sequence, TypeVar

import pyproject_hooks

from . import _ctx, env
from ._compat import tomllib
from ._exceptions import (
    BuildBackendException,
    BuildException,
    BuildSystemTableValidationError,
    TypoWarning,
)
from ._types import ConfigSettings, Distribution, StrPath, SubprocessRunner
from ._util import check_dependency, parse_wheel_filename


_TProjectBuilder = TypeVar('_TProjectBuilder', bound='ProjectBuilder')


_DEFAULT_BACKEND = {
    'build-backend': 'setuptools.build_meta:__legacy__',
    'requires': ['setuptools >= 40.8.0'],
}


def _find_typo(dictionary: Mapping[str, str], expected: str) -> None:
    for obj in dictionary:
        if difflib.SequenceMatcher(None, expected, obj).ratio() >= 0.8:
            warnings.warn(
                f"Found '{obj}' in pyproject.toml, did you mean '{expected}'?",
                TypoWarning,
                stacklevel=2,
            )


def _validate_source_directory(source_dir: StrPath) -> None:
    if not os.path.isdir(source_dir):
        msg = f'Source {source_dir} is not a directory'
        raise BuildException(msg)
    pyproject_toml = os.path.join(source_dir, 'pyproject.toml')
    setup_py = os.path.join(source_dir, 'setup.py')
    if not os.path.exists(pyproject_toml) and not os.path.exists(setup_py):
        msg = f'Source {source_dir} does not appear to be a Python project: no pyproject.toml or setup.py'
        raise BuildException(msg)


def _read_pyproject_toml(path: StrPath) -> Mapping[str, Any]:
    try:
        with open(path, 'rb') as f:
            return tomllib.loads(f.read().decode())
    except FileNotFoundError:
        return {}
    except PermissionError as e:
        msg = f"{e.strerror}: '{e.filename}' "
        raise BuildException(msg) from None
    except tomllib.TOMLDecodeError as e:
        msg = f'Failed to parse {path}: {e} '
        raise BuildException(msg) from None


def _parse_build_system_table(pyproject_toml: Mapping[str, Any]) -> Mapping[str, Any]:
    # If pyproject.toml is missing (per PEP 517) or [build-system] is missing
    # (per PEP 518), use default values
    if 'build-system' not in pyproject_toml:
        _find_typo(pyproject_toml, 'build-system')
        return _DEFAULT_BACKEND

    build_system_table = dict(pyproject_toml['build-system'])

    # If [build-system] is present, it must have a ``requires`` field (per PEP 518)
    if 'requires' not in build_system_table:
        _find_typo(build_system_table, 'requires')
        msg = '`requires` is a required property'
        raise BuildSystemTableValidationError(msg)
    elif not isinstance(build_system_table['requires'], list) or not all(
        isinstance(i, str) for i in build_system_table['requires']
    ):
        msg = '`requires` must be an array of strings'
        raise BuildSystemTableValidationError(msg)

    if 'build-backend' not in build_system_table:
        _find_typo(build_system_table, 'build-backend')
        # If ``build-backend`` is missing, inject the legacy setuptools backend
        # but leave ``requires`` intact to emulate pip
        build_system_table['build-backend'] = _DEFAULT_BACKEND['build-backend']
    elif not isinstance(build_system_table['build-backend'], str):
        msg = '`build-backend` must be a string'
        raise BuildSystemTableValidationError(msg)

    if 'backend-path' in build_system_table and (
        not isinstance(build_system_table['backend-path'], list)
        or not all(isinstance(i, str) for i in build_system_table['backend-path'])
    ):
        msg = '`backend-path` must be an array of strings'
        raise BuildSystemTableValidationError(msg)

    unknown_props = build_system_table.keys() - {'requires', 'build-backend', 'backend-path'}
    if unknown_props:
        msg = f'Unknown properties: {", ".join(unknown_props)}'
        raise BuildSystemTableValidationError(msg)

    return build_system_table


def _wrap_subprocess_runner(runner: SubprocessRunner, env: env.IsolatedEnv) -> SubprocessRunner:
    def _invoke_wrapped_runner(
        cmd: Sequence[str], cwd: str | None = None, extra_environ: Mapping[str, str] | None = None
    ) -> None:
        runner(cmd, cwd, {**(env.make_extra_environ() or {}), **(extra_environ or {})})

    return _invoke_wrapped_runner


class ProjectBuilder:
    """
    The PEP 517 consumer API.
    """

    def __init__(
        self,
        source_dir: StrPath,
        python_executable: str = sys.executable,
        runner: SubprocessRunner = pyproject_hooks.default_subprocess_runner,
    ) -> None:
        """
        :param source_dir: The source directory
        :param python_executable: The python executable where the backend lives
        :param runner: Runner for backend subprocesses

        The ``runner``, if provided, must accept the following arguments:

        - ``cmd``: a list of strings representing the command and arguments to
          execute, as would be passed to e.g. 'subprocess.check_call'.
        - ``cwd``: a string representing the working directory that must be
          used for the subprocess. Corresponds to the provided source_dir.
        - ``extra_environ``: a dict mapping environment variable names to values
          which must be set for the subprocess execution.

        The default runner simply calls the backend hooks in a subprocess, writing backend output
        to stdout/stderr.
        """
        self._source_dir: str = os.path.abspath(source_dir)
        _validate_source_directory(source_dir)

        self._python_executable = python_executable
        self._runner = runner

        pyproject_toml_path = os.path.join(source_dir, 'pyproject.toml')
        self._build_system = _parse_build_system_table(_read_pyproject_toml(pyproject_toml_path))

        self._backend = self._build_system['build-backend']

        self._hook = pyproject_hooks.BuildBackendHookCaller(
            self._source_dir,
            self._backend,
            backend_path=self._build_system.get('backend-path'),
            python_executable=self._python_executable,
            runner=self._runner,
        )

    @classmethod
    def from_isolated_env(
        cls: type[_TProjectBuilder],
        env: env.IsolatedEnv,
        source_dir: StrPath,
        runner: SubprocessRunner = pyproject_hooks.default_subprocess_runner,
    ) -> _TProjectBuilder:
        return cls(
            source_dir=source_dir,
            python_executable=env.python_executable,
            runner=_wrap_subprocess_runner(runner, env),
        )

    @property
    def source_dir(self) -> str:
        """Project source directory."""
        return self._source_dir

    @property
    def python_executable(self) -> str:
        """
        The Python executable used to invoke the backend.
        """
        return self._python_executable

    @property
    def build_system_requires(self) -> set[str]:
        """
        The dependencies defined in the ``pyproject.toml``'s
        ``build-system.requires`` field or the default build dependencies
        if ``pyproject.toml`` is missing or ``build-system`` is undefined.
        """
        return set(self._build_system['requires'])

    def get_requires_for_build(
        self,
        distribution: Distribution,
        config_settings: ConfigSettings | None = None,
    ) -> set[str]:
        """
        Return the dependencies defined by the backend in addition to
        :attr:`build_system_requires` for a given distribution.

        :param distribution: Distribution to get the dependencies of
            (``sdist`` or ``wheel``)
        :param config_settings: Config settings for the build backend
        """
        _ctx.log(f'Getting build dependencies for {distribution}...')
        hook_name = f'get_requires_for_build_{distribution}'
        get_requires = getattr(self._hook, hook_name)

        with self._handle_backend(hook_name):
            return set(get_requires(config_settings))

    def check_dependencies(
        self,
        distribution: Distribution,
        config_settings: ConfigSettings | None = None,
    ) -> set[tuple[str, ...]]:
        """
        Return the dependencies which are not satisfied from the combined set of
        :attr:`build_system_requires` and :meth:`get_requires_for_build` for a given
        distribution.

        :param distribution: Distribution to check (``sdist`` or ``wheel``)
        :param config_settings: Config settings for the build backend
        :returns: Set of variable-length unmet dependency tuples
        """
        dependencies = self.get_requires_for_build(distribution, config_settings).union(self.build_system_requires)
        return {u for d in dependencies for u in check_dependency(d)}

    def prepare(
        self,
        distribution: Distribution,
        output_directory: StrPath,
        config_settings: ConfigSettings | None = None,
    ) -> str | None:
        """
        Prepare metadata for a distribution.

        :param distribution: Distribution to build (must be ``wheel``)
        :param output_directory: Directory to put the prepared metadata in
        :param config_settings: Config settings for the build backend
        :returns: The full path to the prepared metadata directory
        """
        _ctx.log(f'Getting metadata for {distribution}...')
        try:
            return self._call_backend(
                f'prepare_metadata_for_build_{distribution}',
                output_directory,
                config_settings,
                _allow_fallback=False,
            )
        except BuildBackendException as exception:
            if isinstance(exception.exception, pyproject_hooks.HookMissing):
                return None
            raise

    def build(
        self,
        distribution: Distribution,
        output_directory: StrPath,
        config_settings: ConfigSettings | None = None,
        metadata_directory: str | None = None,
    ) -> str:
        """
        Build a distribution.

        :param distribution: Distribution to build (``sdist`` or ``wheel``)
        :param output_directory: Directory to put the built distribution in
        :param config_settings: Config settings for the build backend
        :param metadata_directory: If provided, should be the return value of a
            previous ``prepare`` call on the same ``distribution`` kind
        :returns: The full path to the built distribution
        """
        _ctx.log(f'Building {distribution}...')
        kwargs = {} if metadata_directory is None else {'metadata_directory': metadata_directory}
        return self._call_backend(f'build_{distribution}', output_directory, config_settings, **kwargs)

    def metadata_path(self, output_directory: StrPath) -> str:
        """
        Generate the metadata directory of a distribution and return its path.

        If the backend does not support the ``prepare_metadata_for_build_wheel``
        hook, a wheel will be built and the metadata will be extracted from it.

        :param output_directory: Directory to put the metadata distribution in
        :returns: The path of the metadata directory
        """
        # prepare_metadata hook
        metadata = self.prepare('wheel', output_directory)
        if metadata is not None:
            return metadata

        # fallback to build_wheel hook
        wheel = self.build('wheel', output_directory)
        match = parse_wheel_filename(os.path.basename(wheel))
        if not match:
            msg = 'Invalid wheel'
            raise ValueError(msg)
        distinfo = f"{match['distribution']}-{match['version']}.dist-info"
        member_prefix = f'{distinfo}/'
        with zipfile.ZipFile(wheel) as w:
            w.extractall(
                output_directory,
                (member for member in w.namelist() if member.startswith(member_prefix)),
            )
        return os.path.join(output_directory, distinfo)

    def _call_backend(
        self, hook_name: str, outdir: StrPath, config_settings: ConfigSettings | None = None, **kwargs: Any
    ) -> str:
        outdir = os.path.abspath(outdir)

        callback = getattr(self._hook, hook_name)

        if os.path.exists(outdir):
            if not os.path.isdir(outdir):
                msg = f"Build path '{outdir}' exists and is not a directory"
                raise BuildException(msg)
        else:
            os.makedirs(outdir)

        with self._handle_backend(hook_name):
            basename: str = callback(outdir, config_settings, **kwargs)

        return os.path.join(outdir, basename)

    @contextlib.contextmanager
    def _handle_backend(self, hook: str) -> Iterator[None]:
        try:
            yield
        except pyproject_hooks.BackendUnavailable as exception:
            raise BuildBackendException(
                exception,
                f"Backend '{self._backend}' is not available.",
                sys.exc_info(),
            ) from None
        except subprocess.CalledProcessError as exception:
            raise BuildBackendException(exception, f'Backend subprocess exited when trying to invoke {hook}') from None
        except Exception as exception:
            raise BuildBackendException(exception, exc_info=sys.exc_info()) from None
