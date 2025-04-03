# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import contextlib
import contextvars
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import warnings

from collections.abc import Iterator, Sequence
from functools import partial
from typing import NoReturn, TextIO

import build

from . import ProjectBuilder, _ctx
from . import env as _env
from ._exceptions import BuildBackendException, BuildException, FailedProcessError
from ._types import ConfigSettings, Distribution, StrPath
from .env import DefaultIsolatedEnv


_COLORS = {
    'red': '\33[91m',
    'green': '\33[92m',
    'yellow': '\33[93m',
    'bold': '\33[1m',
    'dim': '\33[2m',
    'underline': '\33[4m',
    'reset': '\33[0m',
}
_NO_COLORS = {color: '' for color in _COLORS}


_styles = contextvars.ContextVar('_styles', default=_COLORS)


def _init_colors() -> None:
    if 'NO_COLOR' in os.environ:
        if 'FORCE_COLOR' in os.environ:
            warnings.warn('Both NO_COLOR and FORCE_COLOR environment variables are set, disabling color', stacklevel=2)
        _styles.set(_NO_COLORS)
    elif 'FORCE_COLOR' in os.environ or sys.stdout.isatty():
        return
    _styles.set(_NO_COLORS)


def _cprint(fmt: str = '', msg: str = '', file: TextIO | None = None) -> None:
    print(fmt.format(msg, **_styles.get()), file=file, flush=True)


def _showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:  # pragma: no cover
    _cprint('{yellow}WARNING{reset} {}', str(message))


_max_terminal_width = shutil.get_terminal_size().columns - 2
if _max_terminal_width <= 0:
    _max_terminal_width = 78


_fill = partial(textwrap.fill, subsequent_indent='  ', width=_max_terminal_width)


def _log(message: str, *, origin: tuple[str, ...] | None = None) -> None:
    if origin is None:
        (first, *rest) = message.splitlines()
        _cprint('{bold}{}{reset}', _fill(first, initial_indent='* '))
        for line in rest:
            print(_fill(line, initial_indent='  '))

    elif origin[0] == 'subprocess':
        initial_indent = '> ' if origin[1] == 'cmd' else '< '
        file = sys.stderr if origin[1] == 'stderr' else None
        for line in message.splitlines():
            _cprint('{dim}{}{reset}', _fill(line, initial_indent=initial_indent), file=file)


def _setup_cli(*, verbosity: int) -> None:
    warnings.showwarning = _showwarning

    if platform.system() == 'Windows':
        try:
            import colorama

            colorama.init()
        except ModuleNotFoundError:
            pass

    _init_colors()

    _ctx.LOGGER.set(_log)
    _ctx.VERBOSITY.set(verbosity)


def _error(msg: str, code: int = 1) -> NoReturn:  # pragma: no cover
    """
    Print an error message and exit. Will color the output when writing to a TTY.

    :param msg: Error message
    :param code: Error code
    """
    _cprint('{red}ERROR{reset} {}', msg)
    raise SystemExit(code)


def _format_dep_chain(dep_chain: Sequence[str]) -> str:
    return ' -> '.join(dep.partition(';')[0].strip() for dep in dep_chain)


def _build_in_isolated_env(
    srcdir: StrPath,
    outdir: StrPath,
    distribution: Distribution,
    config_settings: ConfigSettings | None,
    installer: _env.Installer,
) -> str:
    with DefaultIsolatedEnv(installer=installer) as env:
        builder = ProjectBuilder.from_isolated_env(env, srcdir)
        # first install the build dependencies
        env.install(builder.build_system_requires)
        # then get the extra required dependencies from the backend (which was installed in the call above :P)
        env.install(builder.get_requires_for_build(distribution, config_settings or {}))
        return builder.build(distribution, outdir, config_settings or {})


def _build_in_current_env(
    srcdir: StrPath,
    outdir: StrPath,
    distribution: Distribution,
    config_settings: ConfigSettings | None,
    skip_dependency_check: bool = False,
) -> str:
    builder = ProjectBuilder(srcdir)

    if not skip_dependency_check:
        missing = builder.check_dependencies(distribution, config_settings or {})
        if missing:
            dependencies = ''.join('\n\t' + dep for deps in missing for dep in (deps[0], _format_dep_chain(deps[1:])) if dep)
            _cprint()
            _error(f'Missing dependencies:{dependencies}')

    return builder.build(distribution, outdir, config_settings or {})


def _build(
    isolation: bool,
    srcdir: StrPath,
    outdir: StrPath,
    distribution: Distribution,
    config_settings: ConfigSettings | None,
    skip_dependency_check: bool,
    installer: _env.Installer,
) -> str:
    if isolation:
        return _build_in_isolated_env(srcdir, outdir, distribution, config_settings, installer)
    else:
        return _build_in_current_env(srcdir, outdir, distribution, config_settings, skip_dependency_check)


@contextlib.contextmanager
def _handle_build_error() -> Iterator[None]:
    try:
        yield
    except (BuildException, FailedProcessError) as e:
        _error(str(e))
    except BuildBackendException as e:
        if isinstance(e.exception, subprocess.CalledProcessError):
            _cprint()
            _error(str(e))

        if e.exc_info:
            tb_lines = traceback.format_exception(
                e.exc_info[0],
                e.exc_info[1],
                e.exc_info[2],
                limit=-1,
            )
            tb = ''.join(tb_lines)
        else:
            tb = traceback.format_exc(-1)
        _cprint('\n{dim}{}{reset}\n', tb.strip('\n'))
        _error(str(e))
    except Exception as e:  # pragma: no cover
        tb = traceback.format_exc().strip('\n')
        _cprint('\n{dim}{}{reset}\n', tb)
        _error(str(e))


def _natural_language_list(elements: Sequence[str]) -> str:
    if len(elements) == 0:
        msg = 'no elements'
        raise IndexError(msg)
    elif len(elements) == 1:
        return elements[0]
    else:
        return '{} and {}'.format(
            ', '.join(elements[:-1]),
            elements[-1],
        )


def build_package(
    srcdir: StrPath,
    outdir: StrPath,
    distributions: Sequence[Distribution],
    config_settings: ConfigSettings | None = None,
    isolation: bool = True,
    skip_dependency_check: bool = False,
    installer: _env.Installer = 'pip',
) -> Sequence[str]:
    """
    Run the build process.

    :param srcdir: Source directory
    :param outdir: Output directory
    :param distribution: Distribution to build (sdist or wheel)
    :param config_settings: Configuration settings to be passed to the backend
    :param isolation: Isolate the build in a separate environment
    :param skip_dependency_check: Do not perform the dependency check
    """
    built: list[str] = []
    for distribution in distributions:
        out = _build(isolation, srcdir, outdir, distribution, config_settings, skip_dependency_check, installer)
        built.append(os.path.basename(out))
    return built


def build_package_via_sdist(
    srcdir: StrPath,
    outdir: StrPath,
    distributions: Sequence[Distribution],
    config_settings: ConfigSettings | None = None,
    isolation: bool = True,
    skip_dependency_check: bool = False,
    installer: _env.Installer = 'pip',
) -> Sequence[str]:
    """
    Build a sdist and then the specified distributions from it.

    :param srcdir: Source directory
    :param outdir: Output directory
    :param distribution: Distribution to build (only wheel)
    :param config_settings: Configuration settings to be passed to the backend
    :param isolation: Isolate the build in a separate environment
    :param skip_dependency_check: Do not perform the dependency check
    """
    from ._compat import tarfile

    if 'sdist' in distributions:
        msg = 'Only binary distributions are allowed but sdist was specified'
        raise ValueError(msg)

    sdist = _build(isolation, srcdir, outdir, 'sdist', config_settings, skip_dependency_check, installer)

    sdist_name = os.path.basename(sdist)
    sdist_out = tempfile.mkdtemp(prefix='build-via-sdist-')
    built: list[str] = []
    if distributions:
        # extract sdist
        with tarfile.TarFile.open(sdist) as t:
            t.extractall(sdist_out)
            try:
                _ctx.log(f'Building {_natural_language_list(distributions)} from sdist')
                srcdir = os.path.join(sdist_out, sdist_name[: -len('.tar.gz')])
                for distribution in distributions:
                    out = _build(isolation, srcdir, outdir, distribution, config_settings, skip_dependency_check, installer)
                    built.append(os.path.basename(out))
            finally:
                shutil.rmtree(sdist_out, ignore_errors=True)
    return [sdist_name, *built]


def main_parser() -> argparse.ArgumentParser:
    """
    Construct the main parser.
    """
    parser = argparse.ArgumentParser(
        description=textwrap.indent(
            textwrap.dedent(
                """
                A simple, correct Python build frontend.

                By default, a source distribution (sdist) is built from {srcdir}
                and a binary distribution (wheel) is built from the sdist.
                This is recommended as it will ensure the sdist can be used
                to build wheels.

                Pass -s/--sdist and/or -w/--wheel to build a specific distribution.
                If you do this, the default behavior will be disabled, and all
                artifacts will be built from {srcdir} (even if you combine
                -w/--wheel with -s/--sdist, the wheel will be built from {srcdir}).
                """
            ).strip(),
            '    ',
        ),
        # Prevent argparse from taking up the entire width of the terminal window
        # which impedes readability.
        formatter_class=partial(argparse.RawDescriptionHelpFormatter, width=min(_max_terminal_width, 127)),
    )
    parser.add_argument(
        'srcdir',
        type=str,
        nargs='?',
        default=os.getcwd(),
        help='source directory (defaults to current directory)',
    )
    parser.add_argument(
        '--version',
        '-V',
        action='version',
        version=f"build {build.__version__} ({','.join(build.__path__)})",
    )
    parser.add_argument(
        '--verbose',
        '-v',
        dest='verbosity',
        action='count',
        default=0,
        help='increase verbosity',
    )
    parser.add_argument(
        '--sdist',
        '-s',
        dest='distributions',
        action='append_const',
        const='sdist',
        help='build a source distribution (disables the default behavior)',
    )
    parser.add_argument(
        '--wheel',
        '-w',
        dest='distributions',
        action='append_const',
        const='wheel',
        help='build a wheel (disables the default behavior)',
    )
    parser.add_argument(
        '--outdir',
        '-o',
        type=str,
        help=f'output directory (defaults to {{srcdir}}{os.sep}dist)',
        metavar='PATH',
    )
    parser.add_argument(
        '--skip-dependency-check',
        '-x',
        action='store_true',
        help='do not check that build dependencies are installed',
    )
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        '--no-isolation',
        '-n',
        action='store_true',
        help='disable building the project in an isolated virtual environment. '
        'Build dependencies must be installed separately when this option is used',
    )
    env_group.add_argument(
        '--installer',
        choices=_env.INSTALLERS,
        help='Python package installer to use (defaults to pip)',
    )
    parser.add_argument(
        '--config-setting',
        '-C',
        dest='config_settings',
        action='append',
        help='settings to pass to the backend.  Multiple settings can be provided. '
        'Settings beginning with a hyphen will erroneously be interpreted as options to build if separated '
        'by a space character; use ``--config-setting=--my-setting -C--my-other-setting``',
        metavar='KEY[=VALUE]',
    )
    return parser


def main(cli_args: Sequence[str], prog: str | None = None) -> None:
    """
    Parse the CLI arguments and invoke the build process.

    :param cli_args: CLI arguments
    :param prog: Program name to show in help text
    """
    parser = main_parser()
    if prog:
        parser.prog = prog
    args = parser.parse_args(cli_args)

    _setup_cli(verbosity=args.verbosity)

    config_settings = {}

    if args.config_settings:
        for arg in args.config_settings:
            setting, _, value = arg.partition('=')
            if setting not in config_settings:
                config_settings[setting] = value
            else:
                if not isinstance(config_settings[setting], list):
                    config_settings[setting] = [config_settings[setting]]

                config_settings[setting].append(value)

    # outdir is relative to srcdir only if omitted.
    outdir = os.path.join(args.srcdir, 'dist') if args.outdir is None else args.outdir

    distributions: list[Distribution] = args.distributions
    if distributions:
        build_call = build_package
    else:
        build_call = build_package_via_sdist
        distributions = ['wheel']

    with _handle_build_error():
        built = build_call(
            args.srcdir,
            outdir,
            distributions,
            config_settings,
            not args.no_isolation,
            args.skip_dependency_check,
            args.installer,
        )
        artifact_list = _natural_language_list(
            ['{underline}{}{reset}{bold}{green}'.format(artifact, **_styles.get()) for artifact in built]
        )
        _cprint('{bold}{green}Successfully built {}{reset}', artifact_list)


def entrypoint() -> None:
    main(sys.argv[1:])


if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1:], 'python -m build')


__all__ = [
    'main',
    'main_parser',
]
