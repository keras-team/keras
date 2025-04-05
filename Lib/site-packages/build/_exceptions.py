from __future__ import annotations

import subprocess
import types


class BuildException(Exception):
    """
    Exception raised by :class:`build.ProjectBuilder`.
    """


class BuildBackendException(Exception):
    """
    Exception raised when a backend operation fails.
    """

    def __init__(
        self,
        exception: Exception,
        description: str | None = None,
        exc_info: tuple[type[BaseException], BaseException, types.TracebackType] | tuple[None, None, None] = (
            None,
            None,
            None,
        ),
    ) -> None:
        super().__init__()
        self.exception = exception
        self.exc_info = exc_info
        self._description = description

    def __str__(self) -> str:
        if self._description:
            return self._description
        return f'Backend operation failed: {self.exception!r}'


class BuildSystemTableValidationError(BuildException):
    """
    Exception raised when the ``[build-system]`` table in pyproject.toml is invalid.
    """

    def __str__(self) -> str:
        return f'Failed to validate `build-system` in pyproject.toml: {self.args[0]}'


class FailedProcessError(Exception):
    """
    Exception raised when a setup or preparation operation fails.
    """

    def __init__(self, exception: subprocess.CalledProcessError, description: str) -> None:
        super().__init__()
        self.exception = exception
        self._description = description

    def __str__(self) -> str:
        return self._description


class TypoWarning(Warning):
    """
    Warning raised when a possible typo is found.
    """
