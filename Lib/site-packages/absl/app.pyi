from typing import Any, Callable, Collection, Iterable, List, NoReturn, Optional, TypeVar, Union, overload

from absl.flags import _flag

_MainArgs = TypeVar('_MainArgs')
_Exc = TypeVar('_Exc', bound=Exception)

class ExceptionHandler():

  def wants(self, exc: _Exc) -> bool:
    ...

  def handle(self, exc: _Exc):
    ...

EXCEPTION_HANDLERS: List[ExceptionHandler] = ...

class HelpFlag(_flag.BooleanFlag):
  def __init__(self):
    ...

class HelpshortFlag(HelpFlag):
  ...

class HelpfullFlag(_flag.BooleanFlag):
  def __init__(self):
    ...

class HelpXMLFlag(_flag.BooleanFlag):
  def __init__(self):
    ...

def define_help_flags() -> None:
  ...

@overload
def usage(shorthelp: Union[bool, int] = ...,
          writeto_stdout: Union[bool, int] = ...,
          detailed_error: Optional[Any] = ...,
          exitcode: None = ...) -> None:
  ...

@overload
def usage(shorthelp: Union[bool, int],
          writeto_stdout: Union[bool, int],
          detailed_error: Optional[Any],
          exitcode: int) -> NoReturn:
  ...

@overload
def usage(shorthelp: Union[bool, int] = ...,
          writeto_stdout: Union[bool, int] = ...,
          detailed_error: Optional[Any] = ...,
          *,
          exitcode: int) -> NoReturn:
  ...

def install_exception_handler(handler: ExceptionHandler) -> None:
  ...

class Error(Exception):
  ...

class UsageError(Error):
  exitcode: int

def parse_flags_with_usage(args: List[str]) -> List[str]:
  ...

def call_after_init(callback: Callable[[], Any]) -> None:
  ...

# Without the flag_parser argument, `main` should require a List[str].
@overload
def run(
    main: Callable[[List[str]], Any],
    argv: Optional[List[str]] = ...,
) -> NoReturn:
  ...

@overload
def run(
    main: Callable[[_MainArgs], Any],
    argv: Optional[List[str]] = ...,
    *,
    flags_parser: Callable[[List[str]], _MainArgs],
) -> NoReturn:
  ...
