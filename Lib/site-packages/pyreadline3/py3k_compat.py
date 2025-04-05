import sys
from collections.abc import Callable
from typing import Any, Dict, Optional

is_ironpython = "IronPython" in sys.version


def is_callable(x: Any) -> bool:
    return isinstance(x, Callable)


def execfile(
    fname: str,
    glob: Dict[str, Any],
    loc: Optional[Dict[str, Any]] = None,
) -> None:
    loc = loc if (loc is not None) else glob

    with open(
        fname,
        "r",
        encoding="utf-8",
    ) as file:
        file_contents = file.read()

    # pylint: disable=W0122
    exec(
        compile(
            file_contents,
            fname,
            "exec",
        ),
        glob,
        loc,
    )
