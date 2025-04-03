from pyreadline3.py3k_compat import is_ironpython

if is_ironpython:
    try:
        from .ironpython_console import *
    except ImportError as x:
        raise ImportError(
            "Could not find a console implementation for local " "ironpython version"
        ) from x
else:
    try:
        from .console import *
    except ImportError as x:
        raise ImportError(
            "Could not find a console implementation for local " "python version"
        ) from x
